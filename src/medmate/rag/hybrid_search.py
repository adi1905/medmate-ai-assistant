from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    NamedVector, NamedSparseVector,
    Prefetch, FusionQuery, Fusion
)
from dataclasses import dataclass
from langchain_core.documents import Document
from typing import Optional
import logging
from langsmith import traceable

from .vector_store import QdrantService, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    document: Document
    score: float
    search_type: str

    def __repr__(self):
        source = self.document.metadata.get("source", "?")
        page = self.document.metadata.get("page_start", "?")
        element_type = self.document.metadata.get("element_type", "?")
        
        preview = self.document.page_content[:80].replace("\n", " ")
        section = self.document.metadata.get("section", "?")
        return (
            f"SearchResult(score={self.score:.4f}, type={self.search_type}, "
            f"source={source}, page={page}, element={element_type}, "
            f"section={section}, preview={preview!r})"
        )


@dataclass
class RetrievedContext:
    """Final context ready to pass to LLM."""
    chunks: list[Document]
    formatted_context: str
    total_chars: int


class HybridRetriever:

    # How many chars we allow in the final LLM context
    MAX_CONTEXT_CHARS = 6000

    def __init__(self, qdrant_service: QdrantService, reranker=None,):
        self.store = qdrant_service
        self.reranker = reranker

    # ==========================================================
    # FILTER BUILDER
    # ==========================================================

    def _build_filter(
        self,
        source: Optional[str] = None,
        element_type: Optional[str] = None,
        section: Optional[str] = None,
        page: Optional[int] = None,
    ) -> Optional[Filter]:
        conditions = []
        if source:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
        if element_type:
            conditions.append(FieldCondition(key="element_type", match=MatchValue(value=element_type)))
        if section:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))
        if page is not None:
            conditions.append(FieldCondition(key="page", match=MatchValue(value=page)))
        return Filter(must=conditions) if conditions else None

    # ==========================================================
    # SEARCH METHODS (unchanged)
    # ==========================================================

    def search(
        self,
        query: str,
        query_dense_vector: list[float],
        limit: int = 5,
        prefetch_limit: int = 20,
        source: Optional[str] = None,
        element_type: Optional[str] = None,
        section: Optional[str] = None,
        page: Optional[int] = None,
    ) -> list[SearchResult]:
        query_filter = self._build_filter(source, element_type, section, page)
        sparse_vec = self.store.encode_sparse([query])[0]

        results = self.store.client.query_points(
            collection_name=self.store.collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_vec,
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return self._to_results(results.points, search_type="hybrid")

    def search_dense_only(
        self,
        query_dense_vector: list[float],
        limit: int = 5,
        source: Optional[str] = None,
        element_type: Optional[str] = None,
        section: Optional[str] = None,
        page: Optional[int] = None,
    ) -> list[SearchResult]:
        query_filter = self._build_filter(source, element_type, section, page)
        results = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=query_dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return self._to_results(results.points, search_type="dense")

    def search_sparse_only(
        self,
        query: str,
        limit: int = 5,
        source: Optional[str] = None,
        element_type: Optional[str] = None,
        section: Optional[str] = None,
        page: Optional[int] = None,
    ) -> list[SearchResult]:
        sparse_vec = self.store.encode_sparse([query])[0]
        query_filter = self._build_filter(source, element_type, section, page)
        results = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=sparse_vec,
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return self._to_results(results.points, search_type="sparse")

    # ==========================================================
    # SIBLING EXPANSION
    # ==========================================================

    def _expand_with_siblings(
        self,
        results: list[SearchResult],
        source: Optional[str] = None,
    ) -> list[SearchResult]:

        seen_group_ids: set[str] = set()
        expanded: list[SearchResult] = []

        for result in results:

            meta = result.document.metadata

            group_id = meta.get("group_id")
            chunk_total = meta.get("chunk_total", 1)

            if (
                group_id
                and chunk_total > 1
                and group_id not in seen_group_ids
            ):
                seen_group_ids.add(group_id)

                siblings = self.store.fetch_by_group_id(
                    group_id,
                    source=source,
                )

                for sibling in siblings:
                    expanded.append(
                        SearchResult(
                            document=sibling,
                            score=result.score,
                            search_type=result.search_type,
                        )
                    )

            else:
                expanded.append(result)

        return expanded

    # ==========================================================
    # SIBLING EXPANSION
    # ==========================================================

    def _assemble_context(
        self,
        results: list[SearchResult],
    ) -> RetrievedContext:
        """
        Assemble final context from search results, respecting character limit.
        
        Note: Formatting is handled by the Generator, not here.
        This method only selects and filters documents.
        """
        selected_docs: list[Document] = []
        used_chars = 0
        
        for result in results:
            doc = result.document
            # Estimate chars for this doc (content + metadata)
            estimated_chars = len(doc.page_content) + 100
            
            if used_chars + estimated_chars > self.MAX_CONTEXT_CHARS:
                logger.debug(
                    f"Context limit reached: {used_chars} chars. "
                    f"Stopping at {len(selected_docs)} documents."
                )
                break
            doc.metadata["score"] = result.score
            doc.metadata["search_type"] = result.search_type
            
            selected_docs.append(doc)
            used_chars += estimated_chars

        logger.info(
            f"Selected {len(selected_docs)} documents for context "
            f"({used_chars} chars, max {self.MAX_CONTEXT_CHARS})"
        )

        return RetrievedContext(
            chunks=selected_docs,
            formatted_context="",  # Formatting happens in Generator
            total_chars=used_chars,
        )

    # ==========================================================
    # MAIN RETRIEVE METHOD
    # ==========================================================

    @traceable(name="retrieve_documents", tags=["retrieval", "hybrid_search"])
    def retrieve(
        self,
        query: str,
        query_dense_vector: list[float],
        limit: int = 5,
        prefetch_limit: int = 20,
        source: Optional[str] = None,
        element_type: Optional[str] = None,
        section: Optional[str] = None,
        page: Optional[int] = None,
        expand_siblings: bool = False,
    ) -> RetrievedContext:

        # ==========================================
        # 1. RETRIEVE MORE CANDIDATES
        # ==========================================

        search_limit = max(limit * 4, 20)

        results = self.search(
            query=query,
            query_dense_vector=query_dense_vector,
            limit=search_limit,
            prefetch_limit=prefetch_limit,
            source=source,
            element_type=element_type,
            section=section,
            page=page,
        )

        logger.debug(f"Search completed with {len(results)} results")
        for i, result in enumerate(results, 1):
            logger.debug(
                f"Result {i}: score={result.score:.4f} "
                f"section={result.document.metadata.get('section')}"
            )

        # ==========================================
        # 2. RERANK
        # ==========================================

        if self.reranker:
            try:
                logger.info(f"Reranking {len(results)} results with {self.reranker.__class__.__name__}")
                results = self.reranker.rerank(
                    query=query,
                    results=results,
                    top_k=limit,
                )
                logger.debug(f"Reranking completed, top {len(results[:limit])} selected")
                for i, result in enumerate(results, 1):
                    logger.debug(
                        f"Reranked result {i}: score={result.score:.4f} "
                        f"section={result.document.metadata.get('section')}"
                    )
            except Exception as e:
                logger.warning(f"Reranking failed, using search results: {e}")

        # ==========================================
        # 3. EXPAND ONLY TOP RERANKED RESULTS
        # ==========================================

        if expand_siblings:
            try:
                logger.info("Expanding with sibling chunks")
                results = self._expand_with_siblings(
                    results,
                    source=source,
                )
                logger.debug(f"Sibling expansion completed, {len(results)} total chunks")
                for i, result in enumerate(results, 1):
                    logger.debug(
                        f"Expanded result {i}: score={result.score:.4f} "
                        f"section={result.document.metadata.get('section')} "
                        f"chunk={result.document.metadata.get('chunk_index')}"
                    )
            except Exception as e:
                logger.warning(f"Sibling expansion failed: {e}. Continuing without expansion.")
                # Continue without expansion if it fails

        # ==========================================
        # 4. BUILD CONTEXT
        # ==========================================

        return self._assemble_context(results)
    # ==========================================================
    # HELPERS
    # ==========================================================

    def _to_results(self, points, search_type: str) -> list[SearchResult]:
        results = []
        for point in points:
            payload = point.payload.copy()
            content = payload.pop("content", "")
            results.append(SearchResult(
                document=Document(page_content=content, metadata=payload),
                score=point.score,
                search_type=search_type,
            ))
        return results