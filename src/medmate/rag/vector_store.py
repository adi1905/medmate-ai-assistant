from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct, SparseVector, 
                                  SparseVectorParams, SparseIndexParams, 
                                  FieldCondition, MatchValue, Filter, Modifier)
from langchain_core.documents import Document
from fastembed import SparseTextEmbedding
from uuid import uuid4
from typing import Optional
from config.settings import BM25_MODEL, QDRANT_PATH
import logging


logger = logging.getLogger(__name__)

#Hybrid Search Knowledge
# BM25    = exact words only
# SPLADE  = exact words + learned expansions
# Dense   = semantic meaning



 
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantService:
    """
    Qdrant storage layer — collection management and document ingestion only.
    
    Dense vectors  → supplied externally (OpenAI, 1536-dim)
    Sparse vectors → computed internally via BM25 (fastembed)
    
    Uses singleton pattern to reuse QdrantClient across pipelines,
    preventing file locking issues with local storage.
    """
    
    # Class-level cache for client instances (path -> client)
    _client_cache = {}
    
    def __init__(self,
                 collection_name: str,
                 qdrant_path: str = QDRANT_PATH,
                 sparse_model: str = BM25_MODEL,
                 dense_vector_size: int = 1536,
                 ):
        self.collection_name = collection_name
        self.dense_vector_size = dense_vector_size
        
        # Get or create client from cache (singleton per path)
        if qdrant_path not in QdrantService._client_cache:
            logger.info(f"Creating new Qdrant client for path: {qdrant_path}")
            QdrantService._client_cache[qdrant_path] = QdrantClient(path=qdrant_path)
        
        self.client = QdrantService._client_cache[qdrant_path]
        
        self._sparse_encoder: Optional[SparseTextEmbedding] = None
        self._sparse_model = sparse_model
        logger.debug(f"QdrantService initialized for collection '{collection_name}' with dense vector size {dense_vector_size}")

    @property
    def sparse_encoder(self) -> SparseTextEmbedding:
        if self._sparse_encoder is None:
            try:
                logger.info(f"Loading sparse encoder: {self._sparse_model}")
                self._sparse_encoder = SparseTextEmbedding(
                    model_name=self._sparse_model
                )
                logger.info(f"Sparse encoder loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sparse model '{self._sparse_model}': {e}")
                raise RuntimeError(
                    f"Sparse model initialization failed with model '{self._sparse_model}'. "
                    f"Ensure SPLADE_MODEL env var is set correctly and model is available. "
                    f"Error: {e}"
                )
        return self._sparse_encoder
    
    def encode_sparse(self, texts : list[str]) -> list[SparseVector]:
        sparse_vecs = []
        for sv in self.sparse_encoder.embed(texts):
            sparse_vecs.append(
                SparseVector(
                    indices=sv.indices.tolist(),
                    values=sv.values.tolist(),
                )   
            )
        return sparse_vecs

    def create_collection(
        self,
        vector_size: int = None,
    ):
        if vector_size is None:
            vector_size = self.dense_vector_size
        
        logger.info(f"Creating collection '{self.collection_name}' with vector_size={vector_size}")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    modifier=Modifier.IDF
                )
            }
        )

        for field, schema in [
            ("source", "keyword"),
            ("element_type", "keyword"),
            ("section", "keyword"),
            ("group_id", "keyword"),      # add this
            ("page_start", "integer"),    # add this
            ("page", "integer"),
        ]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
            )

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)
    

    def fetch_by_group_id(
        self,
        group_id: str,
        source: Optional[str] = None,
    ) -> list[Document]:
        """Fetch all chunks belonging to the same section group."""
        conditions = [
            FieldCondition(key="group_id", match=MatchValue(value=group_id))
        ]
        if source:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source))
            )
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=conditions),
            limit=50,  # max chunks per section group
            with_payload=True,
            with_vectors=False,
        )
        docs = []
        for point in results[0]:
            payload = point.payload.copy()
            content = payload.pop("content", "")
            docs.append(Document(page_content=content, metadata=payload))

        # Return in chunk order
        docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        return docs
    
    def upsert_documents(self,
                      chunks : list[Document],
                      dense_vectors: list[list[float]],) -> int:
        """
        Upsert documents with both dense and sparse vectors.
        
        Args:
            chunks: List of documents to upsert
            dense_vectors: List of dense embeddings (OpenAI)
            
        Returns:
            Number of documents stored
        """
        if len(chunks)!=len(dense_vectors):
            raise ValueError(
                "Chunks and vectors length mismatch"
            )
        texts_for_sparse = [self._sparse_text(c) for c in chunks]
        sparse_vectors = self.encode_sparse(texts_for_sparse)


        points = []
        for chunk, dense_vec, sparse_vec in zip(
            chunks, dense_vectors, sparse_vectors
        ):
            payload = {
                **chunk.metadata,
                "content":chunk.page_content,
            }
            point_id = chunk.metadata.get(
                "chunk_id",
                str(uuid4()),
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                            DENSE_VECTOR_NAME:dense_vec,
                            SPARSE_VECTOR_NAME:sparse_vec
                        },
                    payload=payload,
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points = points,
            wait=True)
        
        return len(points)
        
    # def search(self, query_vector: list[float], limit: int = 5) -> list[Document]:
    #     results = self.client.query_points(
    #         collection_name=self.collection_name,
    #         query=query_vector,
    #         limit=limit,
    #     )

    #     documents = []
    #     for result in results.points:
    #         payload = result.payload.copy()
    #         content = payload.pop("content", "")
    #         documents.append(Document(
    #             page_content=content,
    #             metadata=payload,
    #         ))

    #     return documents
    
    def collection_info(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        return {
            "status": info.status,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
        }
    
    def _sparse_text(self, chunk : Document) -> str:
        if chunk.metadata.get("element_type")!="table":
            return chunk.page_content
        parts = []
        section = chunk.metadata.get("section", "")
        if section:
            parts.append(f"Section: {section}")
        preceding = chunk.metadata.get("preceding_text", "")
        if preceding:
            parts.append(f"Context: {preceding}")
        parts.append(chunk.page_content)
        return "\n".join(parts)
    
    def delete_collection(self):

        self.client.delete_collection(
            collection_name=self.collection_name
        )
        
            
        