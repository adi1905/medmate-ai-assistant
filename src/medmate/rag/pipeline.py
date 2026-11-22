from .embeddings import EmbeddingService
from .hybrid_search import HybridRetriever
from .helpers.generation.schemas import RetrievedDocument
from .generator import RAGGenerator
from .vector_store import QdrantService
from .reranker import CohereRanker
from dotenv import load_dotenv

load_dotenv()

# ---- Initialize once, reused across all queries ----
qdrant = QdrantService(collection_name="medmate_rag_docs")
embedding_service = EmbeddingService()
cohere_reranker = CohereRanker()
retriever = HybridRetriever(qdrant, cohere_reranker)
generator = RAGGenerator()


def run_rag_query(query: str):
    """
    Full pipeline: embed -> hybrid retrieve + rerank -> generate.
    Returns (answer, contexts, sources) for eval consumption.
    """
    query_dense = embedding_service.embed_query(query)

    context = retriever.retrieve(query=query, query_dense_vector=query_dense)

    documents = []
    for chunk in context.chunks:
        metadata = chunk.metadata
        documents.append(
            RetrievedDocument(
                content=chunk.page_content,
                source=metadata.get("source"),
                section=metadata.get("header_path_str"),
                page_start=metadata.get("page_start"),
                page_end=metadata.get("page_end"),
                element_type=metadata.get("element_type"),
            )
        )

    response = generator.generate(question=query, documents=documents)

    contexts = [doc.page_content for doc in context.chunks]
    sources = [s.model_dump() for s in response.sources]

    return response.answer, contexts, sources