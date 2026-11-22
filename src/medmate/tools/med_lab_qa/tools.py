from langchain.tools import tool
from ...rag.pipeline import retriever, embedding_service, generator
from ...rag.helpers.generation.schemas import RetrievedDocument
from .schemas import (
    RetrieveChunksInput,
    RelevanceFilterInput,
    ExplainLabTermInput,
    AnswerMedicationQuestionInput,
)
from ..common.tools import ToolResult

NO_INFO_MESSAGE = (
    "I don't have information on that in the approved documents. "
    "Please consult your doctor or pharmacist."
)

LAB_TERM_DISCLAIMER = (
    "\n\nThis is general educational information only, not an interpretation "
    "of your specific results. Please discuss your lab results with your doctor."
)

@tool(args_schema=RetrieveChunksInput)
def retrieve_chunks(query: str, k: int = 5) -> dict:
    """Retrieve top-k relevant document chunks (with scores) for a query, via
    dense embedding + hybrid retrieval + Cohere reranking. Does not generate an answer —
    use relevance_filter and answer_medication_question / explain_lab_term for that."""
    try:
        query_dense = embedding_service.embed_query(query)
        context = retriever.retrieve(query=query, query_dense_vector= query_dense)
        chunks = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "section": doc.metadata.get("header_path_str"),
                "page_start": doc.metadata.get("page_start"),
                "page_end": doc.metadata.get("page_end"),
                "element_type": doc.metadata.get("element_type"),
                "score": doc.metadata.get("score"),
                "search_type": doc.metadata.get("search_type"),
            }
            for doc in context.chunks[:k]
        ]
        return ToolResult(success=True, data={"chunks": chunks}).model_dump()
    except Exception as e:
        return ToolResult(success=False, error=f"Retrieval failed: {e}").model_dump()
    

@tool(args_schema=RelevanceFilterInput)
def relevance_filter(chunks: list[dict], threshold: float = 0.3) -> dict:
    """Filter retrieved chunks by relevance score, dropping anything below threshold.
    If nothing clears the bar, returns an empty list — the caller should refuse to answer
    rather than generate from weak/irrelevant context."""
    try:
        filtered = [c for c in chunks if (c.get("score") or 0) >= threshold]
        return ToolResult(
            success=True,
            data={"chunks": filtered, "kept": len(filtered), "dropped": len(chunks) - len(filtered)},
        ).model_dump()
    except Exception as e:
        return ToolResult(success=False, error=f"Filtering failed: {e}").model_dump()
    

def _generate_from_chunks(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    documents = [
        RetrievedDocument(
            content=c["content"],
            source=c.get("source"),
            section=c.get("section"),
            page_start=c.get("page_start"),
            page_end=c.get("page_end"),
            element_type=c.get("element_type"),
        )
        for c in chunks
    ]
    response = generator.generate(question=question, documents=documents)
    sources = [s.model_dump() for s in response.sources]
    return response.answer, sources

@tool(args_schema=AnswerMedicationQuestionInput)
def answer_medication_question(question: str) -> dict:
    """Answer a medication information question using approved package-insert and clinic
    FAQ documents. Retrieves, filters by relevance, and refuses if nothing relevant is found."""
    try:
        retrieved = retrieve_chunks.invoke({"query": question, "k": 5})
        if not retrieved["success"]:
            return ToolResult(success=False, error=retrieved["error"]).model_dump()

        filtered = relevance_filter.invoke({"chunks": retrieved["data"]["chunks"], "threshold": 0.3})
        kept = filtered["data"]["chunks"]

        if not kept:
            return ToolResult(success=True, data={"answer": NO_INFO_MESSAGE, "sources": [], "grounded": False}).model_dump()

        answer, sources = _generate_from_chunks(question, kept)
        return ToolResult(success=True, data={"answer": answer, "sources": sources, "grounded": True}).model_dump()
    except Exception as e:
        return ToolResult(success=False, error=f"Could not answer: {e}").model_dump()

@tool(args_schema=ExplainLabTermInput)
def explain_lab_term(term: str) -> dict:
    """Explain a lab test or result term using the approved lab glossary documents.
    Always appends an informational-only disclaimer. Never interprets a patient's specific results."""
    try:
        question = f"What does the lab term '{term}' mean?"
        retrieved = retrieve_chunks.invoke({"query": question, "k": 5})
        if not retrieved["success"]:
            return ToolResult(success=False, error=retrieved["error"]).model_dump()

        filtered = relevance_filter.invoke({"chunks": retrieved["data"]["chunks"], "threshold": 0.3})
        kept = filtered["data"]["chunks"]

        if not kept:
            return ToolResult(
                success=True,
                data={"answer": f"I don't have a glossary entry for '{term}'. Please ask your provider.", "sources": []},
            ).model_dump()
        
        answer, sources = _generate_from_chunks(question, kept)
        return ToolResult(success=True, data={"answer": answer + LAB_TERM_DISCLAIMER, "sources": sources}).model_dump()
    except Exception as e:
        return ToolResult(success=False, error=f"Could not explain term: {e}").model_dump()