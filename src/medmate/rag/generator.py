from langchain_openai import ChatOpenAI

from langchain_core.prompts import (
    ChatPromptTemplate,
)

from langchain_core.documents import Document

from langsmith import traceable

from .helpers.generation.prompts import (
    RAG_PROMPT,
)

from .helpers.generation.models import (
    GenerationResponse,
    AnswerSource,
    FinalAnswer,
)

from .helpers.generation.context_formatter import (
    ContextFormatter,
)


class RAGGenerator:

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0,
    ):

        self.formatter = ContextFormatter()

        self.llm = (
            ChatOpenAI(
                model=model_name,
                temperature=temperature,
            )
            .with_structured_output(
                GenerationResponse
            )
        )

        self.prompt = (
            ChatPromptTemplate.from_template(
                RAG_PROMPT
            )
        )

        self.chain = (
            self.prompt
            | self.llm
        )

    @traceable(name="generate_answer", tags=["generation", "rag"])
    def generate(
        self,
        question: str,
        documents: list,
    ) -> FinalAnswer:

        context, citation_map = (
            self.formatter.format(
                documents
            )
        )

        llm_response: GenerationResponse = (
            self.chain.invoke(
                {
                    "question": question,
                    "context": context,
                }
            )
        )

        sources = []

        for citation in llm_response.citations:
            doc = citation_map.get(citation)
            if not doc:
                continue
            
            # Handle both LangChain Document and RetrievedDocument objects
            if isinstance(doc, Document):
                # Extract from LangChain Document metadata
                source = AnswerSource(
                    source=doc.metadata.get("source", "Unknown"),
                    section=doc.metadata.get("section"),
                    page_start=doc.metadata.get("page_start"),
                    page_end=doc.metadata.get("page_end"),
                )
            else:
                # Assume RetrievedDocument
                source = AnswerSource(
                    source=doc.source,
                    section=doc.section,
                    page_start=doc.page_start,
                    page_end=doc.page_end,
                )
            
            sources.append(source)

        return FinalAnswer(
            answer=llm_response.answer,
            sources=sources,
        )