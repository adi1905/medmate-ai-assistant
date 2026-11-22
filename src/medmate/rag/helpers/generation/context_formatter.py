from .schemas import RetrievedDocument
from langchain_core.documents import Document


class ContextFormatter:

    def format(
        self,
        docs: list,
    ) -> tuple[str, dict]:
        context_parts = []
        citation_map = {}
        for idx, doc in enumerate(docs, start=1):
            doc_id = f"DOC_{idx}"
            citation_map[doc_id] = doc
            
            # Handle both LangChain Document and RetrievedDocument objects
            if isinstance(doc, Document):
                # Extract from LangChain Document metadata
                content = doc.page_content
                section = doc.metadata.get("section", "Unknown")
                element_type = doc.metadata.get("element_type", "text")
            else:
                # Assume RetrievedDocument
                content = doc.content
                section = doc.section or "Unknown"
                element_type = doc.element_type or "text"
            
            part = f"""
[{doc_id}]

Section: {section}
Type: {element_type}

Content:
{content}
"""

            context_parts.append(part)

        context = "\n\n----------------------\n\n".join(
            context_parts
        )

        return context, citation_map