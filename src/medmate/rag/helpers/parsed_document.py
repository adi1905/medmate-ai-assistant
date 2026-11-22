from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class ParsedDocument:
    documents: list[Document]
    source_type: str
    image_paths: list[str] | None = None