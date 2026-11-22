from abc import ABC, abstractmethod
from langchain_core.documents import Document
from .parsed_document import ParsedDocument


class BaseChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        parsed_document: ParsedDocument,
    ) -> list[Document]:
        pass