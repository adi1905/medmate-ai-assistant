from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    content: str
    source: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    element_type: str | None = None