from pydantic import BaseModel, Field


class GenerationResponse(BaseModel):
    answer: str = Field(
        description="Answer generated from the retrieved documents."
    )

    citations: list[str] = Field(
        default_factory=list,
        description="Document IDs supporting the answer."
    )


class AnswerSource(BaseModel):
    source: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class FinalAnswer(BaseModel):
    answer: str
    sources: list[AnswerSource]