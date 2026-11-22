from typing import Optional
from pydantic import BaseModel, Field

class RetrieveChunksInput(BaseModel):
    query:str = Field(..., description="The user's question to retrieve relevant document chunks for")
    k: int = Field(5, description="Number of top chunks to retrieve after reranking")

class RelevanceFilterInput(BaseModel):
    chunks: list[dict] = Field(..., description="Chunks with content/score, from retrieve_chunks output")
    threshold: float = Field(0.3, description="Minimum relevance/rerank score to keep a chunk")

class ExplainLabTermInput(BaseModel):
    term: str = Field(..., description="The lab test or result term to explain, e.g. 'ALT', 'eGFR'")

class AnswerMedicationQuestionInput(BaseModel):
    question: str = Field(..., description="Patient or staff question about a medication from approved documents")


