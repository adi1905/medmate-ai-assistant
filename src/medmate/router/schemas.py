from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Literal

AgentName = Literal["patient_records", "med_lab_qa", "analytics", "prescription_ops","lab_reference", "general"]

class RouterDecision(BaseModel):
    """LLM structured output."""
    agents: list[AgentName] = Field(
        ..., min_length=1,
        description="Ordered agents needed to answer the query. Never empty — use ['general'] if no specialist agent applies."
    )
    reasoning: str = Field(..., description="Internal only, not shown to the user.")


@dataclass
class RoutingPlan:
    """Runtime plan object — not stored directly in MedMateState."""
    agents: list[AgentName]
    reasoning: str
    current_index: int = 0

    @property
    def current_agent(self) -> str | None:
        if self.current_index >= len(self.agents):
            return None
        return self.agents[self.current_index]

    def advance(self) -> "RoutingPlan":
        return RoutingPlan(self.agents, self.reasoning, self.current_index + 1)

    def is_done(self) -> bool:
        return self.current_index >= len(self.agents)
    
class RiskClassification(BaseModel):
    reasoning: str = Field(
        ...,
        description="Brief internal reasoning: identify the actual verb/intent "
                    "of the message (booking, factual question, or genuine "
                    "diagnosis/judgment request) before deciding. Not shown to the user."
    )
    requires_refusal: bool = Field(..., description="True if this is a diagnosis/treatment-seeking query")