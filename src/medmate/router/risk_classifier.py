from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .schemas import RiskClassification
from ..graph.state import MedMateState

REFUSAL_MESSAGE = (
    "I'm not able to help with diagnosis or treatment decisions. "
    "Please consult your doctor or a qualified healthcare provider for that. "
    "I can help with appointments, medication information from approved documents, "
    "lab term explanations, or clinic records."
)

RISK_SYSTEM_PROMPT = """You are a safety classifier for a clinic operations copilot.
Your ONLY job is to detect whether the user's message is seeking a diagnosis,
treatment advice, or a personal medical judgment call — NOT to answer it.

Classify as risky (requires_refusal=true) if the user is asking:
- "What's wrong with me" / "do I have X condition" / "is this serious"
- "What should I take for X symptom" / "should I take medication Y for my condition"
- "Interpret my lab result" / "what does my result mean for my health"
- Any request for a clinical judgment specific to their own body/condition

Classify as NOT risky (requires_refusal=false) if the user is asking:
- Factual/definitional questions about a medication or lab term in general
  (e.g. "what is metformin used for", "what does ALT measure")
- Appointment scheduling, records, clinic operations, analytics
- Anything that doesn't ask the system to make a judgment about their
  personal health status

When uncertain, err toward requires_refusal=false and let the downstream
agent's own scope guardrails handle it — this classifier is for clear-cut
diagnosis/treatment-seeking intent only, not a general content filter.

IMPORTANT DISTINCTION: A patient stating a symptom as the REASON for booking
an appointment is NOT risky, even though it mentions symptoms. Only classify
as risky if the patient is asking YOU to interpret, diagnose, or advise on
that symptom.

Examples of NOT risky (requires_refusal=false):
- "I want to book with Dr. Montgomery for my stomach problem, I've been having digestion issues"
  -> This is a reason-for-visit statement attached to a scheduling request. Not risky.
- "book me an appointment, I have a headache"
  -> Reason for visit. Not risky.

Examples of risky (requires_refusal=true):
- "I have digestion issues, what do you think it could be?"
- "I have a headache, should I be worried?"

The key signal is the VERB/ASK: is the patient asking to schedule/see someone
about the symptom (not risky), or asking the assistant itself to judge,
diagnose, or advise on the symptom (risky)?
"""


def classify_intent_risk(state: MedMateState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(RiskClassification)

    last_user_message = state["messages"][-1].content
    recent = state["messages"][-6:]  # last few turns for context
    convo_text = "\n".join(
        f"{'Patient' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in recent if m.content
    )

    result = structured_llm.invoke([
        SystemMessage(content=RISK_SYSTEM_PROMPT),
        HumanMessage(content=f"Conversation so far:\n{convo_text}\n\nClassify the LATEST patient message."),
    ])

    print(f"[RISK_CLASSIFIER] input={last_user_message!r}")
    print(f"[RISK_CLASSIFIER] requires_refusal={result.requires_refusal} reasoning={result.reasoning!r}")

    if result.requires_refusal:
        return {"refused": True, "messages": [AIMessage(content=REFUSAL_MESSAGE)]}

    return {"refused": False}


def refusal_condition(state: MedMateState) -> str:
    return "end" if state.get("refused") else "router"