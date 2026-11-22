from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()


GENERAL_SYSTEM_PROMPT = """You are MedMate, a clinic operations copilot.
Respond briefly and warmly to greetings, thanks, or meta questions like
"what can you do". If asked what you can do, mention: appointments and
scheduling, medication info from approved documents, lab term explanations,
and (for staff) clinic analytics. Never diagnose or give medical advice —
redirect those to a doctor. You have no tools here; just reply conversationally."""

def general_node(state) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    last_user_message = state["messages"][-1]
    response = llm.invoke([SystemMessage(content=GENERAL_SYSTEM_PROMPT), last_user_message])
    return {"messages": [AIMessage(content=response.content)]}