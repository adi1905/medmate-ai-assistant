from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from .state import MedMateState
from ..db.repositories.appointment_repo import (insert_appointment, update_appointment_fields,
                                                cancel_appointment_db)
from ..db.repositories.patient_repo import update_patient_contact
from ..db.repositories.prescription_repo import update_prescription
import json

from dotenv import load_dotenv
load_dotenv()

def _extract_pending_action(state: MedMateState) -> dict:
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "tool":
            content = m.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    continue
            if isinstance(content, dict) and content.get("pending_action"):
                action = content["pending_action"]
                action["summary"] = content.get("data", {}).get("summary", "")
                action["changes"] = content.get("data", {}).get("changes")
                return action
    return None


def route_after_confirm(state: MedMateState) -> str:
    """
    The piece that was structurally missing: hitl_confirm_node was wired
    to execute_write_node with an UNCONDITIONAL edge, so a rejection
    (which never sets confirmed_action) crashed execute_write_node with
    KeyError on its very first line. This routes rejections straight
    back to 'agent' instead, where the "Action cancelled by user."
    message already added to state lets the conversation continue
    normally.
    """
    return "execute_write" if state.get("confirmed_action") else "agent"

def _write_result(content: str) -> dict:
    """
    Every return path in execute_write_node goes through this so
    confirmed_action always gets reset to None afterward. Without this,
    a stale confirmed_action from THIS turn would still be truthy the
    NEXT time hitl_confirm_node runs (e.g. on a later, unrelated write
    in the same thread) -- route_after_confirm would then route straight
    to execute_write_node even on a fresh rejection, reintroducing the
    exact same crash from a different angle.
    """
    return {"messages": [AIMessage(content=content)], "confirmed_action": None}


def _generate_confirmation_question(pending: dict) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "Write a single short, natural confirmation question for a user, "
        "based on this pending action. Be specific about what is changing, "
        "using the 'changes' old/new values if present. Do not add extra "
        "commentary, just the question.\n\n"
        f"Action: {json.dumps(pending)}"
    )
    response = llm.invoke(prompt)
    return response.content

def hitl_confirm_node(state: MedMateState):
    pending = _extract_pending_action(state)
    question = _generate_confirmation_question(pending)
    decision = interrupt({"question": question, "action": pending})
 
    
    if isinstance(decision, dict):
        approved = decision.get("decision") == "approve"
        edited_args = decision.get("edited_args")
    else:
        approved = decision == "approve"
        edited_args = None
 
    if not approved:
        return {
            "pending_action": None,
            "confirmed_action": None,  # explicit reset -- see execute_write_node's comment below
            "messages": [AIMessage(content="Action cancelled by user.")],
        }
 
    if edited_args:
        pending = {**pending, "params": {**pending["params"], **edited_args}}
 
    return {"confirmed_action": pending}


def execute_write_node(state: MedMateState):
    action = state["confirmed_action"]
    if action["action"] == "create_appointment":
        appt_id = insert_appointment(**action["params"])
        return _write_result(f"Appointment {appt_id} created.")
 
    elif action["action"] == "update_appointment":
        params = action["params"]
        appt_id = params["appointment_id"]
        updated = update_appointment_fields(
            appt_id,
            scheduled_time=params.get("scheduled_time"),
            provider_id=params.get("provider_id"),
            reason=params.get("reason"),
        )
        if updated:
            return _write_result(f"Appointment {appt_id} updated.")
        return _write_result(f"Failed to update appointment {appt_id}.")
 
    elif action["action"] == "cancel_appointment":
        appt_id = action["params"]["appointment_id"]
        cancelled = cancel_appointment_db(appt_id)
        if cancelled:
            return _write_result(f"Appointment {appt_id} cancelled.")
        return _write_result(f"Failed to cancel {appt_id} — it may have already changed status.")
 
    elif action["action"] == "update_patient_contact_info":
        params = action["params"]
        patient_id = params["patient_id"]
        try:
            updated_record = update_patient_contact(
                patient_id,
                phone=params.get("phone"),
                email=params.get("email"),
                address=params.get("address"),
                insurance_provider=params.get("insurance_provider"),
            )
            return _write_result(
                f"Your contact info has been updated — "
                f"Phone: {updated_record['phone']}, Email: {updated_record['email']}, "
                f"Address: {updated_record['address']}, "
                f"Insurance: {updated_record['insurance_provider']}."
            )
        except ValueError as e:
            return _write_result(f"Failed to update contact info: {e}")
 
    elif action["action"] == "log_medication_change":
        params = action["params"]
        prescription_id = params["prescription_id"]
        try:
            updated_record_pres = update_prescription(
                prescription_id,
                dosage=params.get("dosage"),
                end_date=params.get("end_date"),
                status=params.get("status"),
            )
            return _write_result(
                f"Your prescription info has been updated — "
                f"Dosage: {updated_record_pres['dosage']}, "
                f"End Date: {updated_record_pres['end_date']}, "
                f"Status: {updated_record_pres['status']}."
            )
        except ValueError as e:
            return _write_result(f"Failed to update prescription: {e}")
 
    return _write_result("Unknown action; nothing executed.")