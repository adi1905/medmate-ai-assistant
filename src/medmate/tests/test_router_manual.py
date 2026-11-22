# medmate/tests/test_router_manual.py
from ..router.router import route_query
from ..graph.state import MedMateState
from langchain_core.messages import HumanMessage

TEST_CASES = [
    # (query, role, expected_agents)
    ("What's my next appointment?", "patient", ["patient_records"]),
    ("What is metformin used for?", "patient", ["med_lab_qa"]),
    ("What does ALT mean in a lab report?", "patient", ["med_lab_qa"]),
    ("What's our no-show rate this month?", "staff", ["analytics"]),
    ("What's our no-show rate this month?", "patient", []),  # should NOT be analytics
    ("Reschedule my appointment and tell me what ibuprofen is for", "patient",
     ["patient_records", "med_lab_qa"]),
    ("Show me top prescribed medications and chart it", "staff", ["analytics"]),
]


def run_router_tests():
    passed, failed = 0, 0
    for query, role, expected in TEST_CASES:
        state: MedMateState = {
            "messages": [HumanMessage(content=query)],
            "role": role,
            "user_id": "TEST-001",
        }
        result = route_query(state)
        actual = result["plan_agents"]

        ok = actual == expected if expected else "analytics" not in actual
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] role={role} query={query!r}")
        print(f"        expected={expected} actual={actual}")
        print(f"        reasoning={result['plan_reasoning']}\n")

    print(f"\n{passed} passed, {failed} failed out of {len(TEST_CASES)}")


if __name__ == "__main__":
    run_router_tests()