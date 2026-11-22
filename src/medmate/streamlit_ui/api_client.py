import requests
import streamlit as st

BASE_URL = "http://localhost:8000"

class APIClient:
    def __init__(self):
        self.base_url = BASE_URL

    def _headers(self):
        token = st.session_state.get("access_token")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _request(self, method, path , **kwargs):
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers = self._headers(), **kwargs)
        if resp.status_code == 401:
            st.session_state.clear()
            st.error("Session expired. Please log in again.")
            st.rerun()
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def login(self, email, password):
        return self._request("POST", "/auth/login", json={"email": email, "password": password})

    def claim_account(self, payload):
        return self._request("POST", "/auth/register/claim", json=payload)

    def me(self):
        return self._request("GET", "/auth/me")

    def create_thread(self, title):
        return self._request("POST", "/chat/threads", json={"title": title})

    def list_threads(self):
        return self._request("GET", "/chat/threads")

    def get_history(self, thread_id):
        return self._request("GET", f"/chat/threads/{thread_id}/messages")

    def send_message(self, thread_id, content):
        return self._request("POST", f"/chat/threads/{thread_id}/messages", json={"content": content})

    def resume_action(self, thread_id, action_id, decision, edited_args=None):
        return self._request(
            "POST", f"/chat/threads/{thread_id}/resume",
            json={"action_id": action_id, "decision": decision, "edited_args": edited_args},
        )

api = APIClient()
        

