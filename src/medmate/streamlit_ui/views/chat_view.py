import streamlit as st
from api_client import api
from state import logout


def render():
    with st.sidebar:
        st.write(f"Logged in as **{st.session_state['user']['name']}**")
        if st.button("Log out"):
            logout()
            st.rerun()

        st.divider()
        st.subheader("Threads")
        threads = api.list_threads().get("threads", [])
        thread_titles = {t["thread_id"]: t["title"] for t in threads}
        if st.button("+ New thread"):
            title = f"Chat {len(threads) + 1}"
            new_thread = api.create_thread(title)
            st.session_state["current_thread_id"] = new_thread["thread_id"]
            st.session_state["messages"] = []
            st.rerun()

        for tid, title in thread_titles.items():
            if st.button(title, key=f"thread-{tid}"):
                st.session_state["current_thread_id"] = tid
                history = api.get_history(tid)
                st.session_state["messages"] = history["messages"]
                st.session_state["pending_action"] = None
                st.rerun()

    st.title("MedMate Chat")
    thread_id = st.session_state.get("current_thread_id")
    if not thread_id:
        st.info("Create or select a thread from the sidebar.")
        return

    for msg in st.session_state["messages"]:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])

    pending = st.session_state.get("pending_action")
    if pending:
        with st.chat_message("assistant"):
            st.write(pending["human_readable_summary"])
            col1, col2 = st.columns(2)
            if col1.button("✅ Approve", key="approve_btn"):
                resp = api.resume_action(thread_id, pending["action_id"], "approve")
                st.session_state["messages"].append(resp["message"])
                st.session_state["pending_action"] = None
                st.rerun()
            if col2.button("❌ Reject", key="reject_btn"):
                resp = api.resume_action(thread_id, pending["action_id"], "reject")
                st.session_state["messages"].append(resp["message"])
                st.session_state["pending_action"] = None
                st.rerun()
        return  # block new input while a confirmation is pending

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        resp = api.send_message(thread_id, user_input)
        st.session_state["messages"].append(resp["message"])
        if resp.get("pending_action"):
            st.session_state["pending_action"] = resp["pending_action"]
        st.rerun()