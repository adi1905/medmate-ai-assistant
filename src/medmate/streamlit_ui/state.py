import streamlit as st

def init_session():
    defaults = {
        "access_token": None,
        "user": None,
        "current_thread_id": None,
        "messages": [],
        "pending_action": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_authenticated():
    return st.session_state.get("access_token") is not None


def logout():
    st.session_state.clear()