import streamlit as st
from api_client import api


def render():
    st.title("MedMate Login")
    tab_login, tab_claim = st.tabs(["Login", "Claim Account"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log in"):
            try:
                tokens = api.login(email, password)
                st.session_state["access_token"] = tokens["access_token"]
                st.session_state["user"] = api.me()
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_claim:
        first_name = st.text_input("First name")
        last_name = st.text_input("Last name")
        dob = st.date_input("Date of birth")
        phone = st.text_input("Phone")
        claim_email = st.text_input("Email", key="claim_email")
        claim_password = st.text_input("Password", type="password", key="claim_pw")
        if st.button("Claim account"):
            try:
                resp = api.claim_account({
                    "first_name": first_name, "last_name": last_name,
                    "dob": str(dob), "phone": phone,
                    "email": claim_email, "password": claim_password,
                })
                st.session_state["access_token"] = resp["tokens"]["access_token"]
                st.session_state["user"] = resp["user"]
                st.rerun()
            except Exception as e:
                st.error(f"Claim failed: {e}")