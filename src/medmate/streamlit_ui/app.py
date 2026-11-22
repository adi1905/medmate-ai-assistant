import streamlit as st
from state import init_session, is_authenticated
from views import login_view, chat_view

st.set_page_config(page_title="MedMate", layout="wide")
init_session()

if not is_authenticated():
    login_view.render()
else:
    chat_view.render()