import streamlit as st

from autoop.core.ml.artifact import Artifact

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("INSTRUCTIONS.md").read())
