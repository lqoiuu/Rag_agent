"""Streamlit pages.

Each module is a plain script that Streamlit executes directly, not a callable with a
render function — that is the documented structure for an ``st.navigation`` app. This
file exists so the directory is a package: without it, test collection would try to
import the pages as test modules and execute the whole UI at import time.
"""
