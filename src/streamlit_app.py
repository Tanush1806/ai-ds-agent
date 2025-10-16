import streamlit as st
import pandas as pd
import os

# Additional helper functions for the Streamlit app can be added here
# This file serves as a module for shared Streamlit components and utilities

def init_session_state():
    """
    Initialize session state variables for the application.
    """
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
