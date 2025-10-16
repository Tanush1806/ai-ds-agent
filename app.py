import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Data Scientist Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🤖 AI Data Scientist Agent")

# Description
st.markdown("""
### Welcome to AI Data Scientist Agent!

This application helps you with end-to-end data science workflows:

1. **📂 Upload & Schema Analysis** - Upload your dataset and explore its structure
2. **🧹 Clean Data** - Handle missing values, duplicates, and outliers
3. **📊 Data Visualization** - Create insightful visualizations
4. **🤖 Modeling & Evaluation** - Build and evaluate ML models
5. **📑 Report** - Generate comprehensive analysis reports

👈 **Select a page from the sidebar to get started!**
""")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "This AI-powered agent assists you through the complete data science pipeline. "
        "Navigate through the pages in sequence for the best experience."
    )
    
    # Check if dataset is uploaded
    if 'df' in st.session_state:
        st.success(f"✅ Dataset loaded: {st.session_state.get('dataset_name', 'Unknown')}")
        st.metric("Rows", len(st.session_state['df']))
        st.metric("Columns", len(st.session_state['df'].columns))
    else:
        st.warning("⚠️ No dataset loaded. Please upload data in the first page.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Made with ❤️ using Streamlit | Powered by AI</div>",
    unsafe_allow_html=True
)
