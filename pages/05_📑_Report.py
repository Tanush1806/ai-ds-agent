import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Report", page_icon="📑", layout="wide")

st.title("📑 Comprehensive Analysis Report")

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ Please upload a dataset first in the 'Upload & Schema' page.")
    st.stop()

df = st.session_state['df']

st.markdown("""
### Generate a Comprehensive Report
This page summarizes all the analysis performed on your dataset.
""")

# Report metadata
st.sidebar.header("📄 Report Information")
report_title = st.sidebar.text_input("Report Title", "Data Analysis Report")
report_author = st.sidebar.text_input("Author", "AI Data Scientist Agent")
report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.sidebar.markdown(f"**Generated on:** {report_date}")

# Main report
st.header(report_title)
st.markdown(f"**Author:** {report_author}")
st.markdown(f"**Date:** {report_date}")
st.markdown("---")

# 1. Dataset Overview
st.subheader("📂 1. Dataset Overview")

st.write(f"**Dataset Name:** {st.session_state.get('dataset_name', 'Unknown')}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rows", len(df))
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
with col4:
    st.metric("Missing Values", df.isnull().sum().sum())

st.markdown("---")

# 2. Data Quality
st.subheader("✅ 2. Data Quality Assessment")

missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]

if len(missing_data) > 0:
    st.warning(f"⚠️ Found missing values in {len(missing_data)} column(s)")
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': (missing_data.values / len(df) * 100).round(2)
    })
    
    st.dataframe(missing_df, use_container_width=True)
else:
    st.success("✅ No missing values found in the dataset.")

# Check for duplicates
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    st.warning(f"⚠️ Found {duplicate_count} duplicate rows")
else:
    st.success("✅ No duplicate rows found.")

st.markdown("---")

# 3. Statistical Summary
st.subheader("📈 3. Statistical Summary")

st.write("**Descriptive Statistics for Numerical Columns:**")
st.dataframe(df.describe(), use_container_width=True)

st.markdown("---")

# 4. Data Types
st.subheader("🗓️ 4. Data Types")

type_df = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes,
    'Unique Values': [df[col].nunique() for col in df.columns],
    'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else 'N/A' for col in df.columns]
})

st.dataframe(type_df, use_container_width=True)

st.markdown("---")

# 5. Model Results (if available)
if 'model_results' in st.session_state and st.session_state['model_results'] is not None:
    st.subheader("🤖 5. Machine Learning Model Results")
    
    results = st.session_state['model_results']
    
    st.write(f"**Model Type:** {results.get('model', 'N/A')}")
    st.write(f"**Problem Type:** {results.get('type', 'N/A').title()}")
    
    if results.get('type') == 'classification':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Precision", f"{results.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{results.get('recall', 0):.3f}")
        with col4:
            st.metric("F1 Score", f"{results.get('f1', 0):.3f}")
    
    elif results.get('type') == 'regression':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{results.get('r2', 0):.3f}")
        with col2:
            st.metric("RMSE", f"{results.get('rmse', 0):.3f}")
        with col3:
            st.metric("MAE", f"{results.get('mae', 0):.3f}")
        with col4:
            st.metric("MSE", f"{results.get('mse', 0):.3f}")
    
    st.markdown("---")

# 6. Conclusions and Recommendations
st.subheader("📝 6. Conclusions and Recommendations")

st.markdown("""
**Key Findings:**
- Dataset successfully loaded and analyzed
- Data quality assessment completed
- Statistical analysis performed
""")

if 'model_results' in st.session_state and st.session_state['model_results'] is not None:
    st.markdown("- Machine learning model trained and evaluated")

st.markdown("""
**Recommendations:**
- Continue monitoring data quality
- Consider feature engineering for improved model performance
- Explore additional visualization techniques
- Regularly update the model with new data
""")

st.markdown("---")

# Download report
st.subheader("💾 Download Report")

if st.button("📥 Generate Downloadable Report"):
    # Create a text summary
    report_text = f"""
{report_title}
{'=' * len(report_title)}

Author: {report_author}
Date: {report_date}

1. DATASET OVERVIEW
-------------------
Dataset Name: {st.session_state.get('dataset_name', 'Unknown')}
Total Rows: {len(df)}
Total Columns: {len(df.columns)}
Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

2. DATA QUALITY
---------------
Missing Values: {df.isnull().sum().sum()}
Duplicate Rows: {df.duplicated().sum()}

3. COLUMNS
----------
{', '.join(df.columns.tolist())}

---
Report generated by AI Data Scientist Agent
    """
    
    st.download_button(
        label="Download Report as Text",
        data=report_text,
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    st.success("✅ Report ready for download!")

st.info("✨ Thank you for using AI Data Scientist Agent!")
