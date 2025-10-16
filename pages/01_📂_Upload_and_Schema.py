import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Upload & Schema", page_icon="📂")

st.title("📂 Upload Dataset and Analyze Schema")

st.markdown("""
### Instructions:
1. Upload your dataset (CSV, Excel)
2. The application will automatically analyze the schema
3. View data types, missing values, and basic statistics
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls"],
    help="Upload your dataset in CSV or Excel format"
)

if uploaded_file is not None:
    try:
        # Read the file based on its extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Store in session state
        st.session_state['df'] = df
        st.session_state['dataset_name'] = uploaded_file.name
        
        st.success(f"✅ Successfully loaded: {uploaded_file.name}")
        
        # Display basic information
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display first few rows
        st.subheader("👀 Preview (First 5 rows)")
        st.dataframe(df.head())
        
        # Schema information
        st.subheader("🗓️ Schema Information")
        
        schema_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        st.dataframe(schema_df, use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Next steps
        st.info("👉 Proceed to the next page to clean your data!")
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
else:
    st.info("⬆️ Please upload a dataset to begin.")
