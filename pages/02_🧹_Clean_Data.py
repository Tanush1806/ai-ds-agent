import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Clean Data", page_icon="🧹")

st.title("🧹 Data Cleaning")

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ Please upload a dataset first in the 'Upload & Schema' page.")
    st.stop()

df = st.session_state['df'].copy()

st.markdown("""
### Data Cleaning Options
Select the operations you want to perform on your dataset:
""")

# Create tabs for different cleaning operations
tab1, tab2, tab3, tab4 = st.tabs([
    "🖍️ Missing Values",
    "🔁 Duplicates",
    "🎯 Outliers",
    "💾 Save Cleaned Data"
])

with tab1:
    st.subheader("Handle Missing Values")
    
    # Show columns with missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        st.write("Columns with missing values:")
        st.dataframe(pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df) * 100).round(2)
        }))
        
        # Select columns to handle
        cols_to_handle = st.multiselect(
            "Select columns to handle missing values:",
            missing_data.index.tolist()
        )
        
        if cols_to_handle:
            method = st.selectbox(
                "Select method:",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"]
            )
            
            if st.button("Apply Missing Value Treatment"):
                for col in cols_to_handle:
                    if method == "Drop rows":
                        df = df.dropna(subset=[col])
                    elif method == "Fill with mean":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "Fill with median":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].median(), inplace=True)
                    elif method == "Fill with mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == "Fill with custom value":
                        custom_value = st.text_input(f"Enter custom value for {col}")
                        if custom_value:
                            df[col].fillna(custom_value, inplace=True)
                
                st.session_state['cleaned_df'] = df
                st.success("✅ Missing values handled successfully!")
    else:
        st.info("✅ No missing values found in the dataset.")

with tab2:
    st.subheader("Handle Duplicate Rows")
    
    duplicate_count = df.duplicated().sum()
    st.metric("Number of duplicate rows", duplicate_count)
    
    if duplicate_count > 0:
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state['cleaned_df'] = df
            st.success(f"✅ Removed {duplicate_count} duplicate rows!")
    else:
        st.info("✅ No duplicate rows found.")

with tab3:
    st.subheader("Handle Outliers")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        col_to_check = st.selectbox("Select column to check for outliers:", numeric_cols)
        
        if col_to_check:
            # Calculate IQR
            Q1 = df[col_to_check].quantile(0.25)
            Q3 = df[col_to_check].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col_to_check] < lower_bound) | (df[col_to_check] > upper_bound)]
            
            st.metric("Number of outliers", len(outliers))
            st.write(f"Lower bound: {lower_bound:.2f}")
            st.write(f"Upper bound: {upper_bound:.2f}")
            
            if len(outliers) > 0:
                method = st.radio(
                    "Select method to handle outliers:",
                    ["Remove outliers", "Cap outliers (Winsorization)"]
                )
                
                if st.button("Apply Outlier Treatment"):
                    if method == "Remove outliers":
                        df = df[(df[col_to_check] >= lower_bound) & (df[col_to_check] <= upper_bound)]
                    elif method == "Cap outliers (Winsorization)":
                        df[col_to_check] = df[col_to_check].clip(lower=lower_bound, upper=upper_bound)
                    
                    st.session_state['cleaned_df'] = df
                    st.success("✅ Outliers handled successfully!")
    else:
        st.info("ℹ️ No numeric columns found for outlier detection.")

with tab4:
    st.subheader("Save Cleaned Dataset")
    
    # Show comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Original Dataset Rows", len(st.session_state['df']))
    
    with col2:
        cleaned_df = st.session_state.get('cleaned_df', df)
        st.metric("Cleaned Dataset Rows", len(cleaned_df))
    
    if st.button("Save Cleaned Data"):
        st.session_state['df'] = cleaned_df
        st.success("✅ Cleaned data saved successfully!")
        st.info("👉 Proceed to the next page for data visualization!")
    
    # Preview cleaned data
    st.subheader("Preview Cleaned Data")
    st.dataframe(cleaned_df.head(), use_container_width=True)
