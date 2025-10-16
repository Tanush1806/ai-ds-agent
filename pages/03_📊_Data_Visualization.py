import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Visualization", page_icon="📊", layout="wide")

st.title("📊 Data Visualization")

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ Please upload a dataset first in the 'Upload & Schema' page.")
    st.stop()

df = st.session_state['df']

st.markdown("""
### Create Interactive Visualizations
Select the type of plot and customize it according to your needs.
""")

# Sidebar for plot selection
plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    [
        "Histogram",
        "Box Plot",
        "Scatter Plot",
        "Line Plot",
        "Bar Chart",
        "Correlation Heatmap",
        "Pair Plot"
    ]
)

if plot_type == "Histogram":
    st.subheader("📉 Histogram")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        col = st.selectbox("Select column:", numeric_cols)
        bins = st.slider("Number of bins:", 5, 100, 30)
        
        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns available.")

elif plot_type == "Box Plot":
    st.subheader("📦 Box Plot")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        col = st.selectbox("Select column:", numeric_cols)
        
        fig = px.box(df, y=col, title=f"Box Plot of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns available.")

elif plot_type == "Scatter Plot":
    st.subheader("⚫ Scatter Plot")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Select X-axis:", numeric_cols)
        with col2:
            y_col = st.selectbox("Select Y-axis:", numeric_cols, index=1)
        
        color_col = st.selectbox("Color by (optional):", [None] + df.columns.tolist())
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                        title=f"Scatter Plot: {x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Need at least 2 numeric columns for scatter plot.")

elif plot_type == "Line Plot":
    st.subheader("📈 Line Plot")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        y_col = st.selectbox("Select Y-axis:", numeric_cols)
        x_col = st.selectbox("Select X-axis (or index):", [None] + df.columns.tolist())
        
        if x_col:
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Plot: {y_col} over {x_col}")
        else:
            fig = px.line(df, y=y_col, title=f"Line Plot: {y_col}")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns available.")

elif plot_type == "Bar Chart":
    st.subheader("📊 Bar Chart")
    
    col = st.selectbox("Select column:", df.columns.tolist())
    
    if df[col].dtype == 'object' or len(df[col].unique()) < 50:
        value_counts = df[col].value_counts().head(20)
        
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                    labels={'x': col, 'y': 'Count'},
                    title=f"Bar Chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Too many unique values. Consider using a histogram instead.")

elif plot_type == "Correlation Heatmap":
    st.subheader("🔥 Correlation Heatmap")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu_r',
                       title="Correlation Heatmap")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Need at least 2 numeric columns for correlation heatmap.")

elif plot_type == "Pair Plot":
    st.subheader("🔷 Pair Plot")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        selected_cols = st.multiselect(
            "Select columns (max 5 for performance):",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if len(selected_cols) >= 2 and len(selected_cols) <= 5:
            st.info("🕒 Generating pair plot... This may take a moment.")
            
            fig = px.scatter_matrix(df[selected_cols],
                                   title="Pair Plot")
            st.plotly_chart(fig, use_container_width=True)
        elif len(selected_cols) > 5:
            st.warning("⚠️ Please select maximum 5 columns for performance reasons.")
        else:
            st.warning("⚠️ Please select at least 2 columns.")
    else:
        st.warning("⚠️ Need at least 2 numeric columns for pair plot.")

# Additional insights
st.markdown("---")
st.info("👉 Once you're done with visualizations, proceed to the Modeling page!")
