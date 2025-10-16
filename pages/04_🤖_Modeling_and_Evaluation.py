import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Modeling & Evaluation", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Modeling & Evaluation")

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ Please upload a dataset first in the 'Upload & Schema' page.")
    st.stop()

df = st.session_state['df'].copy()

st.markdown("""
### Build and Evaluate Machine Learning Models
Select your target variable and features to train a model.
""")

# Select target variable
st.subheader("🎯 Target Variable Selection")
target_col = st.selectbox("Select target variable (what you want to predict):", df.columns.tolist())

if target_col:
    # Determine if it's a classification or regression problem
    is_classification = df[target_col].dtype == 'object' or df[target_col].nunique() < 10
    
    problem_type = "Classification" if is_classification else "Regression"
    st.info(f"🔸 Detected problem type: **{problem_type}**")
    
    # Select features
    st.subheader("📊 Feature Selection")
    feature_cols = st.multiselect(
        "Select feature columns:",
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col][:min(5, len(df.columns)-1)]
    )
    
    if len(feature_cols) > 0:
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            st.warning(f"⚠️ Encoding {len(categorical_cols)} categorical feature(s)...")
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if classification
        if is_classification:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Train-test split
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test set size:", 0.1, 0.5, 0.2)
        
        with col2:
            random_state = st.number_input("Random state:", value=42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        
        # Model selection
        st.subheader("🧠 Model Selection")
        
        if is_classification:
            model_name = st.selectbox(
                "Select classification model:",
                ["Logistic Regression", "Decision Tree", "Random Forest"]
            )
        else:
            model_name = st.selectbox(
                "Select regression model:",
                ["Linear Regression", "Decision Tree", "Random Forest"]
            )
        
        # Train model
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                # Create model
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Decision Tree" and is_classification:
                    model = DecisionTreeClassifier()
                elif model_name == "Decision Tree" and not is_classification:
                    model = DecisionTreeRegressor()
                elif model_name == "Random Forest" and is_classification:
                    model = RandomForestClassifier()
                elif model_name == "Random Forest" and not is_classification:
                    model = RandomForestRegressor()
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Evaluate
                st.success("✅ Model trained successfully!")
                
                st.subheader("📈 Model Evaluation")
                
                if is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Precision", f"{precision:.3f}")
                    with col3:
                        st.metric("Recall", f"{recall:.3f}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.3f}")
                    
                    # Store results
                    st.session_state['model_results'] = {
                        'type': 'classification',
                        'model': model_name,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("R² Score", f"{r2:.3f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.3f}")
                    with col3:
                        st.metric("MAE", f"{mae:.3f}")
                    with col4:
                        st.metric("MSE", f"{mse:.3f}")
                    
                    # Prediction vs Actual plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test, y=y_pred,
                        mode='markers',
                        name='Predictions'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Predicted vs Actual Values",
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store results
                    st.session_state['model_results'] = {
                        'type': 'regression',
                        'model': model_name,
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'mse': mse
                    }
                
                st.info("👉 Proceed to the Report page to generate a comprehensive analysis report!")
    else:
        st.warning("⚠️ Please select at least one feature column.")
