"""Streamlit app: Interactive supervised machine learning explorer

Usage:
  streamlit run streamlit_app.py

Features:
- Choose a sample dataset (Iris, Breast Cancer, Diabetes) or upload CSV
- Select target and features
- Choose model and hyperparameters
- Train/test split, train model, show metrics and plots
"""
from pathlib import Path
import io
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Supervised ML Explorer", layout="wide")


def load_sample(name):
    if name == "Iris (classification)":
        data = datasets.load_iris(as_frame=True)
        df = data.frame
        return df, data.target_names
    if name == "Breast Cancer (classification)":
        data = datasets.load_breast_cancer(as_frame=True)
        df = data.frame
        return df, None
    if name == "Diabetes (regression)":
        data = datasets.load_diabetes(as_frame=True)
        df = data.frame
        return df, None
    return None, None


def infer_task(y_series):
    # Heuristic: if target has <= 20 unique values and is numeric ints or categories -> classification
    if y_series.dtype.kind in "biu" or y_series.dtype == object:
        # integer-like or object
        if y_series.nunique() <= 20:
            return "classification"
    # If few unique and not continuous numeric, classification
    if y_series.nunique() <= 10 and y_series.dtype.kind in "f":
        # Could be classification encoded as floats — keep as classification when small unique
        return "classification"
    # Otherwise regression
    return "regression"


def build_model(model_key, task, params):
    if task == "classification":
        if model_key == "Logistic Regression":
            return LogisticRegression(max_iter=1000, **params)
        if model_key == "Random Forest":
            return RandomForestClassifier(**params)
        if model_key == "K-Nearest Neighbors":
            return KNeighborsClassifier(**params)
        if model_key == "Decision Tree":
            return DecisionTreeClassifier(**params)
    else:
        if model_key == "Linear Regression":
            return LinearRegression(**params)
        if model_key == "Random Forest (Regressor)":
            return RandomForestRegressor(**params)
    return None


def main():
    st.title("Supervised Machine Learning Explorer")

    with st.sidebar:
        st.header("Data")
        sample = st.selectbox(
            "Sample dataset",
            ["None", "Iris (classification)", "Breast Cancer (classification)", "Diabetes (regression)"],
        )
        uploaded = st.file_uploader("Or upload CSV", type=["csv"])
        st.markdown("---")
        st.header("Model & Training")
        test_size = st.slider("Test set proportion", 0.05, 0.5, 0.2)
        random_state = st.number_input("Random seed", value=42)

    # Load data
    df = None
    if sample != "None":
        df, target_names = load_sample(sample)
    elif uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

    if df is None:
        st.info("Choose a sample dataset or upload a CSV to get started.")
        return

    st.subheader("Dataset preview")
    st.write(df.head())

    # Column selection
    all_cols = list(df.columns)
    default_target = all_cols[-1]
    target_col = st.selectbox("Select target column", all_cols, index=max(0, all_cols.index(default_target)))
    feature_cols = st.multiselect("Select feature columns (leave empty to use all except target)", all_cols, default=[c for c in all_cols if c != target_col])
    if not feature_cols:
        feature_cols = [c for c in all_cols if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    task = infer_task(y)
    st.write(f"Detected task: **{task}** (heuristic)")

    # Model selection
    if task == "classification":
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Decision Tree"])
    else:
        model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest (Regressor)"])

    # Simple hyperparameter UI
    params = {}
    if "Random Forest" in model_choice:
        n_estimators = st.number_input("n_estimators", 10, 1000, 100)
        max_depth = st.number_input("max_depth (0 for None)", 0, 100, 0)
        params["n_estimators"] = n_estimators
        if max_depth > 0:
            params["max_depth"] = int(max_depth)
    if model_choice == "K-Nearest Neighbors":
        params["n_neighbors"] = st.number_input("n_neighbors", 1, 50, 5)

    run_train = st.button("Train model")

    if run_train:
        # Prepare data: simple numeric conversion + drop NA
        X_proc = X.select_dtypes(include=[np.number]).copy()
        non_numeric = [c for c in X.columns if c not in X_proc.columns]
        if non_numeric:
            st.warning(f"Dropping non-numeric features: {non_numeric}")

        X_proc = X_proc.fillna(X_proc.mean())

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=int(random_state))

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_model(model_choice, task, params)
        if model is None:
            st.error("Unsupported model")
            return

        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

        st.subheader("Results")
        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("Precision (weighted)", f"{prec:.3f}")
            st.metric("Recall (weighted)", f"{rec:.3f}")
            st.metric("F1 (weighted)", f"{f1:.3f}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification report")
            st.text(classification_report(y_test, y_pred, zero_division=0))

        else:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.metric("R^2", f"{r2:.3f}")
            st.metric("MSE", f"{mse:.3f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

        # Feature importance where available
        st.subheader("Feature importance / coefficients")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.Series(importances, index=X_proc.columns).sort_values(ascending=False)
            st.bar_chart(fi)
        elif hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
            fi = pd.Series(coefs, index=X_proc.columns).sort_values(key=lambda x: np.abs(x), ascending=False)
            st.bar_chart(fi)
        else:
            st.write("No feature importance available for this model type.")

        # Allow downloading the trained model
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button("Download trained model (joblib)", data=buf, file_name="model.joblib")


if __name__ == "__main__":
    main()
