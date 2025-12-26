# app.py
"""
Centered UI Titanic Survival Prediction — Clean, Interview-Ready
End-to-End ML Project Showcase (EDA → Prediction)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================== PAGE SETUP ==================
st.set_page_config(page_title="Titanic ML Project", layout="wide")

# ================== CUSTOM CSS ==================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0c0d;
        color: #e6eef8;
        font-family: "Segoe UI", Roboto, Arial;
    }
    .center-container {
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }
    .big-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 35px;
    }
    .card {
        background: rgba(255,255,255,0.02);
        border-radius: 14px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 22px;
    }
    .result-box {
        background: #0f3d2e;
        border-radius: 12px;
        padding: 18px;
        font-size: 20px;
        font-weight: 700;
        color: #d7f6e8;
        text-align: center;
        margin-top: 20px;
    }
    .footer {
        color: #9aa4b2;
        font-size: 13px;
        text-align: center;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== TITLE ==================
st.markdown('<div class="big-title">🚢 Titanic Survival Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML Project Showcase — Data Understanding → EDA → Prediction</div>', unsafe_allow_html=True)

# ================== FILE PATHS ==================
DATA_FILE = "titanic_updated.csv"
MODEL_FILE = "Titanic_ML_model.pkl"

if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
    st.error("Required data/model files not found in project directory.")
    st.stop()

# ================== LOAD DATA & MODEL ==================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# ================== NAVIGATION ==================
section = st.sidebar.radio("Navigation", ["📊 Dataset Understanding (EDA)", "🤖 Survival Prediction"])

# ======================================================
# ====================== EDA ============================
# ======================================================
if section == "📊 Dataset Understanding (EDA)":
    st.markdown('<div class="center-container">', unsafe_allow_html=True)

    # Overview
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Passengers", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("Survival Rate", f"{df['Survived'].mean()*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # Survival Distribution
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Survival Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df, ax=ax)
    ax.set_xticklabels(["Not Survived", "Survived"])
    st.pyplot(fig)
    st.caption("More passengers did not survive, showing slight class imbalance.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Gender vs Survival
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Gender vs Survival")
    fig, ax = plt.subplots()
    sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)
    st.caption("Females had a much higher survival probability.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Class vs Survival
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Passenger Class Impact")
    fig, ax = plt.subplots()
    sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)
    st.caption("First-class passengers were prioritized and survived more.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Age
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], kde=True, ax=ax)
    st.pyplot(fig)
    st.caption("Children and younger passengers had better chances.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.success("EDA Insight Summary: Survival is strongly influenced by Gender, Passenger Class, Age, and Fare. These features were used for model training.")
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# ================== PREDICTION =========================
# ======================================================
else:
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Passenger Details")

    col1, col2 = st.columns(2)
    with col1:
        Pclass = st.selectbox("Passenger Class", [1, 2, 3])
        Sex = st.selectbox("Sex", ["male", "female"])
        Age = st.slider("Age", 0, 80, 30)
        SibSp = st.number_input("Siblings / Spouses", 0, 8, 0)
    with col2:
        Parch = st.number_input("Parents / Children", 0, 6, 0)
        Fare = st.slider("Fare", 0.0, 500.0, 50.0)
        Embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Encoding
    Sex = 1 if Sex == "male" else 0
    Embarked = {"C": 0, "Q": 1, "S": 2}[Embarked]

    col_btn = st.columns([3, 6, 3])[1]
    with col_btn:
        predict_clicked = st.button("🔮 Predict Survival")

    if predict_clicked:
        X = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        pred = model.predict(X)[0]

        if pred == 1:
            st.markdown('<div class="result-box">Passenger is likely to SURVIVE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box">Passenger is likely to NOT SURVIVE</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown('<div class="footer">Built with Streamlit • Machine Learning • Titanic Dataset</div>', unsafe_allow_html=True)
