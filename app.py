import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Titanic ML Project", layout="wide")

# ---------------- LOAD DATA ----------------
DATA_PATH = "/mnt/data/titanic_updated.csv"
MODEL_PATH = "/mnt/data/Titanic_ML_model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return pickle.load(open(MODEL_PATH, "rb"))

df = load_data()
model = load_model()

# ---------------- TITLE ----------------
st.markdown("## 🚢 Titanic Survival Prediction – ML Project Showcase")
st.caption("End-to-end ML project: Data Understanding → EDA → Prediction")

# ---------------- SIDEBAR ----------------
section = st.sidebar.radio("Navigation", ["📊 Dataset Understanding (EDA)", "🤖 Survival Prediction"])

# ==========================================================
# ===================== EDA SECTION =========================
# ==========================================================
if section == "📊 Dataset Understanding (EDA)":

    st.subheader("📌 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Overall Survival Rate", f"{df['Survived'].mean()*100:.2f}%")

    st.dataframe(df.head())

    st.divider()

    # 1. Target Variable Distribution
    st.subheader("1️⃣ Survival Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax)
    ax.set_xticklabels(["Not Survived", "Survived"])
    st.pyplot(fig)
    st.write("Majority of passengers did not survive, indicating class imbalance.")

    # 2. Gender vs Survival
    st.subheader("2️⃣ Gender vs Survival")
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=df, ax=ax)
    st.pyplot(fig)
    st.write("Female passengers had significantly higher survival chances.")

    # 3. Passenger Class vs Survival
    st.subheader("3️⃣ Passenger Class Impact")
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax)
    st.pyplot(fig)
    st.write("Higher passenger class strongly correlates with survival.")

    # 4. Age Distribution
    st.subheader("4️⃣ Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax)
    st.pyplot(fig)
    st.write("Children and younger passengers showed better survival probability.")

    # 5. Fare vs Survival
    st.subheader("5️⃣ Fare vs Survival")
    fig, ax = plt.subplots()
    sns.boxplot(x='Survived', y='Fare', data=df, ax=ax)
    st.pyplot(fig)
    st.write("Passengers paying higher fares had improved survival rates.")

    # 6. Correlation Heatmap
    st.subheader("6️⃣ Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.success(
        "📌 **EDA Summary:** Survival is most influenced by Gender, Passenger Class, Age, and Fare. "
        "These insights guided feature selection and model training."
    )

# ==========================================================
# ================== PREDICTION SECTION ====================
# ==========================================================
else:
    st.subheader("🤖 Survival Prediction")
    st.caption("Predict passenger survival using trained ML model")

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

    # Encoding (must match training pipeline)
    Sex = 1 if Sex == "male" else 0
    Embarked = {"C": 0, "Q": 1, "S": 2}[Embarked]

    if st.button("🔮 Predict Survival"):
        input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Passenger is likely to **SURVIVE**")
        else:
            st.error("❌ Passenger is likely to **NOT SURVIVE**")

        st.info("Prediction is based on patterns learned from the Titanic dataset using supervised ML.")
