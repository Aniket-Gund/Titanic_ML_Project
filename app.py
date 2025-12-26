import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from datetime import datetime

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Titanic ML Project",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1, h2, h3 {
            margin-top: 0rem;
        }
        .stMetric {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #ff4b4b;
        }
        .main-text {
            font-size: 16px;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load Data & Model
# -------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("titanic_updated.csv")
        if 'age_group' not in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 30, 45, 60, 100], 
                                     labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Old', 'Elderly'])
        return df
    except FileNotFoundError:
        st.error("File 'titanic_updated.csv' not found. Please upload it.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return pickle.load(open("Titanic_ML_model.pkl", "rb"))
    except FileNotFoundError:
        return None

df = load_data()
model = load_model()

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🚢 Titanic Project")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
options = st.sidebar.radio("Navigate", ["Home", "EDA (Analysis)", "Prediction"])
st.sidebar.markdown("---")
st.sidebar.info("Titanic Survival Prediction & Analysis System")

# -------------------------
# SECTION: HOME
# -------------------------
if options == "Home":
    st.title("🚢 Titanic Machine Learning Project")
    
    st.markdown("""
    <div class="main-text">
    Welcome to the <b>Titanic Survival Prediction App</b>. This project explores the famous Titanic dataset to understand the factors that influenced survival and predicts whether a passenger would survive based on their details.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 1. About this file (Text sanitized)
    st.subheader("ℹ️ About this file")
    st.info("""
    The sinking of the Titanic is one of the most infamous shipwrecks in history.
    
    On April 15, 1912, during her maiden voyage, the widely considered 'unsinkable' RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.
    
    While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
    
    In this challenge, we ask you to build a predictive model that answers the question: 'what sorts of people were more likely to survive?' using passenger data (ie name, age, gender, socio-economic class, etc).
    """)

    st.markdown("---")

    # 2. Project Workflow
    st.subheader("📂 Project Workflow")
    st.markdown("""
    <div class="main-text">
    1.  <b>Data Understanding</b>: Analyzing rows, columns, and data types.
    2.  <b>Data Cleaning</b>: Handling missing values and outliers.
    3.  <b>EDA (Exploratory Data Analysis)</b>: Visualizing relationships between features.
    4.  <b>Machine Learning</b>: Training a Random Forest Classifier.
    </div>
    """, unsafe_allow_html=True)
       
# -------------------------
# SECTION: EDA (Analysis)
# -------------------------
elif options == "EDA (Analysis)":
    st.title("📈 Titanic Exploratory Data Analysis")
    
    if df.empty:
        st.stop()

    # --- Sidebar Filters (Preserved) ---
    st.sidebar.header("🔍 Filter Data")
    pclass_opts = sorted(df['pclass'].unique())
    sel_pclass = st.sidebar.multiselect("Passenger Class", pclass_opts, default=pclass_opts)
    
    sex_opts = [0, 1] 
    sex_labels = {0: 'Female', 1: 'Male'}
    sel_sex = st.sidebar.multiselect("Gender", sex_opts, format_func=lambda x: sex_labels[x], default=sex_opts)
    
    df_view = df.copy()
    if sel_pclass:
        df_view = df_view[df_view['pclass'].isin(sel_pclass)]
    if sel_sex:
        df_view = df_view[df_view['sex'].isin(sel_sex)]

    # --- KPIs ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Passengers", f"{len(df_view):,}")
    kpi2.metric("Survival Rate", f"{df_view['survived'].mean()*100:.1f}%")
    kpi3.metric("Avg Age", f"{df_view['age'].mean():.1f} Years")
    kpi4.metric("Avg Fare", f"£{df_view['fare'].mean():.2f}")
    
    st.markdown("---")
    
    figures = []

    # --- ROW 1: Gender Analysis ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Survival Rate by Sex")
        sex_survival = df_view.groupby('sex')['survived'].mean().reset_index()
        sex_survival['Gender'] = sex_survival['sex'].map({0: 'Female', 1: 'Male'})
        
        fig_sex_rate = px.bar(
            sex_survival, x='Gender', y='survived',
            color='Gender',
            color_discrete_map={'Female': '#ff9999', 'Male': '#66b3ff'},
            title="Chance of Survival by Gender",
            text_auto='.1%'
        )
        fig_sex_rate.update_layout(yaxis_title="Survival Probability", showlegend=False)
        st.plotly_chart(fig_sex_rate, use_container_width=True)
        figures.append(fig_sex_rate)

    with c2:
        st.subheader("Gender Imbalance") # NEW
        gender_counts = df_view['sex'].value_counts().reset_index()
        gender_counts.columns = ['sex', 'count']
        gender_counts['Gender'] = gender_counts['sex'].map({0: 'Female', 1: 'Male'})
        
        fig_gender_pie = px.pie(
            gender_counts, values='count', names='Gender',
            color='Gender',
            color_discrete_map={'Female': '#ff9999', 'Male': '#66b3ff'},
            title="Distribution of Male vs Female Passengers",
            hole=0.4
        )
        st.plotly_chart(fig_gender_pie, use_container_width=True)
        figures.append(fig_gender_pie)

    # --- ROW 2: Age Analysis (NEW) ---
    st.subheader("Age Distribution")
    # Preparing data for line chart
    age_counts = df_view['age'].value_counts().sort_index().reset_index()
    age_counts.columns = ['Age', 'Count']
    
    fig_age_line = px.line(
        age_counts, x='Age', y='Count',
        title="Passenger Age Distribution (Line Chart)",
        markers=True,
        template="plotly_white"
    )
    fig_age_line.update_traces(line_color='#8884d8')
    st.plotly_chart(fig_age_line, use_container_width=True)
    figures.append(fig_age_line)

    # --- ROW 3: Hierarchy (Treemap) ---
    st.subheader("Hierarchy of Survival")
    if 'pclass' in df_view and 'sex' in df_view:
        df_tree = df_view.copy()
        df_tree['Gender'] = df_tree['sex'].map({0: 'Female', 1: 'Male'})
        
        tree_data = df_tree.groupby(['pclass', 'Gender', 'age_group'], observed=False).agg(
            Count=('survived', 'count'),
            Survived_Rate=('survived', 'mean')
        ).reset_index()
        
        fig_tree = px.treemap(
            tree_data, path=['pclass', 'Gender', 'age_group'], values='Count',
            color='Survived_Rate', 
            color_continuous_scale='RdYlGn', 
            title="Drill Down: Class → Gender → Age Group"
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        figures.append(fig_tree)

    # --- ROW 4: Class Analysis ---
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("Survival Rate by Class")
        class_surv = df_view.groupby('pclass')['survived'].mean().reset_index()
        fig_class = px.bar(
            class_surv, x='pclass', y='survived',
            color='survived',
            color_continuous_scale='Teal',
            title="Survival Probability per Class",
            text_auto='.1%'
        )
        fig_class.update_xaxes(type='category', title='Passenger Class')
        st.plotly_chart(fig_class, use_container_width=True)
        figures.append(fig_class)
        
    with c4:
        st.subheader("Survival by Class and Gender")
        class_sex_surv = df_view.groupby(['pclass', 'sex'])['survived'].mean().reset_index()
        class_sex_surv['Gender'] = class_sex_surv['sex'].map({0: 'Female', 1: 'Male'})
        
        fig_class_sex = px.bar(
            class_sex_surv, x='pclass', y='survived', 
            color='Gender', barmode='group',
            color_discrete_map={'Female': '#ff9999', 'Male': '#66b3ff'},
            title="Survival Rate: Class vs Gender",
            text_auto='.1%'
        )
        fig_class_sex.update_xaxes(type='category', title='Passenger Class')
        st.plotly_chart(fig_class_sex, use_container_width=True)
        figures.append(fig_class_sex)

    # --- ROW 5: Correlation ---
    st.subheader("Feature Correlations")
    cols_to_corr = ['survived', 'pclass', 'sex', 'age', 'fare', 'family_size']
    available_cols = [c for c in cols_to_corr if c in df_view.columns]
    
    if available_cols:
        corr = df_view[available_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap (Excluding SibSp/Parch)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        figures.append(fig_corr)

    # --- Insights ---
    st.markdown("---")
    st.subheader("💡 Analysis Summary")
    
    insights = []
    # Automated logic
    f_surv = df_view[df_view['sex']==0]['survived'].mean()
    m_surv = df_view[df_view['sex']==1]['survived'].mean()
    insights.append(f"**Gender Gap:** Females had a higher survival rate ({f_surv:.1%}) vs Males ({m_surv:.1%}).")
    
    c1_surv = df_view[df_view['pclass']==1]['survived'].mean()
    c3_surv = df_view[df_view['pclass']==3]['survived'].mean()
    insights.append(f"**Class Privilege:** 1st Class passengers ({c1_surv:.1%}) survived more than 3rd Class ({c3_surv:.1%}).")

    for i in insights:
        st.write(f"✔️ {i}")

    # --- Download HTML ---
    st.markdown("---")
    if st.button("⬇️ Download Analysis Report (HTML)"):
        html_content = f"""
        <html>
        <head><title>Titanic EDA Report</title></head>
        <body style='font-family: Arial; padding: 20px;'>
            <h1 style='text-align:center;'>Titanic EDA Report</h1>
            <p style='text-align:center;'>Generated on {datetime.now().strftime('%Y-%m-%d')}</p>
            <hr>
            <h3>Key Insights:</h3>
            <ul>{''.join([f'<li>{insight}</li>' for insight in insights])}</ul>
            <hr>
        """
        for fig in figures:
            html_content += fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        html_content += "</body></html>"
        st.download_button("Download HTML", html_content.encode("utf-8"), "Titanic_Report.html", "text/html")

# -------------------------
# SECTION: PREDICTION
# -------------------------
elif options == "Prediction":
    st.title("🔮 Survival Prediction")
    st.markdown("Adjust the values below to see the prediction.")

    if model:
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1=1st, 2=2nd, 3=3rd Class")
                sex_inp = st.radio("Gender", ["Male", "Female"])
                age = st.slider("Age", 0, 100, 25)
            with c2:
                sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
                parch = st.number_input("Parents/Children", 0, 10, 0)
                fare = st.number_input("Fare Amount", 0.0, 600.0, 32.0)
            
            submit = st.form_submit_button("Predict Survival")
        
        if submit:
            sex_val = 1 if sex_inp == "Male" else 0
            fam_size = sibsp + parch + 1
            
            input_data = pd.DataFrame({
                'pclass': [pclass],
                'sex': [sex_val],
                'age': [age],
                'fare': [fare],
                'family_size': [fam_size]
            })
            
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            st.markdown("---")
            col_res, col_prob = st.columns([1, 2])
            
            with col_res:
                if pred == 1:
                    st.success("SURVIVED! 🎉")
                    st.balloons()
                else:
                    st.error("DID NOT SURVIVE 💀")
            
            with col_prob:
                st.write("Survival Probability:")
                st.progress(prob)
                st.caption(f"Confidence: {prob*100:.2f}%")
    else:
        st.error("Model file not found.")
