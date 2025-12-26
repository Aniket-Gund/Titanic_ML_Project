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
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS (Remove White Space & Styling)
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
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load Data & Model
# -------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("titanic_updated.csv")
        # Ensure Age Group exists
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
st.sidebar.title("🚢 Titanic Dashboard")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
options = st.sidebar.radio("Navigate", ["Home", "EDA (Analysis)", "Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("Explore the Titanic dataset and predict survival.")

# -------------------------
# SECTION: HOME
# -------------------------
if options == "Home":
    st.title("Welcome to Titanic ML Project 🌊")
    st.markdown("""
    This app provides a comprehensive analysis of the Titanic dataset and a Machine Learning model to predict passenger survival.
    
    ### 🚀 Key Features:
    * **Data Analysis:** Interactive charts to understand survival factors.
    * **Pattern Recognition:** identifying how Class, Sex, and Age affected survival.
    * **Live Prediction:** Test the trained Random Forest model.
    """)
    
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Passengers", len(df))
        c2.metric("Overall Survival Rate", f"{df['survived'].mean()*100:.1f}%")
        c3.metric("Average Fare", f"£{df['fare'].mean():.2f}")

# -------------------------
# SECTION: EDA (Analysis)
# -------------------------
elif options == "EDA (Analysis)":
    st.title("📈 Titanic Exploratory Data Analysis")
    
    if df.empty:
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filter Data")
    pclass_opts = sorted(df['pclass'].unique())
    sel_pclass = st.sidebar.multiselect("Passenger Class", pclass_opts, default=pclass_opts)
    
    sex_opts = [0, 1] 
    sex_labels = {0: 'Female', 1: 'Male'}
    sel_sex = st.sidebar.multiselect("Gender", sex_opts, format_func=lambda x: sex_labels[x], default=sex_opts)
    
    # Filter Logic
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

    # --- ROW 1: Survival Rate by Sex & Gender Dist ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Survival Rate by Sex")
        # Calculate survival rate by sex
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
        st.subheader("Gender Distribution")
        df_view['Gender_Label'] = df_view['sex'].map({0: 'Female', 1: 'Male'})
        fig_gender = px.histogram(
            df_view, x="Gender_Label", 
            color="survived",
            color_discrete_map={0: '#d62728', 1: '#2ca02c'}, # Red (Dead), Green (Alive)
            barmode='group',
            title="Count of Passengers by Gender"
        )
        # Custom legend names
        new_names = {'0': 'Not Survived', '1': 'Survived'}
        fig_gender.for_each_trace(lambda t: t.update(name = new_names.get(t.name, t.name)))
        st.plotly_chart(fig_gender, use_container_width=True)
        figures.append(fig_gender)

    # --- ROW 2: Treemap (Keeping as is, it's good for overview) ---
    st.subheader("Hierarchy of Survival")
    if 'pclass' in df_view and 'sex' in df_view:
        df_tree = df_view.copy()
        df_tree['Gender'] = df_tree['sex'].map({0: 'Female', 1: 'Male'})
        df_tree['Survived_Label'] = df_tree['survived'].map({0: 'No', 1: 'Yes'})
        
        tree_data = df_tree.groupby(['pclass', 'Gender', 'age_group'], observed=False).agg(
            Count=('survived', 'count'),
            Survived_Rate=('survived', 'mean')
        ).reset_index()
        
        fig_tree = px.treemap(
            tree_data, path=['pclass', 'Gender', 'age_group'], values='Count',
            color='Survived_Rate', 
            color_continuous_scale='RdYlGn', # Red to Green scale
            title="Drill Down: Class → Gender → Age Group (Color = Survival Rate)"
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        figures.append(fig_tree)

    # --- ROW 3: Class & Class+Gender Analysis ---
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("Survival Rate by Class")
        class_surv = df_view.groupby('pclass')['survived'].mean().reset_index()
        fig_class = px.bar(
            class_surv, x='pclass', y='survived',
            color='survived',
            color_continuous_scale='Teal', # Better color combination
            title="Survival Probability per Class",
            text_auto='.1%'
        )
        fig_class.update_xaxes(type='category', title='Passenger Class')
        fig_class.update_yaxes(title='Survival Rate')
        st.plotly_chart(fig_class, use_container_width=True)
        figures.append(fig_class)
        
    with c4:
        st.subheader("Survival by Class and Gender")
        # Grouped bar chart
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
        fig_class_sex.update_yaxes(title='Survival Rate')
        st.plotly_chart(fig_class_sex, use_container_width=True)
        figures.append(fig_class_sex)

    # --- ROW 4: Correlation Matrix ---
    st.subheader("Feature Correlations")
    # Drop sibsp and parch as requested, keep others
    cols_to_corr = ['survived', 'pclass', 'sex', 'age', 'fare', 'family_size']
    # Check if cols exist
    available_cols = [c for c in cols_to_corr if c in df_view.columns]
    
    if available_cols:
        corr = df_view[available_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", # Red-Blue Diverging (standard for corr)
            title="Correlation Heatmap (Excluding SibSp/Parch)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        figures.append(fig_corr)

    # --- Summary & Insights ---
    st.markdown("---")
    st.subheader("💡 Analysis Summary")
    
    insights = []
    
    # 1. Gender Insight
    f_surv = df_view[df_view['sex']==0]['survived'].mean()
    m_surv = df_view[df_view['sex']==1]['survived'].mean()
    insights.append(f"**Gender Gap:** Females had a significantly higher survival rate ({f_surv:.1%}) compared to males ({m_surv:.1%}).")
    
    # 2. Class Insight
    c1_surv = df_view[df_view['pclass']==1]['survived'].mean()
    c3_surv = df_view[df_view['pclass']==3]['survived'].mean()
    insights.append(f"**Class Privilege:** First-class passengers were much more likely to survive ({c1_surv:.1%}) than those in third class ({c3_surv:.1%}).")
    
    # 3. Combined Insight
    try:
        f_c1 = df_view[(df_view['sex']==0) & (df_view['pclass']==1)]['survived'].mean()
        m_c3 = df_view[(df_view['sex']==1) & (df_view['pclass']==3)]['survived'].mean()
        insights.append(f"**Extreme Cases:** First-class females had the highest survival chance (~{f_c1:.1%}), while third-class males had the lowest (~{m_c3:.1%}).")
    except:
        pass

    # 4. Family Size Insight
    if 'family_size' in df_view.columns:
        single = df_view[df_view['family_size']==1]['survived'].mean()
        small_fam = df_view[(df_view['family_size'] > 1) & (df_view['family_size'] <= 4)]['survived'].mean()
        large_fam = df_view[df_view['family_size'] > 4]['survived'].mean()
        insights.append(f"**Family Factor:** Small families (2-4 members) survived better ({small_fam:.1%}) than single travelers ({single:.1%}) or large families ({large_fam:.1%}).")

    for i in insights:
        st.write(f"✔️ {i}")

    # --- Download Report ---
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
            
            # Create input DF
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
