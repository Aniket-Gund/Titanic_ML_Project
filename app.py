import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime

# Set Page Configuration
st.set_page_config(
    page_title="Titanic Project",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Data and Model ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("titanic_updated.csv")
        # Ensure correct data types for plotting
        if 'pclass' in df.columns:
            df['pclass'] = df['pclass'].astype(str)
        if 'survived' in df.columns:
            df['survived_label'] = df['survived'].map({0: 'Not Survived', 1: 'Survived'})
        return df
    except FileNotFoundError:
        st.error("File 'titanic_updated.csv' not found. Please upload it.")
        return None

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("Titanic_ML_model.pkl", "rb"))
        return model
    except FileNotFoundError:
        st.error("File 'Titanic_ML_model.pkl' not found. Please upload it.")
        return None

df_raw = load_data()
model = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("🚢 Titanic Project")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
options = st.sidebar.radio("Navigate", ["Home", "EDA (Analysis)", "Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("Titanic Survival Prediction & Analysis Dashboard")

# --- HOME SECTION ---
if options == "Home":
    st.title("🚢 Titanic Machine Learning Project")
    st.markdown("""
    Welcome to the **Titanic Survival Prediction App**.
    
    ### 📂 Project Workflow:
    1.  **Data Understanding**: Analyzing rows, columns, and data types.
    2.  **Data Cleaning**: Handling missing values and outliers.
    3.  **EDA (Exploratory Data Analysis)**: Visualizing relationships (See 'EDA' tab).
    4.  **Machine Learning**: Random Forest Classifier (See 'Prediction' tab).
    
    👈 **Use the sidebar to navigate.**
    """)

# --- EDA SECTION (Updated with Plotly & HTML Export) ---
elif options == "EDA (Analysis)":
    st.title("📈 Titanic EDA Dashboard")
    
    if df_raw is not None:
        # -------------------------
        # Sidebar Filters
        # -------------------------
        st.sidebar.header("EDA Filters")
        
        df_view = df_raw.copy()
        
        # Filter by Class
        classes = sorted(df_view['pclass'].unique())
        selected_classes = st.sidebar.multiselect("Passenger Class", classes, default=classes)
        if selected_classes:
            df_view = df_view[df_view['pclass'].isin(selected_classes)]
            
        # Filter by Gender
        genders = sorted(df_view['sex'].unique())
        selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
        if selected_genders:
            df_view = df_view[df_view['sex'].isin(selected_genders)]

        # -------------------------
        # KPI Metrics
        # -------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Passengers", f"{len(df_view):,}")
        c2.metric("Survival Rate", f"{df_view['survived'].mean()*100:.1f}%")
        c3.metric("Avg Fare", f"£{df_view['fare'].mean():.2f}")
        c4.metric("Avg Age", f"{df_view['age'].mean():.1f} yrs")
        
        st.markdown("---")

        # -------------------------
        # Charts (Converted to Plotly for Export)
        # -------------------------
        figures = []

        # 1. Survival Distribution & Gender Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Survival Distribution")
            surv_counts = df_view['survived_label'].value_counts().reset_index()
            surv_counts.columns = ['Status', 'Count']
            fig_surv = px.bar(surv_counts, x='Status', y='Count', color='Status', 
                              color_discrete_map={'Survived': 'green', 'Not Survived': 'red'},
                              template="plotly_white", title="Survival Counts")
            st.plotly_chart(fig_surv, use_container_width=True)
            figures.append(fig_surv)

        with col2:
            st.subheader("Gender Distribution")
            fig_gender = px.pie(df_view, names='sex', title="Gender Distribution", 
                                color_discrete_sequence=['pink', 'skyblue'], hole=0.4)
            st.plotly_chart(fig_gender, use_container_width=True)
            figures.append(fig_gender)

        # 2. Survival by Class & Sex
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Survival Rate by Class")
            class_surv = df_view.groupby('pclass')['survived'].mean().reset_index()
            fig_class = px.bar(class_surv, x='pclass', y='survived', 
                               labels={'survived': 'Survival Rate', 'pclass': 'Passenger Class'},
                               color='survived', color_continuous_scale='Viridis',
                               template="plotly_white", title="Survival Rate by Pclass")
            st.plotly_chart(fig_class, use_container_width=True)
            figures.append(fig_class)
            
        with col4:
            st.subheader("Survival Rate by Sex")
            sex_surv = df_view.groupby('sex')['survived'].mean().reset_index()
            fig_sex = px.bar(sex_surv, x='sex', y='survived', 
                             labels={'survived': 'Survival Rate', 'sex': 'Gender'},
                             color='sex', color_discrete_map={'male': 'grey', 'female': 'pink'},
                             template="plotly_white", title="Survival Rate by Gender")
            st.plotly_chart(fig_sex, use_container_width=True)
            figures.append(fig_sex)

        # 3. Age Distribution
        st.subheader("Age Distribution by Survival")
        fig_age = px.histogram(df_view, x="age", color="survived_label", nbins=30,
                               marginal="box", # Adds boxplot at the top
                               color_discrete_map={'Survived': 'green', 'Not Survived': 'red'},
                               template="plotly_white", title="Age Distribution (Survived vs Not)")
        st.plotly_chart(fig_age, use_container_width=True)
        figures.append(fig_age)
        
        # 4. Correlation Heatmap
        st.subheader("Correlation Matrix")
        numeric_df = df_view.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                             title="Feature Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
        figures.append(fig_corr)
        
        # 5. Complex Relations: Class & Sex
        st.subheader("Survival by Class and Gender")
        # Group data for proper bar chart
        cat_data = df_view.groupby(['pclass', 'sex'])['survived'].mean().reset_index()
        fig_cat = px.bar(cat_data, x='pclass', y='survived', color='sex', barmode='group',
                         color_discrete_map={'male': 'grey', 'female': 'pink'},
                         title="Survival Rate by Class & Gender", template="plotly_white")
        st.plotly_chart(fig_cat, use_container_width=True)
        figures.append(fig_cat)

        # 6. Family Size & Age Group
        col5, col6 = st.columns(2)
        with col5:
             st.subheader("Survival by Family Size")
             fam_data = df_view.groupby('family_size')['survived'].mean().reset_index()
             fig_fam = px.bar(fam_data, x='family_size', y='survived', template="plotly_white",
                              title="Survival by Family Size")
             st.plotly_chart(fig_fam, use_container_width=True)
             figures.append(fig_fam)
             
        with col6:
            st.subheader("Survival by Age Group")
            if 'age_group' in df_view.columns:
                age_grp_data = df_view.groupby('age_group')['survived'].mean().reset_index()
                # Sort order for age groups
                order = ['Child', 'Teen', 'Young Adult', 'Adult', 'Old']
                fig_age_grp = px.bar(age_grp_data, x='age_group', y='survived', 
                                     category_orders={'age_group': order},
                                     template="plotly_white", title="Survival by Age Group")
                st.plotly_chart(fig_age_grp, use_container_width=True)
                figures.append(fig_age_grp)

        # -------------------------
        # Statistical Summary
        # -------------------------
        st.markdown("---")
        st.subheader("📊 Statistical Summary")
        st.dataframe(df_view.describe().T)

        # -------------------------
        # Insights Summary
        # -------------------------
        st.subheader("📝 Insights Summary")
        insights = []
        
        # Auto-generate some text insights based on current view
        surv_rate = df_view['survived'].mean()
        insights.append(f"The average survival rate in this selection is {surv_rate:.1%}.")
        
        female_surv = df_view[df_view['sex'] == 'female']['survived'].mean()
        male_surv = df_view[df_view['sex'] == 'male']['survived'].mean()
        if female_surv > male_surv:
            insights.append(f"Females ({female_surv:.1%}) had a higher survival chance than Males ({male_surv:.1%}).")
            
        p1_surv = df_view[df_view['pclass'] == '1']['survived'].mean() if '1' in df_view['pclass'].values else 0
        p3_surv = df_view[df_view['pclass'] == '3']['survived'].mean() if '3' in df_view['pclass'].values else 0
        if p1_surv > p3_surv:
            insights.append("1st Class passengers had a significantly better survival rate than 3rd Class.")

        for item in insights:
            st.write("✔", item)

        # -------------------------
        # HTML Export (ALL charts + summary)
        # -------------------------
        st.markdown("---")
        st.subheader("📤 Download Interactive HTML Snapshot")

        if st.button("⬇️ Download HTML Report"):
            html_blocks = []
            first = True
            for fig in figures:
                if fig is None: continue
                # include plotlyjs only once to preserve shared behavior & styles
                if first:
                    html_blocks.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                    first = False
                else:
                    html_blocks.append(fig.to_html(full_html=False, include_plotlyjs=False))

            # Build summary HTML
            summary_html = "<section style='font-family:Arial,Helvetica,sans-serif;'><h2>Insights Summary</h2><ul>"
            for i in insights:
                summary_html += f"<li>{i}</li>"
            summary_html += "</ul></section>"

            final_html = (
                "<html><head><meta charset='utf-8'></head><body>"
                f"<div style='font-family:Arial,Helvetica,sans-serif;padding:16px;'><h1>Titanic EDA Report</h1><p>Generated: {datetime.utcnow().isoformat()}</p><hr></div>"
                + "".join(html_blocks)
                + "<div style='padding:16px;'>" + summary_html + "</div>"
                + "</body></html>"
            )

            st.download_button(
                "Download HTML file",
                data=final_html.encode("utf-8"),
                file_name="titanic_eda_snapshot.html",
                mime="text/html"
            )

# --- PREDICTION SECTION ---
elif options == "Prediction":
    st.title("🔮 Survival Prediction")
    st.markdown("Enter passenger details below to check if they would have survived.")
    
    if model is not None:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], help="1 = 1st, 2 = 2nd, 3 = 3rd")
                sex_display = st.selectbox("Sex", ["Male", "Female"])
                sex = 1 if sex_display == "Male" else 0  # Mapping: Male=1, Female=0
                age = st.slider("Age", 0, 100, 25)
            
            with col2:
                fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
                sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
                parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
            
            # Calculate Family Size
            family_size = sibsp + parch + 1
            
            submit_btn = st.form_submit_button("Predict Survival")
        
        if submit_btn:
            input_data = pd.DataFrame({
                'pclass': [pclass],
                'sex': [sex],
                'age': [age],
                'fare': [fare],
                'family_size': [family_size]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.subheader("Prediction Result:")
                
                if prediction == 1:
                    st.success(f"**Result: SURVIVED** 🎉")
                    st.write(f"Probability of Survival: {probability[1]*100:.2f}%")
                    st.balloons()
                else:
                    st.error(f"**Result: DID NOT SURVIVE** 💀")
                    st.write(f"Probability of Survival: {probability[1]*100:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.error("Model could not be loaded.")
