import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Set Page Configuration
st.set_page_config(
    page_title="Titanic ML Project",
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

df = load_data()
model = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("🚢 Titanic Project")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
options = st.sidebar.radio("Navigate", ["Home", "EDA (Analysis)", "Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("Project based on Titanic Dataset analysis and survival prediction.")

# --- HOME SECTION ---
if options == "Home":
    st.title("🚢 Titanic Machine Learning Project")
    st.markdown("""
    Welcome to the **Titanic Survival Prediction App**. This project explores the famous Titanic dataset to understand the factors that influenced survival and predicts whether a passenger would survive based on their details.
    
    ### 📂 Project Workflow:
    1.  **Data Understanding**: Analyzing rows, columns, and data types.
    2.  **Data Cleaning**: Handling missing values and outliers.
    3.  **EDA (Exploratory Data Analysis)**: Visualizing relationships between features.
    4.  **Machine Learning**: Training a Random Forest Classifier.
    
    👈 **Use the sidebar to navigate to the EDA or Prediction sections.**
    """)
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Passengers", len(df))
        col2.metric("Survival Rate", f"{df['survived'].mean()*100:.2f}%")
        col3.metric("Features", len(df.columns))

# --- EDA SECTION ---
elif options == "EDA (Analysis)":
    st.title("📊 Exploratory Data Analysis")
    
    if df is not None:
        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Visualizations", "Key Insights & Q&A"])
        
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            st.subheader("Statistical Summary")
            st.write(df.describe())
            
            st.subheader("Data Info")
            buffer = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
            st.table(buffer)

        with tab2:
            st.subheader("📈 Data Visualizations")
            
            # Row 1: Survival Distribution & Sex Distribution
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Survival Distribution**")
                fig, ax = plt.subplots()
                sns.countplot(x='survived', data=df, palette=['red', 'green'], ax=ax)
                ax.set_xticklabels(['Not Survived', 'Survived'])
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Gender Distribution**")
                fig, ax = plt.subplots()
                df['sex'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Female', 'Male'], colors=['pink', 'skyblue'], ax=ax)
                st.pyplot(fig)

            st.markdown("---")

            # Row 2: Survival by Class & Sex
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Survival Rate by Pclass**")
                fig, ax = plt.subplots()
                sns.barplot(x='pclass', y='survived', data=df, palette='viridis', ax=ax)
                ax.set_ylabel("Survival Probability")
                st.pyplot(fig)
            
            with col4:
                st.markdown("**Survival Rate by Sex**")
                fig, ax = plt.subplots()
                sns.barplot(x='sex', y='survived', data=df, palette=['pink', 'grey'], ax=ax)
                ax.set_xticklabels(['Female', 'Male']) # Assuming 0=Female, 1=Male based on analysis
                ax.set_ylabel("Survival Probability")
                st.pyplot(fig)

            st.markdown("---")

            # Row 3: Age Distribution & Correlation
            col5, col6 = st.columns(2)
            with col5:
                st.markdown("**Age Distribution**")
                fig, ax = plt.subplots()
                sns.histplot(df['age'], kde=True, color='purple', bins=30, ax=ax)
                st.pyplot(fig)
            
            with col6:
                st.markdown("**Correlation Heatmap**")
                fig, ax = plt.subplots()
                # Select only numeric columns for correlation
                numeric_df = df.select_dtypes(include=[np.number])
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Row 4: Complex Relations
            st.markdown("**Survival by Pclass and Sex**")
            fig = sns.catplot(x='pclass', y='survived', hue='sex', data=df, kind='bar', palette={0: 'pink', 1: 'grey'}, height=4, aspect=2)
            # Customizing legend
            new_labels = ['Female', 'Male']
            for t, l in zip(fig._legend.texts, new_labels): t.set_text(l)
            st.pyplot(fig)

        with tab3:
            st.subheader("💡 Key Insights & Summary")
            st.markdown("""
            Based on the analysis performed on the dataset:
            
            1.  **Gender Impact**: Females had a significantly higher chance of survival compared to males.
            2.  **Class Distinction**: Passengers in 1st Class were more likely to survive than those in 3rd Class, indicating socio-economic status played a role.
            3.  **Age Factor**: Children (Age < 10) had higher survival rates. The age distribution shows a majority of passengers were young adults (20-30).
            4.  **Family Size**: Small families (size 2-4) tended to have better survival rates than those traveling alone or in very large families.
            5.  **Fare**: Higher fares correlated positively with survival, likely linked to Passenger Class.
            """)
            
            st.subheader("❓ Q&A Section")
            with st.expander("What is the overall survival rate?"):
                st.write(f"The overall survival rate is approximately {df['survived'].mean()*100:.1f}%.")
            with st.expander("Which gender survived more?"):
                st.write("Females survived more often than males.")
            with st.expander("Does Age affect survival?"):
                st.write("Yes, younger children had a better survival rate, and priority was likely given to them.")

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
            
            # Calculate Family Size (internal calculation as used in model training)
            family_size = sibsp + parch + 1
            
            submit_btn = st.form_submit_button("Predict Survival")
        
        if submit_btn:
            # Prepare input data matching the model's training features
            # Features expected: ['pclass', 'sex', 'age', 'fare', 'family_size']
            input_data = pd.DataFrame({
                'pclass': [pclass],
                'sex': [sex],
                'age': [age],
                'fare': [fare],
                'family_size': [family_size]
            })
            
            # Make Prediction
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
    else:
        st.error("Model could not be loaded. Please check if the .pkl file is present.")
