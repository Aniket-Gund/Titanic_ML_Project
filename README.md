# **🚢 Titanic ML Dashboard – Project Overview**

### **Direct Live Link: https://titanic-ml-project-by-aniket-gund.streamlit.app/**

## **🎯 Objective:**

The goal of this Titanic Survival Prediction Project is to investigate one of history’s most infamous shipwrecks through the lens of data science. By analyzing 541 passenger records, this dashboard uncovers the critical factors—such as socio-economic status, gender, age, and family size—that determined who survived.

Beyond analysis, this project deploys a Random Forest Machine Learning model to predict the survival probability of any hypothetical passenger, turning historical data into actionable predictive insights.

## **🔍 Key Questions Explored in the EDA:**

1️⃣ How does Gender influence survival chances?

Visual: Bar Chart (Survival Rate by Sex)

Insight: Reveals the "Women and Children First" protocol. The data shows a stark contrast: Females had a ~66.3% survival rate, whereas Males had only ~19.6%.

2️⃣ What is the gender balance on board?

Visual: Pie Chart (Gender Imbalance)

Insight: Illustrates the demographic split of the passengers analyzed, highlighting the distribution between male and female travelers.

3️⃣ What is the age demographic of the passengers?

Visual: Line Chart (Age Distribution)

Insight: Traces the age spread across the ship, from infants (0.4 years) to the elderly (61 years), with an average passenger age of ~27.8 years.

4️⃣ How do Class, Gender, and Age collectively impact survival?

Visual: Interactive Treemap (Hierarchy of Survival)

Insight: A hierarchical breakdown that allows drilling down into specific subgroups (e.g., 1st Class -> Female -> Adult) to see exact survival counts and rates, revealing pockets of high and low survival.

5️⃣ Does Socio-Economic Status (Ticket Class) affect survival?

Visual: Bar Chart (Survival Rate by Class)

Insight: Confirms that wealth played a major role. 1st Class passengers survived at a rate of ~54.4%, significantly higher than the ~24.3% survival rate of 3rd Class passengers.

6️⃣ How does the interaction between Class and Gender affect outcomes?

Visual: Grouped Bar Chart (Survival by Class & Gender)

Insight: Dissects the data to show extreme cases, such as the high survival probability of 1st Class Females versus the dire outcomes for 3rd Class Males.

7️⃣ Which features are most strongly correlated with survival?

Visual: Correlation Heatmap

Insight: A matrix identifying relationships between numerical variables, highlighting strong positive/negative correlations (e.g., Fare vs. Survival, Pclass vs. Survival).

8️⃣ What are the key statistical properties of the dataset?

Visual: Statistical Summary Table

Insight: Provides a numeric snapshot (Mean, Std Dev, Min, Max) for all numerical features, excluding categorical counts like siblings/parents for cleaner analysis.

9️⃣ What insights can be derived overall?

## **Insights Summary (Auto-Generated):**

Gender Gap: Confirms females were ~3x more likely to survive than males.

Class Privilege: Higher fares and better classes correlated directly with safety.

Family Factor: Small families (2-4 members) often fared better than solo travelers or large families.

## **🧠 Machine Learning Model**
Algorithm: Random Forest Classifier

Features Used: Passenger Class, Sex, Age, Fare, Family Size

Function: Accepts user inputs to calculate a real-time Survival Probability Score (e.g., "85% Confidence").

## **📥 Export Options Available**
The dashboard includes robust reporting features:

✔ Download Interactive HTML Snapshot: Generates a standalone HTML report containing all current charts, insights, and summaries—preserving interactivity (zoom, hover) outside the app.
