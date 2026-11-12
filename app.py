"""
Streamlit Interactive ML Web App: Titanic Survival Predictor
Intern Circle - Project 05

Requirements fulfilled:
1. Streamlit interface creation (modern, responsive UI with sliders and selectors)
2. Model loading using joblib
3. Real-time interactive prediction display with confidence probabilities
4. Feature summary & sensitivity analysis
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="Titanic Survival Predictor | Intern Circle ML",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
    }
    .survived-card {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .not-survived-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Load pre-trained machine learning pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "titanic_classifier.joblib")
    if not os.path.exists(model_path):
        # Fallback to 04_ml_classifier outputs
        model_path = os.path.join(base_dir, "..", "04_ml_classifier", "outputs", "titanic_classifier.joblib")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run Project 04 first!")
        st.stop()
    
    model = joblib.load(model_path)
    return model


def main():
    model = load_trained_model()

    # Title & Header
    st.markdown("<div class='main-header'>🚢 Titanic Survival Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive Machine Learning Web Application — <b>Intern Circle Project 05</b></div>", unsafe_allow_html=True)

    # Sidebar: Input parameters
    st.sidebar.header("⚙️ Passenger Profile Settings")
    st.sidebar.markdown("Adjust the sliders and options below to predict the passenger's likelihood of survival.")

    pclass_label = st.sidebar.selectbox(
        "🎟️ Ticket Class (Pclass):",
        options=["1st Class (Upper / Luxury)", "2nd Class (Middle / Business)", "3rd Class (Lower / Standard)"],
        index=0
    )
    pclass_map = {
        "1st Class (Upper / Luxury)": 1,
        "2nd Class (Middle / Business)": 2,
        "3rd Class (Lower / Standard)": 3
    }
    pclass = pclass_map[pclass_label]

    sex = st.sidebar.radio(
        "👤 Passenger Gender:",
        options=["female", "male"],
        format_func=lambda x: "Female 👩" if x == "female" else "Male 👨",
        horizontal=True
    )

    age = st.sidebar.slider(
        "🎂 Passenger Age (Years):",
        min_value=0.5,
        max_value=80.0,
        value=28.0,
        step=0.5
    )

    fare = st.sidebar.slider(
        "💳 Ticket Fare (£):",
        min_value=0.0,
        max_value=512.0,
        value=65.0 if pclass == 1 else (25.0 if pclass == 2 else 10.0),
        step=1.0,
        help="Typical 1st class fares: £50-£500; 2nd class: £15-£40; 3rd class: £7-£15."
    )

    st.sidebar.subheader("👨‍👩‍👧 Family Onboard")
    sibsp = st.sidebar.number_input("Siblings / Spouses Aboard:", min_value=0, max_value=8, value=0, step=1)
    parch = st.sidebar.number_input("Parents / Children Aboard:", min_value=0, max_value=8, value=0, step=1)
    family_size = sibsp + parch + 1

    embarked_label = st.sidebar.selectbox(
        "⚓ Port of Embarkation:",
        options=["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
        index=2
    )
    embarked_map = {
        "Cherbourg (C)": "C",
        "Queenstown (Q)": "Q",
        "Southampton (S)": "S"
    }
    embarked = embarked_map[embarked_label]

    # Main Layout: Two Columns
    col1, col2 = st.columns([1, 1], gap="large")

    # Construct input dataframe
    input_data = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "fare": [fare],
        "embarked": [embarked],
        "family_size": [family_size]
    })

    with col1:
        st.subheader("📋 Configured Passenger Summary")
        summary_table = pd.DataFrame({
            "Feature": ["Passenger Class", "Gender", "Age", "Ticket Fare", "Family Size", "Embarkation Port"],
            "Value": [
                f"{pclass} ({pclass_label.split(' ')[0]})",
                sex.capitalize(),
                f"{age:.1f} years",
                f"£{fare:.2f}",
                f"{family_size} person(s)",
                f"{embarked_label.split(' ')[0]} ({embarked})"
            ]
        })
        st.table(summary_table)

        st.markdown("""
        **ℹ️ About the Machine Learning Model:**
        - **Algorithm:** Supervised Decision Tree Classifier
        - **Pipeline:** Automated StandardScaler & One-Hot Encoding
        - **Trained on:** 712 passenger records (80/20 train/test split)
        - **Benchmark Accuracy:** ~81.0% with 5-fold cross validation
        """)

    with col2:
        st.subheader("🔮 Real-Time Prediction Output")
        
        # Make real-time prediction
        probabilities = model.predict_proba(input_data)[0]
        prob_not_survived = probabilities[0]
        prob_survived = probabilities[1]
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class='survived-card'>
                <h2 style='color: #065F46; margin: 0;'>🎉 PREDICTED: SURVIVED</h2>
                <h3 style='color: #047857; margin: 10px 0;'>Estimated Survival Probability: {prob_survived * 100:.1f}%</h3>
                <p style='color: #065F46;'>The model predicts this passenger would have likely secured a spot on a lifeboat under standard maritime evacuation protocols.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='not-survived-card'>
                <h2 style='color: #991B1B; margin: 0;'>⚠️ PREDICTED: DID NOT SURVIVE</h2>
                <h3 style='color: #B91C1C; margin: 10px 0;'>Estimated Survival Probability: {prob_survived * 100:.1f}%</h3>
                <p style='color: #991B1B;'>The model estimates high mortality risk based on historical lifeboat priority and passenger cabin location.</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("**Probability Gauge:**")
        st.progress(float(prob_survived))

        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.metric("Survival Chance", f"{prob_survived * 100:.1f}%")
        with col_prob2:
            st.metric("Fatality Risk", f"{prob_not_survived * 100:.1f}%")

    st.markdown("---")

    # Exploratory Scenario Simulator
    st.subheader("💡 Interactive 'What-If' Scenario Analysis")
    st.markdown("Discover how changing a single variable alters the survival odds:")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        st.markdown("**1. What if passenger was 1st Class?**")
        alt_class = input_data.copy()
        alt_class["pclass"] = 1
        alt_class["fare"] = max(fare, 80.0)
        prob_alt_class = model.predict_proba(alt_class)[0][1] * 100
        delta_class = prob_alt_class - (prob_survived * 100)
        st.metric(label="1st Class Survival Chance", value=f"{prob_alt_class:.1f}%", delta=f"{delta_class:+.1f}%")

    with sim_col2:
        st.markdown("**2. What if passenger was Female?**")
        alt_sex = input_data.copy()
        alt_sex["sex"] = "female"
        prob_alt_sex = model.predict_proba(alt_sex)[0][1] * 100
        delta_sex = prob_alt_sex - (prob_survived * 100)
        st.metric(label="Female Survival Chance", value=f"{prob_alt_sex:.1f}%", delta=f"{delta_sex:+.1f}%")

    with sim_col3:
        st.markdown("**3. What if traveling with Small Family (3)?**")
        alt_fam = input_data.copy()
        alt_fam["family_size"] = 3
        prob_alt_fam = model.predict_proba(alt_fam)[0][1] * 100
        delta_fam = prob_alt_fam - (prob_survived * 100)
        st.metric(label="Small Family Survival Chance", value=f"{prob_alt_fam:.1f}%", delta=f"{delta_fam:+.1f}%")

    st.markdown("---")
    st.caption("Intern Circle Data Science Internship Track — Project 05 | Developed with Streamlit & Scikit-Learn")


if __name__ == "__main__":
    main()
