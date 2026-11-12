# 🚢 Titanic Survival Predictor — Interactive Streamlit ML Web App

**Organization / Program:** Intern Circle  
**Track:** Data Science & Machine Learning  
**Project:** Project 05 — Streamlit / Gradio Interactive ML Web App  

---

## 📌 Project Overview
An interactive, end-to-end deployed **Machine Learning Web Application** built using **Streamlit**, **Scikit-Learn**, and **Joblib**.

This application deploys a pre-trained **Decision Tree Classifier** pipeline to deliver real-time survival predictions for historical Titanic passenger profiles. Users can interactively adjust passenger class, age, gender, ticket fare, family size, and embarkation port through intuitive input controls and instantly observe probability distributions, fatality risks, and "what-if" scenario simulations.

---

## 🎯 Requirements & Key Features

| Requirement | Implementation Details | Status |
| :--- | :--- | :---: |
| **Streamlit Interface Creation** | Wide layout, responsive sidebar, styled prediction status cards (Emerald Green / Crimson Red), and custom CSS | ✅ Complete |
| **Model Loading with Joblib** | Efficient cached loading (`@st.cache_resource`) of the serialized Scikit-Learn `Pipeline` | ✅ Complete |
| **Interactive Prediction Display** | Binary outcome card (*Predicted: Survived* vs. *Predicted: Did Not Survive*), real-time probability gauge, and risk metrics | ✅ Complete |
| **Real-time Input Sliders** | Continuous sliders for Age (0.5 to 80.0 yrs), Fare (£0 to £512), and onboard Family count inputs | ✅ Complete |
| **Scenario Simulator** | Live sensitivity simulator showing immediate delta changes in survival odds (e.g., *What if Female?*, *What if 1st Class?*) | ✅ Complete |

---

## 🧠 Model Architecture & Pipeline
The web app executes a pre-trained Scikit-Learn pipeline encapsulating:
- **Numerical Scaling:** `StandardScaler` for `age`, `fare`, and `family_size`.
- **Categorical Encoding:** `OneHotEncoder(drop="first")` for `pclass`, `sex`, and `embarked`.
- **Classification Estimator:** `DecisionTreeClassifier(max_depth=4, min_samples_split=10)` trained on 712 records, achieving **~81.0% accuracy** with 5-fold cross-validation.

---

## 🖥️ User Interface & Control Flow

1. **Passenger Profile Settings (Sidebar):**
   - **Ticket Class:** 1st Class (Luxury), 2nd Class (Business), 3rd Class (Economy)
   - **Gender:** Female 👩 / Male 👨
   - **Age:** Slider from 0.5 to 80.0 years
   - **Ticket Fare:** Slider from £0 to £512 (with contextual typical fare benchmarks)
   - **Family Onboard:** Number inputs for siblings, spouses, parents, and children
   - **Embarkation Port:** Cherbourg (C), Queenstown (Q), Southampton (S)
2. **Real-time Prediction Output:**
   - Visual banner indicating prediction outcome.
   - Interactive horizontal probability progress bar.
   - Dual metrics: **Survival Chance %** vs. **Fatality Risk %**.
3. **Interactive 'What-If' Comparative Scenario Analysis:**
   - Instant delta calculation comparing baseline odds against alternate scenarios (e.g., upgrading to 1st class or traveling with a small family).

---

## 📂 Repository Structure

```
interactive-ml-prediction-app/
├── models/
│   └── titanic_classifier.joblib         # Serialized Scikit-Learn ML pipeline
├── data/
│   └── titanic.csv                       # Historical Titanic passenger manifest
├── app.py                                # Main Streamlit web application
├── interactive_ml_webapp.ipynb           # Model verification & architecture notebook
├── requirements.txt                      # Python dependencies for deployment
├── .gitignore                            # Git ignore rules
└── README.md                             # Comprehensive application documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/IMRIKRUPA/interactive-ml-prediction-app.git
cd interactive-ml-prediction-app
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit application
```bash
streamlit run app.py
```

### 4. Open in browser
The web application will automatically launch at:
```
http://localhost:8501
```

---

## 🌐 Free Cloud Deployment (Streamlit Community Cloud)

You can deploy this web application publicly for free in less than 2 minutes:

1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New App"**.
3. Configure the settings:
   - **Repository:** `IMRIKRUPA/interactive-ml-prediction-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy"**! Streamlit will install all packages from `requirements.txt` and generate your live public URL.
