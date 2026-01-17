# =====================================================
# ETAPE 6 : INTERFACE STREAMLIT
# Application de prediction du risque COVID
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------
# TITRE DE L'APPLICATION
# -----------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Risk Prediction",
    page_icon="🦠",
    layout="centered"
)

st.markdown(
    "<h1 style='color:#7A3DB8;'>🦠 Prédiction du risque COVID-19</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size:16px;'>Cette application estime si un patient présente un risque élevé "
    "en fonction de ses caractéristiques médicales.</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------------------------
# CHARGEMENT DES DONNEES
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/covid19_data.csv")
    df[df.columns.drop('AGE')] = df[df.columns.drop('AGE')].replace([97, 98, 99], np.nan)
    df = df.drop(columns=['ICU', 'INTUBED'])
    df = df.dropna()
    df['high_risk'] = ((df['PNEUMONIA'] == 1) & (df['PATIENT_TYPE'] == 1)).astype(int)
    return df

df = load_data()

# -----------------------------------------------------
# ENTRAINEMENT DU MODELE
# -----------------------------------------------------
features = [
    'AGE', 'SEX', 'PNEUMONIA', 'DIABETES', 'COPD',
    'ASTHMA', 'INMSUPR', 'HIPERTENSION',
    'OBESITY', 'RENAL_CHRONIC', 'TOBACCO'
]

X = df[features]
y = df['high_risk']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------------------------------------
# INTERFACE UTILISATEUR
# -----------------------------------------------------
st.subheader("🧾 Informations du patient")

age = st.slider("Âge du patient", 0, 100, 30)
sex = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")

pneumonia = st.selectbox("Pneumonie", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
diabetes = st.selectbox("Diabète", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
copd = st.selectbox("COPD", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
asthma = st.selectbox("Asthme", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
inmsupr = st.selectbox("Immunodépression", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
hipertension = st.selectbox("Hypertension", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
obesity = st.selectbox("Obésité", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
renal = st.selectbox("Maladie rénale chronique", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
tobacco = st.selectbox("Tabagisme", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
input_data = np.array([[age, sex, pneumonia, diabetes, copd,
                        asthma, inmsupr, hipertension,
                        obesity, renal, tobacco]])

if st.button("🔍 Lancer la prédiction"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Patient à haut risque (Probabilité : {proba*100:.2f}%)")
    else:
        st.success(f"✅ Patient à faible risque (Probabilité : {proba*100:.2f}%)")

st.divider()
st.caption("Projet Machine Learning – COVID-19 | Étape 6")
