import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="COVID-19 Risk Prediction",
    page_icon="🦠",
    layout="wide" 
)

# 2. STYLE CSS PERSONNALISÉ (Look Dribbble Pastel & Violet)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #FFF5E6 0%, #FFDFD3 100%);
    }
    h1 {
        color: #7A3DB8 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    /* Style de la carte formulaire */
    [data-testid="stVerticalBlock"] > div:has(div.stButton) {
        background-color: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #7A3DB8;
        color: white;
        border-radius: 10px;
        width: 100%;
        font-weight: bold;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. CHARGEMENT ET ENTRAÎNEMENT SÉCURISÉ
@st.cache_data
def load_and_train():
    # Chargement
    df = pd.read_csv("data/covid19_data.csv")
    
    # Nettoyage des valeurs 97, 98, 99 (Inconnu)
    cols_medicales = ['SEX', 'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 
                      'INMSUPR', 'HIPERTENSION', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']
    df[cols_medicales] = df[cols_medicales].replace([97, 98, 99], np.nan)
    
    # Création de la variable cible (Pneumonie + Hospitalisé)
    # PATIENT_TYPE: 2 = Hospitalisé, 1 = Retour maison
    df['high_risk'] = ((df['PNEUMONIA'] == 1) & (df['PATIENT_TYPE'] == 2)).astype(int)
    
    features = ['AGE'] + cols_medicales
    
    # Suppression des lignes avec des valeurs manquantes (Évite la ValueError)
    df = df.dropna(subset=features + ['high_risk'])
    
    # Entraînement sur un échantillon pour la rapidité Streamlit
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
        
    model = LogisticRegression(max_iter=1000)
    model.fit(df[features], df['high_risk'])
    return model

# Initialisation du modèle
try:
    model = load_and_train()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

# 4. MISE EN PAGE "SPLIT SCREEN"
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("Predict the world's patient safety.")
    st.write("Évaluez instantanément les risques de complications grâce à notre algorithme basé sur 1 million de dossiers médicaux.")
    st.image("https://img.freepik.com/free-vector/medical-technology-concept-illustration_114360-6363.jpg")

with col2:
    st.markdown("### Patient Health Profile")
    
    age = st.slider("Âge du patient", 0, 100, 30)
    
    # Grille de sélection
    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")
        pneumonia = st.selectbox("Pneumonie", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        diabetes = st.selectbox("Diabète", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        copd = st.selectbox("BPCO (COPD)", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        asthma = st.selectbox("Asthme", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
    with c2:
        inmsupr = st.selectbox("Immunodéprimé", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        hipertension = st.selectbox("Hypertension", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        obesity = st.selectbox("Obésité", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        renal = st.selectbox("Maladie Rénale", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        tobacco = st.selectbox("Tabagisme", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")

    if st.button("Create Prediction"):
        # Données d'entrée pour le modèle
        input_data = np.array([[age, sex, pneumonia, diabetes, copd, asthma, inmsupr, hipertension, obesity, renal, tobacco]])
        
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **Haut Risque Détecté** (Probabilité : {proba*100:.1f}%)")
        else:
            st.success(f"✅ **Risque Faible** (Probabilité : {proba*100:.1f}%)")

st.markdown("<p style='text-align: center; color: grey; font-size: 12px; margin-top:50px;'>Projet Python Groupe 5 - Analyse COVID-19</p>", unsafe_allow_html=True)
