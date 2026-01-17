import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="COVID-19 Risk Prediction",
    page_icon="🦠",
    layout="wide" # Mode large pour l'effet "Split Screen"
)

# 2. STYLE CSS PERSONNALISÉ (Inspiré de l'image Dribbble)
st.markdown("""
    <style>
    /* Fond dégradé doux */
    .stApp {
        background: linear-gradient(135deg, #FFF5E6 0%, #FFDFD3 100%);
    }
    /* Style du titre violet */
    h1 {
        color: #7A3DB8 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    /* Carte blanche pour le formulaire */
    .stColumn > div > div > div {
        background-color: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    /* Style du bouton */
    .stButton>button {
        background-color: #7A3DB8;
        color: white;
        border-radius: 10px;
        width: 100%;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. CHARGEMENT ET MODÈLE (Caché pour l'utilisateur)
@st.cache_data
def load_and_train():
    df = pd.read_csv("data/covid19_data.csv")
    df[df.columns.drop('AGE')] = df[df.columns.drop('AGE')].replace([97, 98, 99], np.nan)
    df = df.dropna()
    # Variable cible selon vos critères
    df['high_risk'] = ((df['PNEUMONIA'] == 1) & (df['PATIENT_TYPE'] == 1)).astype(int)
    
    features = ['AGE', 'SEX', 'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 
                'INMSUPR', 'HIPERTENSION', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(df[features], df['high_risk'])
    return model

model = load_and_train()

# 4. MISE EN PAGE "SPLIT" (Gauche: Illustration / Droite: Formulaire)
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("Predict the world's patient safety.") # Clin d'oeil à votre image
    st.write("Utilisez notre intelligence artificielle pour évaluer instantanément les facteurs de risque liés au COVID-19.")
    # On simule l'illustration avec une icône ou une image
    st.image("https://img.freepik.com/free-vector/medical-technology-concept-illustration_114360-6363.jpg", use_container_width=True)

with col2:
    st.markdown("### Sign up for Analysis")
    
    # On regroupe les inputs pour le design
    age = st.slider("Âge du patient", 0, 100, 30)
    
    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")
        pneumonia = st.selectbox("Pneumonie", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        diabetes = st.selectbox("Diabète", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
    with c2:
        hipertension = st.selectbox("Hypertension", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        obesity = st.selectbox("Obésité", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")
        tobacco = st.selectbox("Tabagisme", [1, 2], format_func=lambda x: "Oui" if x == 1 else "Non")

    # Bouton de prédiction
    if st.button("Create Prediction"):
        input_data = np.array([[age, sex, pneumonia, diabetes, 2, 2, 2, hipertension, obesity, 2, tobacco]])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"Alerte : Haut Risque ({proba*100:.1f}%)")
        else:
            st.success(f"Sécurité : Risque Faible ({proba*100:.1f}%)")

st.markdown("<p style='text-align: center; color: grey; font-size: 12px;'>Already a member? Sign in</p>", unsafe_allow_html=True)
