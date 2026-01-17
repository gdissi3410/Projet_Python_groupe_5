# Analyse des données COVID-19 et prédiction des risques de santé

##  Description du projet

Ce projet vise à développer un modèle de machine learning pour prédire le risque de complications graves chez les patients atteints de COVID-19. En utilisant les données fournies par le gouvernement mexicain, nous analysons les symptômes, l'état de santé et les antécédents médicaux des patients pour établir des prédictions précises du niveau de risque.

##  Objectif

Construire un modèle d'apprentissage automatique capable de :
- Analyser les données de plus d'un million de patients
- Identifier les facteurs de risque les plus significatifs
- Prédire si un patient est à haut risque de complications liées au COVID-19
- Aider les autorités sanitaires à allouer efficacement les ressources médicales

##  Données

### Source
Ensemble de données fourni par le gouvernement mexicain contenant :
- **1 048 576** patients uniques
- **21 caractéristiques** cliniques et démographiques
- Informations sur les conditions préalables, symptômes et résultats des tests

### Caractéristiques principales
- `sex` : Sexe du patient (homme/femme)
- `age` : Âge du patient
- `classification` : Résultats du test COVID (1-3 = positif, 4+ = négatif/inconclus)
- `patient_type` : Hospitalisé ou non
- `pneumonia`, `diabetes`, `obesity`, `hypertension` : Conditions médicales
- `asthma`, `copd`, `inmsupr`, `cardiovascular` : Maladies chronicques
- `tobacco`, `pregnancy` : Autres facteurs de risque
- `intubed`, `icu` : Interventions médicales
- Et bien d'autres...

### Traitement des données
- Valeur `1` = Oui
- Valeur `2` = Non
- Valeurs `97, 99` = Données manquantes (à traiter)

##  Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Dépendances
```bash
pip install -r requirements.txt
```

Les packages requis incluent :
- **pandas** : Manipulation et analyse de données
- **numpy** : Calculs numériques
- **matplotlib** & **seaborn** : Visualisation des données
- **scikit-learn** : Modèles de machine learning
- **streamlit** : Interface interactive (optionnel)

##  Structure du projet

```
.
├── README.md                    # Ce fichier
├── Projet_Python.ipynb         # Notebook Jupyter principal
├── requirements.txt            # Dépendances Python
├── covid19_final_ready.csv     # Données nettoyées (format final)
├── data/
│   └── covid19_data.csv        # Données brutes originales
```

##  Étapes du projet

### 1. Préparation et nettoyage des données
- Chargement du jeu de données dans un DataFrame Pandas
- Exploration des dimensions, plages de valeurs
- Traitement des données manquantes et invalides
- Suppression des doublons
- Création de colonnes additionnelles si nécessaire

### 2. Analyse exploratoire des données (EDA)
- Statistiques descriptives
- Visualisations des distributions
- Analyse des corrélations
- Identification des tendances et anomalies

### 3. Préparation pour le machine learning
- Sélection des features pertinentes
- Normalisation/standardisation des données
- Séparation train/test

### 4. Modélisation
- Entraînement de plusieurs modèles :
  - Forêts aléatoires (Random Forest)
  - Support Vector Classifier (SVC)
  - Naive Bayes Gaussien
- Évaluation et comparaison des performances
- Optimisation des hyperparamètres

### 5. Interprétation et insights
- Analyse de l'importance des features
- Matrice de confusion et métriques de performance
- Recommandations basées sur les résultats

##  Utilisation

### Exécuter le notebook
```bash
jupyter notebook Projet_Python.ipynb
```

### Ou utiliser Streamlit (si implémenté)
```bash
streamlit run app.py
```

##  Résultats attendus

Le projet fournira :
- Un modèle entraîné capable de prédire le risque de complications COVID-19
- Une analyse détaillée des facteurs de risque les plus importants
- Des visualisations explorant les relations entre variables
- Des recommandations pour l'allocation des ressources médicales

##  Références

- Source des données : Gouvernement mexicain
- Contexte : Pandémie COVID-19 2020-2021
- Problématique : Allocation efficace des ressources médicales

##  Auteur

Projet développé dans le cadre d'une formation en Python et Data Science.

##  Notes

- Le fichier `covid19_final_ready.csv` contient les données nettoyées et prêtes à l'analyse
- Consulter le notebook pour tous les détails des analyses effectuées
- Les valeurs manquantes (97, 99) ont été traitées lors du nettoyage des données

---

**Dernière mise à jour** : Janvier 2026
