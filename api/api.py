from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import re
from typing import Optional
import numpy as np

app = FastAPI()

# Modèle qui correspond à votre CSV original
class RawJobPosting(BaseModel):
    job_id: Optional[int] = None
    title: str
    location: str
    department: Optional[str] = None
    salary_range: Optional[str] = None
    company_profile: Optional[str] = None
    description: str
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    telecommuting: int
    has_company_logo: int
    has_questions: int
    employment_type: Optional[str] = None
    required_experience: Optional[str] = None
    required_education: Optional[str] = None
    industry: Optional[str] = None
    function: Optional[str] = None

# Chargement du modèle et du scaler
try:
    model = joblib.load('../models_training_test/best_model.pkl')
    scaler = joblib.load('../models_training_test/scaler.pkl')
    print("✅ Modèle et scaler chargés avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle/scaler: {str(e)}")
    raise e

def preprocess_data(raw: dict):
    """Transforme les données brutes en features pour le modèle"""
    # Calcul des indicateurs manquants
    missing_description = int(not raw.get('description', ''))
    missing_profile = int(not raw.get('company_profile', ''))
    
    # Features textuelles
    clean_desc = re.sub(r'[^\w\s]', '', str(raw.get('description', '')).lower())
    desc_length = len(clean_desc)
    
    # Mots-clés frauduleux (adaptés à votre fichier fraud_keywords.json)
    fraud_keywords = [
        "travail à domicile", "gagner argent", "sans expérience", 
        "opportunité unique", "revenu facile", "argent rapide",
        "travail chez soi", "emploi immédiat", "sans diplôme"
    ]
    fraud_keywords_count = sum(clean_desc.count(kw) for kw in fraud_keywords)
    
    # Qualité de la description
    word_count = len(clean_desc.split())
    exclamation_count = clean_desc.count('!')
    desc_quality = word_count / (1 + exclamation_count) if exclamation_count > 0 else word_count
    
    # Features salariales
    salary_anomaly = 0
    if raw.get('salary_range'):
        try:
            parts = str(raw['salary_range']).split('-')
            if len(parts) == 2:
                min_sal, max_sal = map(float, parts)
                salary_anomaly = int(max_sal > 200000 or min_sal < 10000)
        except:
            pass
    
    # Complétude du profil
    complete_profile = (
        raw.get('has_company_logo', 0) + 
        (1 if raw.get('company_profile') else 0) + 
        raw.get('has_questions', 0)
    ) / 3
    
    # Télétravail
    telecommuting = raw.get('telecommuting', 0)
    
    # Création du DataFrame avec les features dans le bon ordre
    features = pd.DataFrame([{
        "missing_description": missing_description,
        "missing_profile": missing_profile,
        "desc_length": desc_length,
        "fraud_keyword_count": fraud_keywords_count,
        "desc_quality": desc_quality,
        "salary_anomaly": salary_anomaly,
        "complete_profile": complete_profile,
        "telecommuting": telecommuting
    }])
    
    # Ordre exact des features attendues par le modèle
    expected_columns = [
        "missing_description",
        "missing_profile",
        "desc_length",
        "fraud_keyword_count",
        "desc_quality",
        "salary_anomaly",
        "complete_profile",
        "telecommuting"
    ]
    
    # Vérification et réorganisation des colonnes
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0  # Valeur par défaut si colonne manquante
    
    return features[expected_columns]

@app.post("/predict")
async def predict(posting: RawJobPosting):
    try:
        # Transformation des données
        features = preprocess_data(posting.dict())
        
        # Vérification des features
        if features.isnull().values.any():
            features = features.fillna(0)  # Remplace les NaN par 0
            
        # Journalisation pour débogage
        print("Features avant scaling:", features.to_dict('records')[0])
        
        # Scaling des features
        scaled_features = scaler.transform(features)
        
        # Prédiction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0, 1]
        
        return {
            "fraud_prediction": bool(prediction),
            "fraud_probability": float(probability),
            "processed_features": features.to_dict('records')[0]
        }
    except Exception as e:
        return {"error": str(e), "details": f"Features reçues: {features.to_dict() if 'features' in locals() else 'N/A'}"}

@app.get("/")
async def root():
    return {"message": "API de détection d'offres d'emploi frauduleuses"}

@app.get("/model-info")
async def model_info():
    """Endpoint pour obtenir des informations sur le modèle chargé"""
    if not model:
        return {"error": "Modèle non chargé"}
    
    return {
        "model_type": str(type(model)),
        "features_expected": [
            "missing_description",
            "missing_profile",
            "desc_length",
            "fraud_keyword_count",
            "desc_quality",
            "salary_anomaly",
            "complete_profile",
            "telecommuting"
        ]
    }