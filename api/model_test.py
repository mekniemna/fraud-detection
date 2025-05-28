# Charger le modèle et le scaler comme dans votre API
model = joblib.load('../models_training_test/best_model.pkl')
scaler = joblib.load('../models_training_test/scaler.pkl')

# Créer un exemple clairement frauduleux
test_data = {
    "missing_description": 0,
    "missing_profile": 1,  # profil manquant
    "desc_length": 50,    # description très courte
    "fraud_keyword_count": 3,  # mots-clés frauduleux
    "desc_quality": 2,    # qualité faible
    "salary_anomaly": 1,  # anomalie salariale
    "complete_profile": 0.3,  # profil incomplet
    "telecommuting": 1    # télétravail
}

# Mettre sous forme de DataFrame
features = pd.DataFrame([test_data])

# Vérifier l'ordre des colonnes
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
features = features[expected_columns]

# Prétraitement et prédiction
scaled_features = scaler.transform(features)
prediction = model.predict(scaled_features)
probability = model.predict_proba(scaled_features)

print("Prediction:", prediction)
print("Probabilité:", probability)