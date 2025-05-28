"""
fraud_detection_api.py - API Flask pour la détection d'offres d'emploi frauduleuses
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import json
from urllib.parse import urlparse
from datetime import datetime
import langdetect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin

class FraudDetectionAPI:
    def __init__(self):
        """Initialise l'API avec le modèle pré-entraîné"""
        self.model_artifacts = None
        self.fraud_keywords = {}
        self.blacklisted_domains = []
        self.stopwords = []
        self.load_model_and_configs()
        
    def load_model_and_configs(self):
        """Charge le modèle et les configurations"""
        try:
            # Chargement du modèle et des artefacts
            self.model_artifacts = joblib.load('../models_training_test/best_model_artifacts.pkl')
            logger.info("✅ Modèle chargé avec succès")
            
            # Chargement des configurations (avec valeurs par défaut)
            try:
                with open('../config/enhanced_fraud_keywords.json', 'r') as f:
                    self.fraud_keywords = json.load(f)
            except FileNotFoundError:
                logger.warning("⚠️ Fichier keywords non trouvé, utilisation des valeurs par défaut")
                self.fraud_keywords = {
                    "money": ["urgent", "money", "cash", "payment", "immediate"],
                    "urgent": ["urgent", "asap", "immediately", "rush"],
                    "payment": ["pay", "salary", "earn", "income"]
                }
            
            try:
                with open('../config/blacklisted_domains.json', 'r') as f:
                    self.blacklisted_domains = json.load(f)
            except FileNotFoundError:
                self.blacklisted_domains = ["suspicious.com", "fake-jobs.net"]
                
            try:
                with open('../config/stopword.json', 'r') as f:
                    self.stopwords = json.load(f)
            except FileNotFoundError:
                self.stopwords = ["the", "and", "or", "but", "in", "on", "at", "to", "for"]
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Nettoie et normalise le texte"""
        if pd.isna(text) or not text:
            return ""
            
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)  # Supprime les balises HTML
        text = re.sub(r'[^\w\s]', ' ', text)  # Supprime la ponctuation
        text = re.sub(r'\d+', ' ', text)  # Supprime les nombres
        text = re.sub(r'\s+', ' ', text).strip()  # Normalise les espaces
        
        # Suppression des stopwords
        words = [word for word in text.split() if word not in self.stopwords]
        return ' '.join(words)
    
    def extract_features(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les features à partir des données d'entrée"""
        features = {}
        
        # Récupération des champs avec valeurs par défaut
        title = job_data.get('title', '')
        description = job_data.get('description', '')
        company_profile = job_data.get('company_profile', '')
        requirements = job_data.get('requirements', '')
        benefits = job_data.get('benefits', '')
        salary_range = job_data.get('salary_range', '')
        job_url = job_data.get('job_url', '')
        
        # Nettoyage du texte
        clean_desc = self.clean_text(description)
        clean_title = self.clean_text(title)
        
        # === INDICATEURS DE DONNÉES MANQUANTES ===
        features['missing_description'] = 1 if not description else 0
        features['missing_profile'] = 1 if not company_profile else 0
        features['missing_requirements'] = 1 if not requirements else 0
        features['missing_benefits'] = 1 if not benefits else 0
        features['missing_salary'] = 1 if not salary_range else 0
        
        # === FEATURES TEXTUELLES ===
        features['desc_length'] = len(clean_desc)
        features['title_length'] = len(clean_title)
        
        # Détection de langue
        try:
            lang = langdetect.detect(clean_desc) if clean_desc else 'unknown'
            features['is_english'] = 1 if lang == 'en' else 0
        except:
            features['is_english'] = 0
        
        # Compte des mots-clés frauduleux par catégorie
        for category, keywords in self.fraud_keywords.items():
            features[f'kw_{category}'] = sum(clean_desc.count(kw) for kw in keywords)
        
        # Similarité titre-description (version simplifiée)
        if clean_title and clean_desc:
            title_words = set(clean_title.split())
            desc_words = set(clean_desc.split())
            if title_words and desc_words:
                features['title_desc_similarity'] = len(title_words & desc_words) / len(title_words | desc_words)
            else:
                features['title_desc_similarity'] = 0
        else:
            features['title_desc_similarity'] = 0
        
        # Qualité du texte
        if clean_desc:
            urgent_count = clean_desc.count('urgent') + clean_desc.count('!')
            word_count = len(clean_desc.split())
            features['desc_quality'] = word_count / (1 + urgent_count) if word_count > 0 else 0
        else:
            features['desc_quality'] = 0
        
        # === FEATURES URL ===
        if job_url:
            try:
                parsed_url = urlparse(job_url)
                domain = parsed_url.netloc
                features['suspect_domain'] = 1 if any(dom.lower() in domain.lower() for dom in self.blacklisted_domains) else 0
                features['domain_length'] = len(domain)
                features['subdomain_count'] = len(domain.split('.')) if domain else 0
            except:
                features['suspect_domain'] = 0
                features['domain_length'] = 0
                features['subdomain_count'] = 0
        else:
            features['suspect_domain'] = 0
            features['domain_length'] = 0
            features['subdomain_count'] = 0
        
        # === FEATURES SALARIALES ===
        if salary_range:
            try:
                # Extraction simple des valeurs min/max
                numbers = re.findall(r'\d+', salary_range.replace(',', ''))
                if len(numbers) >= 2:
                    min_sal, max_sal = float(numbers[0]), float(numbers[1])
                    features['salary_ratio'] = max_sal / min_sal if min_sal > 0 else 1
                    features['salary_anomaly'] = 1 if (max_sal > 200000 or min_sal < 10000 or features['salary_ratio'] > 5) else 0
                else:
                    features['salary_ratio'] = 1
                    features['salary_anomaly'] = 0
            except:
                features['salary_ratio'] = 1
                features['salary_anomaly'] = 0
        else:
            features['salary_ratio'] = 1
            features['salary_anomaly'] = 0
        
        # === FEATURES ENTREPRISE ===
        features['complete_profile'] = (
            (1 if company_profile else 0) +
            job_data.get('has_company_logo', 0) +
            job_data.get('has_questions', 0)
        ) / 3
        
        features['telecommuting'] = job_data.get('telecommuting', 0)
        features['company_age'] = job_data.get('company_age', -1)
        
        # === FEATURES TEMPORELLES (simplifiées) ===
        current_hour = datetime.now().hour
        current_dow = datetime.now().weekday()
        
        features['posting_date_dayofweek'] = current_dow
        features['posting_date_is_weekend'] = 1 if current_dow >= 5 else 0
        features['job_duration_days'] = job_data.get('job_duration_days', 30)  # Valeur par défaut
        
        return features
    
    def predict_fraud(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit si une offre d'emploi est frauduleuse"""
        try:
            # Extraction des features
            features = self.extract_features(job_data)
            
            # Conversion en DataFrame pour le preprocessing
            feature_names = self.model_artifacts['features']
            
            # Création d'un vecteur avec toutes les features attendues
            feature_vector = []
            for feature_name in feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            # Conversion en array numpy
            X = np.array(feature_vector).reshape(1, -1)
            
            # Imputation et normalisation
            X_imputed = self.model_artifacts['imputer'].transform(X)
            X_scaled = self.model_artifacts['scaler'].transform(X_imputed)
            
            # Prédiction
            model = self.model_artifacts['model']
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            # Résultat
            result = {
                'is_fraudulent': bool(prediction),
                'fraud_probability': float(probability[1]),
                'legitimate_probability': float(probability[0]),
                'confidence': float(max(probability)),
                'risk_level': self._get_risk_level(probability[1]),
                'extracted_features': features
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise
    
    def _get_risk_level(self, fraud_prob: float) -> str:
        """Détermine le niveau de risque basé sur la probabilité"""
        if fraud_prob < 0.3:
            return "FAIBLE"
        elif fraud_prob < 0.6:
            return "MOYEN"
        elif fraud_prob < 0.8:
            return "ÉLEVÉ"
        else:
            return "TRÈS ÉLEVÉ"

# Initialisation de l'API
detector = FraudDetectionAPI()

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        'message': 'API de détection de fraude d\'offres d\'emploi',
        'version': '1.0',
        'endpoints': {
            'POST /predict': 'Analyse une offre d\'emploi',
            'GET /health': 'Vérification de l\'état de l\'API'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model_artifacts is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal pour la prédiction"""
    try:
        # Vérification du contenu
        if not request.json:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400
        
        job_data = request.json
        
        # Vérification des champs requis
        required_fields = ['title', 'description']
        missing_fields = [field for field in required_fields if not job_data.get(field)]
        
        if missing_fields:
            return jsonify({
                'error': f'Champs requis manquants: {missing_fields}'
            }), 400
        
        # Prédiction
        result = detector.predict_fraud(job_data)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /predict: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint pour traiter plusieurs offres d'emploi en une fois"""
    try:
        if not request.json or 'jobs' not in request.json:
            return jsonify({'error': 'Format invalide. Attendu: {"jobs": [...]}'}), 400
        
        jobs = request.json['jobs']
        results = []
        
        for i, job_data in enumerate(jobs):
            try:
                result = detector.predict_fraud(job_data)
                results.append({
                    'index': i,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(jobs),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /batch_predict: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("""
    ====================================
    API DÉTECTION DE FRAUDE - DÉMARRAGE
    ====================================
    """)
    
    # Démarrage de l'API
    app.run(
        host='0.0.0.0',  # Accessible depuis l'extérieur
        port=5000,       # Port par défaut
        debug=True       # Mode debug pour le développement
    )