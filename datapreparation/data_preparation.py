"""
data_preparation.py - Script complet de préparation des données (version améliorée)
"""

import pandas as pd
import numpy as np
import re
import json
from urllib.parse import urlparse
from datetime import datetime
import warnings
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import langdetect
from typing import Dict, List, Optional

# Configuration des warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

class JobDataPreprocessor:
    def __init__(self):
        """Initialise les chemins des fichiers"""
        self.input_path = "../data/fake_job_postings.csv"
        self.output_path = "../data/clean_jobs.csv"
        self.keywords_path = "../config/enhanced_fraud_keywords.json"  # Fichier amélioré
        self.domains_path = "../config/blacklisted_domains.json"
        self.stopwords_path = "../config/stopword.json"
        
    def load_data(self) -> bool:
        """Charge les données et configurations"""
        print("\n=== CHARGEMENT DES DONNÉES ===")
        try:
            self.df = pd.read_csv(self.input_path)
            
            # Vérification des colonnes requises
            required_cols = ['job_id', 'title', 'description', 'fraudulent']
            missing = [col for col in required_cols if col not in self.df.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes: {missing}")
                
            # Chargement des configurations
            with open(self.keywords_path) as f:
                self.fraud_keywords = json.load(f)  # Dictionnaire avec catégories
                
            with open(self.domains_path) as f:
                self.blacklisted_domains = json.load(f)
                
            with open(self.stopwords_path) as f:
                self.stopwords = json.load(f)
                
            print(f"✅ Données chargées: {len(self.df)} offres d'emploi")
            return True
            
        except Exception as e:
            print(f"❌ Erreur de chargement: {str(e)}")
            return False

    def clean_text(self, text: str) -> str:
        """Nettoie et normalise le texte de manière plus approfondie"""
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)  # Supprime les balises HTML
        text = re.sub(r'[^\w\s]', ' ', text)  # Supprime la ponctuation
        text = re.sub(r'\d+', ' ', text)  # Supprime les nombres
        text = re.sub(r'\s+', ' ', text).strip()  # Normalise les espaces
        
        # Suppression des stopwords
        words = [word for word in text.split() if word not in self.stopwords]
        return ' '.join(words)

    def create_missing_indicators(self) -> None:
        """Crée des indicateurs pour les valeurs manquantes"""
        print("\n=== INDICATEURS DE DONNÉES MANQUANTES ===")
        
        indicators = {
            'missing_description': 'description',
            'missing_profile': 'company_profile',
            'missing_requirements': 'requirements',
            'missing_benefits': 'benefits',
            'missing_salary': 'salary_range'
        }
        
        for col, source in indicators.items():
            if source in self.df.columns:
                self.df[col] = self.df[source].isna().astype(int)
                print(f"{col}: {self.df[col].mean():.1%} manquants")
            else:
                self.df[col] = 1
                print(f"⚠️ {source} non trouvé - marqué comme manquant")

    def create_text_features(self) -> None:
        """Crée des features basées sur le texte de manière plus complète"""
        print("\n=== FEATURES TEXTUELLES AMÉLIORÉES ===")
        
        # Nettoyage du texte
        self.df['clean_desc'] = self.df['description'].progress_apply(self.clean_text)
        self.df['clean_title'] = self.df['title'].progress_apply(self.clean_text)
        
        # Longueur du texte
        self.df['desc_length'] = self.df['clean_desc'].apply(len)
        self.df['title_length'] = self.df['clean_title'].apply(len)
        
        # Détection de langue
        def detect_lang(text):
            try:
                return langdetect.detect(text) if text else 'unknown'
            except:
                return 'unknown'
        
        self.df['description_lang'] = self.df['clean_desc'].apply(detect_lang)
        self.df['is_english'] = (self.df['description_lang'] == 'en').astype(int)
        
        # Compte des mots-clés frauduleux par catégorie
        for category, keywords in self.fraud_keywords.items():
            self.df[f'kw_{category}'] = self.df['clean_desc'].progress_apply(
                lambda x: sum(x.count(kw) for kw in keywords)
            )
        
        # Similarité titre-description
        tfidf = TfidfVectorizer(max_features=500)
        title_vectors = tfidf.fit_transform(self.df['clean_title'])
        desc_vectors = tfidf.transform(self.df['clean_desc'])
        
        self.df['title_desc_similarity'] = cosine_similarity(title_vectors, desc_vectors).diagonal()
        
        # Qualité du texte
        self.df['desc_quality'] = self.df['clean_desc'].apply(
            lambda x: len(x.split()) / (1 + x.count('!') + x.count('urgent'))
        )
        
        print("✅ Features textuelles avancées créées")

    def create_url_features(self) -> None:
        """Analyse des URLs avec plus de détails"""
        if 'job_url' not in self.df.columns:
            print("\n⚠️ Colonne 'job_url' absente - skip des features URL")
            return
            
        print("\n=== ANALYSE DES URLS APPROFONDIE ===")
        
        self.df['url_domain'] = self.df['job_url'].apply(
            lambda x: urlparse(str(x)).netloc if pd.notna(x) else ''
        )
        
        # Détection de domaines suspects
        self.df['suspect_domain'] = self.df['url_domain'].progress_apply(
            lambda x: any(dom.lower() in x.lower() for dom in self.blacklisted_domains)
        ).astype(int)
        
        # Longueur du domaine
        self.df['domain_length'] = self.df['url_domain'].apply(len)
        
        # Nombre de sous-domaines
        self.df['subdomain_count'] = self.df['url_domain'].apply(
            lambda x: len(x.split('.')) if x else 0
        )
        
        print(f"✅ {self.df['suspect_domain'].sum()} URLs suspectes détectées")

    def create_salary_features(self) -> None:
        """Analyse des salaires avec plus de contrôles"""
        if 'salary_range' not in self.df.columns:
            print("\n⚠️ Colonne 'salary_range' absente - skip des features salariales")
            return
            
        print("\n=== ANALYSE DES SALAIRES APPROFONDIE ===")
        
        # Extraction des valeurs min/max avec plus de robustesse
        salary_parts = self.df['salary_range'].str.extract(
            r'(\d{1,3}(?:,\d{3})*)(?:\s*-\s*|\s+to\s+)(\d{1,3}(?:,\d{3})*)'
        )
        
        # Nettoyage des valeurs numériques
        for i in [0, 1]:
            salary_parts[i] = salary_parts[i].str.replace(',', '').astype(float)
        
        self.df['min_salary'] = salary_parts[0]
        self.df['max_salary'] = salary_parts[1]
        
        # Calcul des indicateurs
        self.df['salary_range'] = self.df['max_salary'] - self.df['min_salary']
        self.df['salary_ratio'] = np.where(
            self.df['min_salary'] > 0,
            self.df['max_salary'] / self.df['min_salary'],
            np.nan
        )
        
        # Détection d'anomalies plus précise
        self.df['salary_anomaly'] = (
            (self.df['max_salary'] > 200000) | 
            (self.df['min_salary'] < 10000) |
            (self.df['salary_ratio'] > 5) |
            (self.df['salary_range'] > 100000)
        ).astype(int)
        
        print("✅ Features salariales avancées créées")

    def create_company_features(self) -> None:
        """Crée des features sur l'entreprise améliorées"""
        print("\n=== FEATURES ENTREPRISE APPROFONDIES ===")
        
        # Complétude du profil
        self.df['complete_profile'] = (
            (~self.df['company_profile'].isna()).astype(int) +
            self.df.get('has_company_logo', pd.Series(0)) +
            self.df.get('has_questions', pd.Series(0))
        ) / 3
        
        # Télétravail
        if 'telecommuting' in self.df.columns:
            self.df['telecommuting'] = self.df['telecommuting'].fillna(0).astype(int)
        else:
            self.df['telecommuting'] = 0
            
        # Âge de l'entreprise (si info disponible)
        if 'founded_year' in self.df.columns:
            current_year = datetime.now().year
            self.df['company_age'] = current_year - self.df['founded_year']
            self.df['company_age'] = self.df['company_age'].apply(lambda x: x if x > 0 else 0)
        else:
            self.df['company_age'] = -1  # Valeur manquante
            
        print("✅ Features entreprise avancées créées")

    def add_temporal_features(self) -> None:
        """Ajoute des features temporelles si disponibles"""
        date_cols = ['posting_date', 'deadline']
        
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                
                # Features temporelles
                self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
                self.df[f'{col}_hour'] = self.df[col].dt.hour
                self.df[f'{col}_is_weekend'] = (self.df[col].dt.dayofweek >= 5).astype(int)
                
                # Durée de validité si deadline disponible
                if col == 'posting_date' and 'deadline' in self.df.columns:
                    self.df['job_duration_days'] = (self.df['deadline'] - self.df['posting_date']).dt.days
                    
        print("✅ Features temporelles ajoutées")

    def save_processed_data(self) -> None:
        """Sauvegarde les données préparées avec plus de features"""
        print("\n=== SAUVEGARDE ===")
        
        # Colonnes finales à garder
        features = [
            'job_id', 'title', 'location', 'department',
            'missing_description', 'missing_profile',
            'desc_length', 'title_length', 
            'kw_money', 'kw_urgent', 'kw_payment',  # Exemples de catégories
            'title_desc_similarity', 'desc_quality',
            'is_english', 'suspect_domain', 
            'domain_length', 'subdomain_count',
            'salary_anomaly', 'salary_ratio',
            'complete_profile', 'telecommuting', 'company_age',
            'posting_date_dayofweek', 'posting_date_is_weekend',
            'job_duration_days',
            'fraudulent'
        ]
        
        # Sélection des colonnes existantes
        keep_cols = [col for col in features if col in self.df.columns]
        final_df = self.df[keep_cols]
        
        # Sauvegarde
        final_df.to_csv(self.output_path, index=False)
        
        print(f"✅ Données sauvegardées dans {self.output_path}")
        print(f"📊 Dimensions: {final_df.shape}")
        print(f"💰 Taux de fraude: {final_df['fraudulent'].mean():.2%}")

    def run_pipeline(self) -> bool:
        """Exécute le pipeline complet amélioré"""
        if not self.load_data():
            return False
            
        try:
            self.create_missing_indicators()
            self.create_text_features()
            self.create_url_features()
            self.create_salary_features()
            self.create_company_features()
            self.add_temporal_features()
            self.save_processed_data()
            return True
        except Exception as e:
            print(f"❌ Erreur lors du traitement: {str(e)}")
            return False

if __name__ == "__main__":
    print("""
    ====================================
    PRÉPARATION DES DONNÉES - DÉTECTION DE FRAUDE (VERSION AMÉLIORÉE)
    ====================================
    """)
    
    processor = JobDataPreprocessor()
    success = processor.run_pipeline()
    
    print("\n" + "="*50)
    print("✅ PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS" if success else "❌ PRÉTRAITEMENT ÉCHOUÉ")
    print("="*50)