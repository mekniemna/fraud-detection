"""
data_preparation.py - Script complet de préparation des données
"""

import pandas as pd
import numpy as np
import re
import json
from urllib.parse import urlparse
from datetime import datetime
import warnings
from tqdm import tqdm

# Configuration des warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

class JobDataPreprocessor:
    def __init__(self):
        """Initialise les chemins des fichiers"""
        self.input_path = "../data/raw_jobs.csv"
        self.output_path = "../data/clean_jobs.csv"
        self.keywords_path = "../config/fraud_keywords.json"
        self.domains_path = "../config/blacklisted_domains.json"
        
    def load_data(self):
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
                self.fraud_keywords = json.load(f)
                
            with open(self.domains_path) as f:
                self.blacklisted_domains = json.load(f)
                
            print(f"✅ Données chargées: {len(self.df)} offres d'emploi")
            return True
            
        except Exception as e:
            print(f"❌ Erreur de chargement: {str(e)}")
            return False

    def clean_text(self, text):
        """Nettoie et normalise le texte"""
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)  # Supprime les balises HTML
        text = re.sub(r'[^\w\s]', ' ', text)  # Supprime la ponctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalise les espaces
        return text

    def create_missing_indicators(self):
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
                self.df[col] = 1  # Si colonne source absente, considérer comme manquante
                print(f"⚠️ {source} non trouvé - marqué comme manquant")

    def create_text_features(self):
        """Crée des features basées sur le texte"""
        print("\n=== FEATURES TEXTUELLES ===")
        
        # Nettoyage du texte
        self.df['clean_desc'] = self.df['description'].progress_apply(self.clean_text)
        self.df['clean_title'] = self.df['title'].progress_apply(self.clean_text)
        
        # Longueur du texte
        self.df['desc_length'] = self.df['clean_desc'].apply(len)
        self.df['title_length'] = self.df['clean_title'].apply(len)
        
        # Compte des mots-clés frauduleux
        self.df['fraud_keyword_count'] = self.df['clean_desc'].progress_apply(
            lambda x: sum(x.count(kw) for kw in self.fraud_keywords)
        )
        
        # Qualité du texte
        self.df['desc_quality'] = self.df['clean_desc'].apply(
            lambda x: len(x.split()) / (1 + x.count('!'))
        )
        
        print("✅ Features textuelles créées")

    def create_url_features(self):
        """Analyse les URLs suspectes"""
        if 'job_url' not in self.df.columns:
            print("\n⚠️ Colonne 'job_url' absente - skip des features URL")
            return
            
        print("\n=== ANALYSE DES URLS ===")
        
        self.df['url_domain'] = self.df['job_url'].apply(
            lambda x: urlparse(str(x)).netloc if pd.notna(x) else ''
        )
        
        self.df['suspect_domain'] = self.df['url_domain'].progress_apply(
            lambda x: any(dom.lower() in x.lower() for dom in self.blacklisted_domains)
        ).astype(int)
        
        print(f"✅ {self.df['suspect_domain'].sum()} URLs suspectes détectées")

    def create_salary_features(self):
        """Analyse les salaires"""
        if 'salary_range' not in self.df.columns:
            print("\n⚠️ Colonne 'salary_range' absente - skip des features salariales")
            return
            
        print("\n=== ANALYSE DES SALAIRES ===")
        
        # Extraction des valeurs min/max
        salary_parts = self.df['salary_range'].str.extract(r'(\d+)\s*-\s*(\d+)')
        self.df['min_salary'] = pd.to_numeric(salary_parts[0], errors='coerce')
        self.df['max_salary'] = pd.to_numeric(salary_parts[1], errors='coerce')
        
        # Calcul des indicateurs
        self.df['salary_range'] = self.df['max_salary'] - self.df['min_salary']
        self.df['salary_ratio'] = np.where(
            self.df['min_salary'] > 0,
            self.df['max_salary'] / self.df['min_salary'],
            np.nan
        )
        
        # Détection d'anomalies
        self.df['salary_anomaly'] = (
            (self.df['max_salary'] > 200000) | 
            (self.df['min_salary'] < 10000) |
            (self.df['salary_ratio'] > 5)
        ).astype(int)
        
        print("✅ Features salariales créées")

    def create_company_features(self):
        """Crée des features sur l'entreprise"""
        print("\n=== FEATURES ENTREPRISE ===")
        
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
            
        print("✅ Features entreprise créées")

    def save_processed_data(self):
        """Sauvegarde les données préparées"""
        print("\n=== SAUVEGARDE ===")
        
        # Colonnes finales à garder
        features = [
            'job_id', 'title', 'location', 'department',
            'missing_description', 'missing_profile',
            'desc_length', 'fraud_keyword_count', 'desc_quality',
            'suspect_domain', 'salary_anomaly',
            'complete_profile', 'telecommuting',
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

    def run_pipeline(self):
        """Exécute le pipeline complet"""
        if not self.load_data():
            return False
            
        try:
            self.create_missing_indicators()
            self.create_text_features()
            self.create_url_features()
            self.create_salary_features()
            self.create_company_features()
            self.save_processed_data()
            return True
        except Exception as e:
            print(f"❌ Erreur lors du traitement: {str(e)}")
            return False

if __name__ == "__main__":
    print("""
    ====================================
    PRÉPARATION DES DONNÉES - DÉTECTION DE FRAUDE
    ====================================
    """)
    
    processor = JobDataPreprocessor()
    success = processor.run_pipeline()
    
    print("\n" + "="*50)
    print("✅ PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS" if success else "❌ PRÉTRAITEMENT ÉCHOUÉ")
    print("="*50)