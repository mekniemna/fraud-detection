"""
model_training.py - Script de comparaison de modèles (version corrigée)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import joblib

warnings.filterwarnings('ignore')

class FraudDetectionModels:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = pd.DataFrame(columns=[
            'Model', 'Accuracy', 'Precision', 'Recall', 
            'F1', 'ROC AUC', 'Training Time'
        ])
        
    def load_and_prepare_data(self):
        """Charge et prépare les données pour l'entraînement"""
        print("\n=== CHARGEMENT DES DONNÉES ===")
        self.df = pd.read_csv(self.data_path)
        
        # Séparation features/target
        X = self.df.drop(['job_id', 'title', 'location', 'department', 'fraudulent'], axis=1, errors='ignore')
        y = self.df['fraudulent']
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalisation
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✅ Données préparées: {X.shape[1]} features")
        print(f"📊 Répartition des classes: {np.mean(y):.2%} de fraudes")
        
    def initialize_models(self):
        """Initialise les modèles à comparer"""
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(scale_pos_weight=np.sum(self.y_train==0)/np.sum(self.y_train==1), 
                                   random_state=42),
            "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
            "SVM": SVC(class_weight='balanced', probability=True, random_state=42)
        }
        
    def train_and_evaluate(self):
        """Entraîne et évalue tous les modèles"""
        print("\n=== ENTRAÎNEMENT DES MODÈLES ===")
        for name, model in self.models.items():
            start_time = time.time()
            
            # Entraînement
            model.fit(self.X_train, self.y_train)
            
            # Prédictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calcul des métriques
            train_time = time.time() - start_time
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            
            # Stockage des résultats (version corrigée)
            new_row = pd.DataFrame({
                'Model': [name],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1': [f1],
                'ROC AUC': [roc_auc],
                'Training Time': [train_time]
            })
            self.results = pd.concat([self.results, new_row], ignore_index=True)
            
            print(f"\n📊 {name} - Performance:")
            print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f} | F1: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f} | Time: {train_time:.2f}s")
            
            # Matrice de confusion
            self.plot_confusion_matrix(y_pred, name)
            
            # Rapport de classification
            print(f"\n📝 Rapport de classification pour {name}:")
            print(classification_report(self.y_test, y_pred))
    
    def plot_confusion_matrix(self, y_pred, model_name):
        """Affiche la matrice de confusion"""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Légitime', 'Fraude'],
                    yticklabels=['Légitime', 'Fraude'])
        plt.title(f'Matrice de confusion - {model_name}')
        plt.ylabel('Vérité terrain')
        plt.xlabel('Prédictions')
        plt.show()
    
    def compare_results(self):
        """Compare les performances des modèles"""
        print("\n=== COMPARAISON DES MODÈLES ===")
        plt.figure(figsize=(12, 8))
        
        # Préparation des données pour le graphique
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
        melted_results = self.results.melt(id_vars='Model', 
                                         value_vars=metrics,
                                         var_name='Metric', 
                                         value_name='Value')
        
        # Graphique de comparaison
        sns.barplot(x='Model', y='Value', hue='Metric', data=melted_results)
        plt.title('Comparaison des performances des modèles')
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # Affichage des résultats
        print("\n🔍 Tableau comparatif:")
        print(self.results.sort_values('F1', ascending=False))
        
        # Sauvegarde des résultats
        self.results.to_csv('model_comparison_results.csv', index=False)
        print("\n💾 Résultats sauvegardés dans model_comparison_results.csv")
    
    def save_best_model(self):
        """Sauvegarde le meilleur modèle"""
        best_model_name = self.results.sort_values('F1', ascending=False).iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        print(f"\n🏆 Meilleur modèle sauvegardé: {best_model_name} (best_model.pkl)")
    
    def run(self):
        """Exécute le pipeline complet"""
        self.load_and_prepare_data()
        self.initialize_models()
        self.train_and_evaluate()
        self.compare_results()
        self.save_best_model()

if __name__ == "__main__":
    print("""
    ====================================
    COMPARAISON DE MODÈLES - DÉTECTION DE FRAUDE
    ====================================
    """)
    
    # Chemin vers vos données préparées
    DATA_PATH = "../data/clean_jobs.csv"
    
    trainer = FraudDetectionModels(DATA_PATH)
    trainer.run()