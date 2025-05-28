"""
model_training.py - Script amélioré avec gestion des NaN et optimisation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, 
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import joblib
import json
from collections import defaultdict

warnings.filterwarnings('ignore')

class FraudDetectionModels:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.models = {}
        self.results = pd.DataFrame(columns=[
            'Model', 'Accuracy', 'Precision', 'Recall', 
            'F1', 'ROC AUC', 'Avg Precision', 'Training Time'
        ])
        self.feature_importances = defaultdict(dict)
        
    def load_and_prepare_data(self) -> None:
        """Charge et prépare les données avec gestion des NaN"""
        print("\n=== CHARGEMENT ET PRÉPARATION DES DONNÉES ===")
        
        try:
            # Chargement des données
            self.df = pd.read_csv(self.data_path)
            
            # Vérification des colonnes requises
            if 'fraudulent' not in self.df.columns:
                raise ValueError("La colonne cible 'fraudulent' est manquante")
            
            # Séparation features/target
            non_feature_cols = ['job_id', 'title', 'location', 'department', 'fraudulent']
            X = self.df.drop([col for col in non_feature_cols if col in self.df.columns], axis=1)
            y = self.df['fraudulent']
            
            # Conversion des colonnes catégorielles en numériques si nécessaire
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Imputation des valeurs manquantes
            X_imputed = self.imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            
            # Split train/test stratifié
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            # Normalisation
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            print(f"✅ Données préparées: {X.shape[1]} features")
            print(f"📊 Répartition des classes - Train: {np.mean(self.y_train):.2%} fraude")
            print(f"📊 Répartition des classes - Test: {np.mean(self.y_test):.2%} fraude")
            print(f"🔍 Valeurs manquantes traitées avec stratégie: {self.imputer.strategy}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données: {str(e)}")
            raise

    def initialize_models(self) -> None:
        """Initialise les modèles avec gestion du déséquilibre"""
        print("\n=== INITIALISATION DES MODÈLES ===")
        ratio = np.sum(self.y_train==0)/np.sum(self.y_train==1)
        
        self.models = {
            "HistGradientBoosting": HistGradientBoostingClassifier(
                class_weight='balanced',
                random_state=42,
                max_iter=200
            ),
            "Balanced RF": BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                sampling_strategy='auto',
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                scale_pos_weight=ratio,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                eval_metric='aucpr',
                random_state=42,
                n_jobs=-1,
                tree_method='hist'  # Plus rapide et gère mieux les données
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                penalty='l2',
                solver='lbfgs'
            ),
            "LightGBM": LGBMClassifier(
                class_weight='balanced',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                metric='aucpr',
                random_state=42,
                n_jobs=-1
            )
        }
        
        print(f"✅ {len(self.models)} modèles initialisés")

    def train_and_evaluate(self) -> None:
        """Entraîne et évalue les modèles avec gestion des erreurs"""
        print("\n=== ENTRAÎNEMENT ET ÉVALUATION ===")
        
        for name, model in self.models.items():
            try:
                print(f"\n--- Entraînement du modèle {name} ---")
                start_time = time.time()
                
                # Entraînement avec gestion des erreurs
                model.fit(self.X_train, self.y_train)
                
                # Prédictions
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calcul des métriques
                metrics = {
                    'Model': name,
                    'Accuracy': accuracy_score(self.y_test, y_pred),
                    'Precision': precision_score(self.y_test, y_pred, zero_division=0),
                    'Recall': recall_score(self.y_test, y_pred),
                    'F1': f1_score(self.y_test, y_pred),
                    'ROC AUC': roc_auc_score(self.y_test, y_proba),
                    'Avg Precision': average_precision_score(self.y_test, y_proba),
                    'Training Time': time.time() - start_time
                }
                
                # Stockage des résultats
                self.results = pd.concat([self.results, pd.DataFrame([metrics])], ignore_index=True)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, val in enumerate(importances):
                        self.feature_importances[name][f"Feature_{i}"] = val
                
                # Affichage des résultats
                self.print_metrics(metrics)
                self.plot_confusion_matrix(y_pred, name)
                self.plot_pr_curve(y_proba, name, metrics['Avg Precision'])
                
            except Exception as e:
                print(f"❌ Erreur avec le modèle {name}: {str(e)}")
                continue

    def print_metrics(self, metrics: dict) -> None:
        """Affiche les métriques de performance"""
        print(f"\n📊 Performance du modèle {metrics['Model']}:")
        print(f"Accuracy: {metrics['Accuracy']:.4f} | Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")
        print(f"ROC AUC: {metrics['ROC AUC']:.4f} | Avg Precision: {metrics['Avg Precision']:.4f}")
        print(f"Training Time: {metrics['Training Time']:.2f}s")
        print("\n📝 Rapport de classification:")
        print(classification_report(self.y_test, 
                                  self.models[metrics['Model']].predict(self.X_test)))

    def plot_confusion_matrix(self, y_pred: np.array, model_name: str) -> None:
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

    def plot_pr_curve(self, y_proba: np.array, model_name: str, avg_precision: float) -> None:
        """Affiche la courbe Precision-Recall"""
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f'{model_name} (AP={avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Courbe Precision-Recall')
        plt.legend()
        plt.show()

    def compare_results(self) -> None:
        """Compare les performances des modèles"""
        print("\n=== COMPARAISON DES RÉSULTATS ===")
        
        # Tri par F1-score
        sorted_results = self.results.sort_values('F1', ascending=False)
        
        # Affichage
        print("\n🔍 Classement des modèles:")
        print(sorted_results[['Model', 'F1', 'Avg Precision', 'Recall', 'Precision']])
        
        # Visualisation
        plt.figure(figsize=(12, 6))
        metrics = ['F1', 'Avg Precision', 'Recall', 'Precision']
        melted_results = sorted_results.melt(id_vars='Model', 
                                          value_vars=metrics,
                                          var_name='Metric', 
                                          value_name='Value')
        
        sns.barplot(x='Model', y='Value', hue='Metric', data=melted_results)
        plt.title('Comparaison des performances')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def save_best_model(self) -> None:
        """Sauvegarde le meilleur modèle et les artefacts"""
        if self.results.empty:
            raise ValueError("Aucun résultat disponible pour sélectionner le meilleur modèle")
            
        best_model_name = self.results.sort_values('F1', ascending=False).iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        # Sauvegarde
        artifacts = {
            'model': best_model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'features': list(self.df.drop(['job_id', 'title', 'location', 'department', 'fraudulent'], 
                                      axis=1, errors='ignore').columns)
        }
        
        joblib.dump(artifacts, 'best_model_artifacts.pkl')
        
        print(f"\n🏆 Meilleur modèle sauvegardé: {best_model_name}")
        print(f"📦 Artefacts sauvegardés: modèle, scaler, imputer et noms de features")

    def run(self) -> None:
        """Exécute le pipeline complet"""
        try:
            self.load_and_prepare_data()
            self.initialize_models()
            self.train_and_evaluate()
            self.compare_results()
            self.save_best_model()
        except Exception as e:
            print(f"❌ Erreur critique dans le pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    print("""
    ====================================
    DÉTECTION DE FRAUDE - ENTRAÎNEMENT (VERSION ROBUSTE)
    ====================================
    """)
    
    try:
        DATA_PATH = "../data/clean_jobs.csv"  # Ajustez ce chemin
        trainer = FraudDetectionModels(DATA_PATH)
        trainer.run()
    except Exception as e:
        print(f"❌ Le programme a rencontré une erreur: {str(e)}")