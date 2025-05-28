"""
test_api.py - Script de test pour l'API de détection de fraude
"""

import requests
import json
from datetime import datetime

class APITester:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        
    def test_health_check(self):
        """Test du health check"""
        print("\n=== TEST HEALTH CHECK ===")
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return False
    
    def test_home_endpoint(self):
        """Test de l'endpoint home"""
        print("\n=== TEST HOME ENDPOINT ===")
        try:
            response = requests.get(f"{self.base_url}/")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return False
    
    def test_single_prediction(self):
        """Test d'une prédiction simple"""
        print("\n=== TEST PRÉDICTION SIMPLE ===")
        
        # Exemple d'offre légitime
        legitimate_job = {
            "title": "Software Developer",
            "description": "We are looking for an experienced software developer to join our team. You will work on exciting projects using modern technologies like Python, React, and AWS. Requirements include 3+ years of experience and strong problem-solving skills.",
            "company_profile": "Tech Solutions Inc. is a leading software company founded in 2010.",
            "salary_range": "60000-80000",
            "job_url": "https://techsolutions.com/careers/dev-001",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=legitimate_job,
                headers={'Content-Type': 'application/json'}
            )
            print(f"Status Code: {response.status_code}")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                fraud_prob = result['result']['fraud_probability']
                print(f"\n📊 Probabilité de fraude: {fraud_prob:.2%}")
                print(f"🎯 Niveau de risque: {result['result']['risk_level']}")
                
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return False
    
    def test_suspicious_job(self):
        """Test avec une offre suspecte"""
        print("\n=== TEST OFFRE SUSPECTE ===")
        
        # Exemple d'offre suspecte
        suspicious_job = {
            "title": "URGENT!!! Make Money Fast!!!",
            "description": "Earn $5000 per week working from home! No experience needed! Send money for training materials. Contact us immediately! This is urgent! Pay only $99 to start earning thousands!",
            "company_profile": "",  # Profil manquant
            "salary_range": "5000-50000",  # Salaire suspect
            "job_url": "http://suspicious.com/jobs/123",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=suspicious_job,
                headers={'Content-Type': 'application/json'}
            )
            print(f"Status Code: {response.status_code}")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                fraud_prob = result['result']['fraud_probability']
                print(f"\n📊 Probabilité de fraude: {fraud_prob:.2%}")
                print(f"🎯 Niveau de risque: {result['result']['risk_level']}")
                
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return False
    
    def test_batch_prediction(self):
        """Test de prédiction en lot"""
        print("\n=== TEST PRÉDICTION EN LOT ===")
        
        jobs_batch = {
            "jobs": [
                {
                    "title": "Data Scientist",
                    "description": "Looking for a data scientist with experience in machine learning and Python.",
                    "salary_range": "70000-90000"
                },
                {
                    "title": "MAKE MONEY NOW!!!",
                    "description": "Urgent! Pay $50 to start earning $1000 daily! No skills needed!",
                    "salary_range": "1000-10000"
                },
                {
                    "title": "Marketing Manager",
                    "description": "We need a marketing manager to lead our digital marketing campaigns.",
                    "salary_range": "55000-75000"
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/batch_predict",
                json=jobs_batch,
                headers={'Content-Type': 'application/json'}
            )
            print(f"Status Code: {response.status_code}")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print(f"\n📊 Résultats pour {result['total_processed']} offres:")
                for res in result['results']:
                    if res['success']:
                        fraud_prob = res['result']['fraud_probability']
                        risk_level = res['result']['risk_level']
                        print(f"  Job {res['index']}: {fraud_prob:.2%} de fraude ({risk_level})")
                    else:
                        print(f"  Job {res['index']}: Erreur - {res['error']}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return False
    
    def test_invalid_request(self):
        """Test avec des données invalides"""
        print("\n=== TEST REQUÊTE INVALIDE ===")
        
        # Test sans données
        try:
            response = requests.post(f"{self.base_url}/predict")
            print(f"Test sans données - Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Erreur attendue: {str(e)}")
        
        # Test avec données incomplètes
        try:
            incomplete_job = {"title": "Test"}  # Description manquante
            response = requests.post(
                f"{self.base_url}/predict",
                json=incomplete_job,
                headers={'Content-Type': 'application/json'}
            )
            print(f"Test données incomplètes - Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 400
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return False

    def run_all_tests(self):
        """Exécute tous les tests"""
        print("""
        ====================================
        TESTS DE L'API DÉTECTION DE FRAUDE
        ====================================
        """)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Home Endpoint", self.test_home_endpoint),
            ("Single Prediction", self.test_single_prediction),
            ("Suspicious Job", self.test_suspicious_job),
            ("Batch Prediction", self.test_batch_prediction),
            ("Invalid Request", self.test_invalid_request)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, "✅ PASS" if success else "❌ FAIL"))
            except Exception as e:
                results.append((test_name, f"❌ ERROR: {str(e)}"))
        
        # Résumé
        print("\n" + "="*50)
        print("RÉSUMÉ DES TESTS")
        print("="*50)
        for test_name, result in results:
            print(f"{test_name}: {result}")
        
        passed = sum(1 for _, result in results if "PASS" in result)
        total = len(results)
        print(f"\nRésultat: {passed}/{total} tests réussis")

if __name__ == "__main__":
    # Vérifiez que l'API est démarrée avant de lancer les tests
    print("🚀 Démarrage des tests de l'API...")
    print("⚠️  Assurez-vous que l'API est lancée sur http://localhost:5000")
    
    tester = APITester()
    tester.run_all_tests()