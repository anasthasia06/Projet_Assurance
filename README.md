# 🚗 API Tarification Assurance Auto — MLOps CY Tech

Pipeline MLOps complet : du notebook à une API déployable avec CI/CD.

## 📂 Structure du projet

```
Projet_Assurance/
├── src/
│   ├── processing.py     ← nettoyage + feature engineering
│   ├── training.py       ← entraînement + sauvegarde des modèles
│   └── api.py            ← API FastAPI (4 routes)
├── models/               ← fichiers .pkl (générés par training.py)
├── data/
│   └── train.csv         ← données brutes (non versionnées)
├── tests/
│   └── test_project.py   ← tests unitaires (pytest)
├── .github/
│   └── workflows/
│       └── ci.yml        ← pipeline CI/CD GitHub Actions
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Lancement en local

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Mettre train.csv dans data/
```bash
mkdir data
# copier votre train.csv dans data/
```

### 3. Entraîner les modèles
```bash
python src/training.py
```
→ Génère `models/count_encoder.pkl`, `models/xgb_frequence.pkl`, `models/xgb_gravite.pkl`

### 4. Lancer l'API
```bash
uvicorn src.api:app --reload
```
→ Ouvrir **http://localhost:8000/docs** pour le Swagger

### 5. Lancer les tests
```bash
pytest tests/ -v
```

## 🌐 Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | État de l'API et des modèles |
| POST | `/predict_frequency` | Probabilité de sinistre |
| POST | `/predict_amount` | Coût moyen prédit |
| POST | `/predict` | Prime pure = fréquence × montant |

## 🐳 Docker

```bash
# Construire l'image
docker build -t api-assurance .

# Lancer le conteneur
docker run -p 8000:8000 api-assurance
```

## 🔀 Branches Git

```
main       ← stable, production
 └── dev   ← intégration
      ├── feature/api
      ├── feature/tests
      └── feature/ci
```

## ⚙️ CI/CD

La CI se déclenche automatiquement à chaque push :
1. Installation des dépendances
2. Flake8 (qualité du code)
3. Black (formatage)
4. Pytest (tests unitaires)
