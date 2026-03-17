"""
api.py — API FastAPI Assurance Auto
4 routes exactes demandées dans le TD :
  GET  /health              → état de l'API
  POST /predict_frequency   → probabilité de sinistre
  POST /predict_amount      → coût moyen prédit (si sinistre)
  POST /predict             → prime pure = fréquence × montant
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Permet d'importer processing.py depuis src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processing import clean_data, build_features, CAT_COLS  # noqa: E402

# ── Chargement des modèles au démarrage ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _load(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


encoder = _load("count_encoder.pkl")
model_freq = _load("xgb_frequence.pkl")
model_grav = _load("xgb_gravite.pkl")

_models_ok = all(x is not None for x in [encoder, model_freq, model_grav])


# ── Schéma d'entrée ───────────────────────────────────────────────────────────
class AssureInput(BaseModel):
    """Données d'un assuré. Tous les champs correspondent aux colonnes du train.csv d'origine."""

    # Contrat
    type_contrat: Optional[str] = Field("Maxi", example="Maxi")
    duree_contrat: Optional[float] = Field(12, example=12)
    anciennete_info: Optional[float] = Field(5, example=5)
    freq_paiement: Optional[str] = Field("Biannual", example="Biannual")
    paiement: Optional[str] = Field("No", example="No")
    utilisation: Optional[str] = Field("Retired", example="Retired")
    code_postal: Optional[float] = Field(75001, example=75001)

    # Conducteur principal
    age_conducteur1: float = Field(..., example=35, description="Âge du conducteur (obligatoire)")
    sex_conducteur1: Optional[str] = Field("M", example="M", description="M ou F")
    anciennete_permis1: float = Field(..., example=15, description="Ancienneté du permis en années (obligatoire)")

    # Conducteur 2
    conducteur2: Optional[str] = Field("No", example="No")
    age_conducteur2: Optional[float] = Field(0, example=0)
    sex_conducteur2: Optional[str] = Field(None, example=None)
    anciennete_permis2: Optional[float] = Field(0, example=0)

    # Véhicule
    anciennete_vehicule: Optional[float] = Field(5, example=5)
    cylindre_vehicule: Optional[float] = Field(1600, example=1600)
    din_vehicule: Optional[float] = Field(110, example=110)
    essence_vehicule: Optional[str] = Field("Gasoline", example="Gasoline")
    marque_vehicule: Optional[str] = Field("PEUGEOT", example="PEUGEOT")
    modele_vehicule: Optional[str] = Field("308", example="308")
    debut_vente_vehicule: Optional[float] = Field(2015, example=2015)
    fin_vente_vehicule: Optional[float] = Field(2022, example=2022)
    vitesse_vehicule: Optional[float] = Field(180, example=180)
    type_vehicule: Optional[str] = Field("Tourism", example="Tourism")
    prix_vehicule: Optional[float] = Field(20000, example=20000)
    poids_vehicule: Optional[float] = Field(1200, example=1200)

    class Config:
        json_schema_extra = {
            "example": {
                "age_conducteur1": 35,
                "anciennete_permis1": 15,
                "sex_conducteur1": "M",
                "din_vehicule": 110,
                "poids_vehicule": 1200,
                "prix_vehicule": 20000,
                "anciennete_vehicule": 5,
                "type_contrat": "Maxi",
                "fin_vente_vehicule": 2022,
                "debut_vente_vehicule": 2015,
            }
        }


# ── Fonction de pipeline ──────────────────────────────────────────────────────
def _preprocess(data: AssureInput) -> pd.DataFrame:
    """Transforme l'input Pydantic en DataFrame encodé prêt pour le modèle."""
    df = pd.DataFrame([data.model_dump()])
    df_clean = clean_data(df)
    df_feat = build_features(df_clean)
    cat_cols_present = [c for c in CAT_COLS if c in df_feat.columns]
    df_encoded = encoder.transform(df_feat[df_feat.columns])
    return df_encoded.astype(float)


def _check_models():
    if not _models_ok:
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés. Lancez d'abord : python src/training.py",
        )


# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="API Tarification Assurance Auto",
    description=(
        "API de prédiction de prime d'assurance automobile. "
        "Basée sur deux modèles XGBoost : fréquence de sinistres (classification) "
        "et gravité (régression). La prime pure = P(sinistre) × Coût moyen."
    ),
    version="1.0.0",
)


# ── Route 1 : Health ──────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["Monitoring"],
    summary="État de santé de l'API",
)
def health():
    """Vérifie que l'API est opérationnelle et que les 3 modèles sont chargés."""
    return {
        "status": "ok",
        "models_loaded": _models_ok,
        "encoder": encoder is not None,
        "model_frequence": model_freq is not None,
        "model_gravite": model_grav is not None,
    }


# ── Route 2 : Prédiction Fréquence ───────────────────────────────────────────
@app.post(
    "/predict_frequency",
    tags=["Prédiction"],
    summary="Probabilité de sinistre (modèle fréquence)",
)
def predict_frequency(data: AssureInput):
    """
    Prédit la **probabilité qu'un assuré ait au moins un sinistre**.

    - **proba_sinistre** : probabilité entre 0 et 1
    - **risque** : faible / moyen / élevé
    """
    _check_models()
    X = _preprocess(data)
    proba = float(model_freq.predict_proba(X)[0][1])
    proba = round(max(0.0, min(1.0, proba)), 4)
    risque = "faible" if proba < 0.1 else ("moyen" if proba < 0.3 else "élevé")
    return {"proba_sinistre": proba, "risque": risque}


# ── Route 3 : Prédiction Montant ─────────────────────────────────────────────
@app.post(
    "/predict_amount",
    tags=["Prédiction"],
    summary="Coût moyen prédit par sinistre (modèle gravité)",
)
def predict_amount(data: AssureInput):
    """
    Prédit le **coût moyen d'un sinistre** si l'assuré en a un.

    - **cout_moyen_predit** : montant en euros
    """
    _check_models()
    X = _preprocess(data)
    montant = float(model_grav.predict(X)[0])
    montant = round(max(0.0, montant), 2)
    return {"cout_moyen_predit": montant, "unite": "euros"}


# ── Route 4 : Prédiction Prime (agrégée) ──────────────────────────────────────
@app.post(
    "/predict",
    tags=["Prédiction"],
    summary="Prime pure = fréquence × montant (modèle agrégé)",
)
def predict(data: AssureInput):
    """
    Calcule la **prime pure annuelle** de l'assuré.

    Formule : `prime_pure = P(sinistre) × Coût_moyen_sinistre`

    Retourne :
    - **proba_sinistre** : sortie du modèle fréquence
    - **cout_moyen_predit** : sortie du modèle gravité
    - **prime_pure** : prime calculée en euros
    - **risque** : faible / moyen / élevé
    """
    _check_models()
    X = _preprocess(data)

    proba = float(model_freq.predict_proba(X)[0][1])
    proba = round(max(0.0, min(1.0, proba)), 4)

    montant = float(model_grav.predict(X)[0])
    montant = round(max(0.0, montant), 2)

    prime = round(proba * montant, 2)
    risque = "faible" if proba < 0.1 else ("moyen" if proba < 0.3 else "élevé")

    return {
        "proba_sinistre": proba,
        "cout_moyen_predit": montant,
        "prime_pure": prime,
        "risque": risque,
        "unite": "euros",
    }
