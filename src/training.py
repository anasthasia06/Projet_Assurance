"""
training.py — Entraînement et sauvegarde des modèles
Entraîne le modèle de fréquence (classification XGBoost) et le modèle de gravité (régression XGBoost),
puis sauvegarde les 3 fichiers .pkl dans le dossier models/.

Usage :
    python src/training.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from category_encoders import CountEncoder

from processing import load_data, clean_data, build_features, CAT_COLS

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def save_pickle(obj, filename: str):
    """Sauvegarde un objet Python en .pkl dans le dossier models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✅ Sauvegardé : {path}")


def train_and_save(train_path: str = DATA_PATH):
    """
    Pipeline complet :
    1. Charge les données
    2. Nettoie + feature engineering
    3. Encode avec CountEncoder
    4. Entraîne modèle fréquence (classification binaire)
    5. Entraîne modèle gravité (régression gamma, sur sinistrés uniquement)
    6. Sauvegarde les 3 .pkl
    """

    print("=" * 55)
    print("  PIPELINE ENTRAÎNEMENT — ASSURANCE AUTO")
    print("=" * 55)

    # ── 1. Chargement ────────────────────────────────────────
    print("\n📂 1. Chargement des données...")
    df_raw = load_data(train_path)
    print(f"   {df_raw.shape[0]} lignes chargées")

    # ── 2. Extraction des cibles AVANT nettoyage ─────────────
    y_freq = (df_raw["nombre_sinistres"].fillna(0) > 0).astype(int)  # binaire 0/1
    y_grav = (
        df_raw["montant_sinistre"].fillna(0) / df_raw["nombre_sinistres"].replace(0, np.nan)
    ).fillna(0)  # coût moyen par sinistre
    mask_sinistres = y_freq == 1  # sinistrés uniquement pour la gravité

    print(f"   Taux de sinistralité : {y_freq.mean():.2%}")
    print(f"   Coût moyen sinistré  : {y_grav[mask_sinistres].mean():.2f} €")

    # ── 3. Nettoyage + Feature Engineering ───────────────────
    print("\n🔧 2. Nettoyage et Feature Engineering...")
    df_clean = clean_data(df_raw)
    X = build_features(df_clean)
    print(f"   {X.shape[1]} features prêtes")

    # ── 4. Encodage CountEncoder ──────────────────────────────
    print("\n🔢 3. Encodage des variables catégorielles (CountEncoder)...")
    cat_cols_present = [c for c in CAT_COLS if c in X.columns]
    encoder = CountEncoder(cols=cat_cols_present, handle_unknown=0)
    X_encoded = encoder.fit_transform(X)
    X_encoded = X_encoded.astype(float)
    save_pickle(encoder, "count_encoder.pkl")

    # ── 5. Modèle Fréquence ───────────────────────────────────
    print("\n🎯 4. Entraînement modèle Fréquence (XGBoost Classifier)...")
    model_freq = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        verbosity=0,
        random_state=42,
    )
    model_freq.fit(X_encoded, y_freq)
    save_pickle(model_freq, "xgb_frequence.pkl")

    # ── 6. Modèle Gravité ─────────────────────────────────────
    print("\n💶 5. Entraînement modèle Gravité (XGBoost Regressor, sinistrés uniquement)...")
    X_grav = X_encoded[mask_sinistres]
    y_grav_filtered = y_grav[mask_sinistres]
    print(f"   {len(X_grav)} dossiers sinistrés utilisés")

    model_grav = XGBRegressor(
        objective="reg:gamma",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )
    model_grav.fit(X_grav, y_grav_filtered)
    save_pickle(model_grav, "xgb_gravite.pkl")

    # ── Résumé ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ✅ ENTRAÎNEMENT TERMINÉ")
    print("  Fichiers générés dans models/ :")
    print("    - count_encoder.pkl")
    print("    - xgb_frequence.pkl")
    print("    - xgb_gravite.pkl")
    print("=" * 55)


if __name__ == "__main__":
    train_and_save()
