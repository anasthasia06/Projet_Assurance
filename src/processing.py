"""
processing.py — Nettoyage et Feature Engineering
Transforme une observation brute (dict ou DataFrame 1 ligne) en features prêtes pour la prédiction.
"""

import pandas as pd
import numpy as np


# Colonnes à supprimer (identifiants + cibles)
COLS_TO_DROP = ["index", "id_client", "id_vehicule", "id_contrat", "nombre_sinistres", "montant_sinistre"]

# Colonnes binaires Yes/No
YES_NO_COLS = ["conducteur2", "paiement"]

# Colonnes sexe
SEX_COLS = ["sex_conducteur1"]

# Seuil de suppression pour les valeurs manquantes (40%)
MISSING_THRESHOLD = 0.4

# Colonnes retenues après tout le pipeline (calculées sur le train)
# C'est la liste exacte que le modèle attend en entrée
EXPECTED_FEATURES = [
    "type_contrat", "duree_contrat", "anciennete_info", "freq_paiement",
    "utilisation", "code_postal", "age_conducteur1", "sex_conducteur1",
    "anciennete_permis1", "anciennete_vehicule", "cylindre_vehicule",
    "din_vehicule", "essence_vehicule", "marque_vehicule", "modele_vehicule",
    "fin_vente_vehicule", "vitesse_vehicule", "type_vehicule", "prix_vehicule",
    "poids_vehicule", "ratio_poids_puissance", "age_obtention_permis",
    "duree_vie_modele", "log_prix_vehicule",
]

# Colonnes catégorielles (celles qui resteront pour le CountEncoder)
CAT_COLS = [
    "type_contrat", "freq_paiement", "utilisation", "essence_vehicule",
    "marque_vehicule", "modele_vehicule", "type_vehicule",
]


def load_data(file_path: str) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame :
    - Supprime les colonnes identifiants et cibles
    - Remplit les NaN numériques par 0 et catégoriels par '-999'
    - Encode les colonnes binaires (Yes/No → 1/0, M/F → 1/0)

    Fonctionne pour 1 ligne comme pour un DataFrame entier.
    """
    df_clean = df.copy()

    # 1. Suppression des colonnes inutiles
    df_clean = df_clean.drop(
        columns=[c for c in COLS_TO_DROP if c in df_clean.columns]
    )

    # 2. Gestion des valeurs manquantes
    num_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_clean.select_dtypes(exclude=["number"]).columns.tolist()

    df_clean[num_cols] = df_clean[num_cols].fillna(0)
    df_clean[cat_cols] = df_clean[cat_cols].fillna("-999")

    # 3. Encodage binaire Yes/No → 1/0
    for col in YES_NO_COLS:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col].replace({"Yes": 1, "No": 0, "-999": 0}).astype(float)
            )

    # 4. Encodage sexe M/F → 1/0
    for col in SEX_COLS:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col].replace({"M": 1, "F": 0, "-999": 0}).astype(float)
            )

    return df_clean


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le feature engineering :
    - Crée les nouvelles variables (ratios, indicateurs, log)
    - Supprime les colonnes non retenues dans le modèle final
    - Aligne les colonnes sur EXPECTED_FEATURES

    Fonctionne pour 1 ligne comme pour un DataFrame entier.
    """
    df_feat = df.copy()

    # A. Ratio Poids / Puissance
    if "poids_vehicule" in df_feat.columns and "din_vehicule" in df_feat.columns:
        df_feat["ratio_poids_puissance"] = df_feat["poids_vehicule"] / (
            df_feat["din_vehicule"] + 1e-5
        )

    # B. Âge d'obtention du permis
    if "age_conducteur1" in df_feat.columns and "anciennete_permis1" in df_feat.columns:
        df_feat["age_obtention_permis"] = (
            df_feat["age_conducteur1"] - df_feat["anciennete_permis1"]
        )

    # C. Durée de vie commerciale du modèle véhicule
    if "fin_vente_vehicule" in df_feat.columns and "debut_vente_vehicule" in df_feat.columns:
        df_feat["duree_vie_modele"] = (
            df_feat["fin_vente_vehicule"] - df_feat["debut_vente_vehicule"]
        )

    # D. Log du prix véhicule
    if "prix_vehicule" in df_feat.columns:
        df_feat["log_prix_vehicule"] = np.log1p(df_feat["prix_vehicule"])

    # E. Nettoyage code_postal (peut contenir '2A032' pour la Corse → on garde juste les 2 premiers chiffres)
    if "code_postal" in df_feat.columns:
        df_feat["code_postal"] = (
            df_feat["code_postal"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna(0)
            .astype(float)
        )

    # F. Alignement final sur les colonnes attendues par le modèle
    # On ajoute les colonnes manquantes à 0 et on supprime les colonnes en trop
    for col in EXPECTED_FEATURES:
        if col not in df_feat.columns:
            df_feat[col] = 0

    df_feat = df_feat[EXPECTED_FEATURES]

    return df_feat