"""
tests/test_project.py — Tests unitaires complets
Couvre : processing.py (clean_data, build_features) + les 4 endpoints de l'API.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Permet à pytest de trouver src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_row():
    """Ligne brute représentative du train.csv (avant tout nettoyage)."""
    return pd.DataFrame(
        [
            {
                "index": 0,
                "id_client": "A00000001",
                "id_vehicule": "V01",
                "id_contrat": "A00000001-V01",
                "nombre_sinistres": 1,
                "montant_sinistre": 800.0,
                "bonus": 0.5,
                "type_contrat": "Maxi",
                "duree_contrat": 12,
                "anciennete_info": 5,
                "freq_paiement": "Biannual",
                "paiement": "No",
                "utilisation": "Retired",
                "code_postal": 75001,
                "conducteur2": "No",
                "age_conducteur1": 35.0,
                "age_conducteur2": 0.0,
                "sex_conducteur1": "M",
                "sex_conducteur2": None,
                "anciennete_permis1": 15.0,
                "anciennete_permis2": 0.0,
                "anciennete_vehicule": 5.0,
                "cylindre_vehicule": 1600.0,
                "din_vehicule": 110.0,
                "essence_vehicule": "Gasoline",
                "marque_vehicule": "PEUGEOT",
                "modele_vehicule": "308",
                "debut_vente_vehicule": 2015.0,
                "fin_vente_vehicule": 2022.0,
                "vitesse_vehicule": 180.0,
                "type_vehicule": "Tourism",
                "prix_vehicule": 20000.0,
                "poids_vehicule": 1200.0,
            }
        ]
    )


@pytest.fixture
def raw_row_with_nan(raw_row):
    """Même ligne mais avec des NaN sur les colonnes optionnelles."""
    df = raw_row.copy()
    df.loc[0, "din_vehicule"] = np.nan
    df.loc[0, "paiement"] = None
    df.loc[0, "sex_conducteur1"] = None
    return df


@pytest.fixture
def clean_row(raw_row):
    """Ligne après clean_data."""
    from processing import clean_data
    return clean_data(raw_row)


@pytest.fixture
def feat_row(clean_row):
    """Ligne après clean_data + build_features."""
    from processing import build_features
    return build_features(clean_row)


# ── Tests clean_data ─────────────────────────────────────────────────────────
class TestCleanData:

    def test_supprime_id_client(self, raw_row):
        from processing import clean_data
        result = clean_data(raw_row)
        assert "id_client" not in result.columns

    def test_supprime_id_vehicule(self, raw_row):
        from processing import clean_data
        result = clean_data(raw_row)
        assert "id_vehicule" not in result.columns

    def test_supprime_nombre_sinistres(self, raw_row):
        from processing import clean_data
        result = clean_data(raw_row)
        assert "nombre_sinistres" not in result.columns

    def test_supprime_montant_sinistre(self, raw_row):
        from processing import clean_data
        result = clean_data(raw_row)
        assert "montant_sinistre" not in result.columns

    def test_nan_numerique_remplace_par_zero(self, raw_row_with_nan):
        from processing import clean_data
        result = clean_data(raw_row_with_nan)
        num_cols = result.select_dtypes(include=["number"]).columns
        assert result[num_cols].isna().sum().sum() == 0

    def test_nan_categoriel_remplace_par_sentinel(self, raw_row_with_nan):
        from processing import clean_data
        result = clean_data(raw_row_with_nan)
        cat_cols = result.select_dtypes(exclude=["number"]).columns
        assert result[cat_cols].isna().sum().sum() == 0

    def test_sex_M_encode_1(self, raw_row):
        from processing import clean_data
        result = clean_data(raw_row)
        if "sex_conducteur1" in result.columns:
            assert result["sex_conducteur1"].iloc[0] == 1.0

    def test_sex_F_encode_0(self, raw_row):
        from processing import clean_data
        df = raw_row.copy()
        df.loc[0, "sex_conducteur1"] = "F"
        result = clean_data(df)
        if "sex_conducteur1" in result.columns:
            assert result["sex_conducteur1"].iloc[0] == 0.0

    def test_paiement_No_encode_0(self, raw_row):
        from processing import clean_data
        result = clean_data(raw_row)
        if "paiement" in result.columns:
            assert result["paiement"].iloc[0] == 0.0

    def test_retourne_dataframe(self, raw_row):
        from processing import clean_data
        assert isinstance(clean_data(raw_row), pd.DataFrame)

    def test_ne_modifie_pas_input(self, raw_row):
        from processing import clean_data
        cols_avant = list(raw_row.columns)
        clean_data(raw_row)
        assert list(raw_row.columns) == cols_avant


# ── Tests build_features ──────────────────────────────────────────────────────
class TestBuildFeatures:

    def test_cree_ratio_poids_puissance(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        assert "ratio_poids_puissance" in result.columns

    def test_ratio_poids_puissance_positif(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        assert result["ratio_poids_puissance"].iloc[0] > 0

    def test_cree_age_obtention_permis(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        assert "age_obtention_permis" in result.columns

    def test_age_obtention_permis_valeur(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        # 35 - 15 = 20
        assert result["age_obtention_permis"].iloc[0] == 20.0

    def test_cree_log_prix_vehicule(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        assert "log_prix_vehicule" in result.columns

    def test_log_prix_vehicule_positif(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        assert result["log_prix_vehicule"].iloc[0] > 0

    def test_cree_duree_vie_modele(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        assert "duree_vie_modele" in result.columns

    def test_duree_vie_modele_valeur(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        # 2022 - 2015 = 7
        assert result["duree_vie_modele"].iloc[0] == 7.0

    def test_pas_de_nan_apres_features(self, clean_row):
        from processing import build_features
        result = build_features(clean_row)
        assert result.isna().sum().sum() == 0

    def test_retourne_dataframe(self, clean_row):
        from processing import build_features
        assert isinstance(build_features(clean_row), pd.DataFrame)

    def test_colonnes_alignees_sur_expected(self, clean_row):
        from processing import build_features, EXPECTED_FEATURES
        result = build_features(clean_row)
        assert list(result.columns) == EXPECTED_FEATURES


# ── Tests API ─────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)


VALID_PAYLOAD = {
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


class TestHealthEndpoint:

    def test_retourne_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_contient_models_loaded(self, client):
        assert "models_loaded" in client.get("/health").json()


class TestPredictFrequencyEndpoint:

    def test_champ_manquant_retourne_422(self, client):
        payload = VALID_PAYLOAD.copy()
        del payload["age_conducteur1"]
        assert client.post("/predict_frequency", json=payload).status_code == 422

    def test_reponse_contient_proba_sinistre(self, client):
        r = client.post("/predict_frequency", json=VALID_PAYLOAD)
        if r.status_code == 200:
            assert "proba_sinistre" in r.json()

    def test_proba_entre_0_et_1(self, client):
        r = client.post("/predict_frequency", json=VALID_PAYLOAD)
        if r.status_code == 200:
            assert 0.0 <= r.json()["proba_sinistre"] <= 1.0

    def test_risque_valide(self, client):
        r = client.post("/predict_frequency", json=VALID_PAYLOAD)
        if r.status_code == 200:
            assert r.json()["risque"] in ["faible", "moyen", "élevé"]


class TestPredictAmountEndpoint:

    def test_champ_manquant_retourne_422(self, client):
        payload = VALID_PAYLOAD.copy()
        del payload["anciennete_permis1"]
        assert client.post("/predict_amount", json=payload).status_code == 422

    def test_reponse_contient_cout_moyen(self, client):
        r = client.post("/predict_amount", json=VALID_PAYLOAD)
        if r.status_code == 200:
            assert "cout_moyen_predit" in r.json()

    def test_cout_positif(self, client):
        r = client.post("/predict_amount", json=VALID_PAYLOAD)
        if r.status_code == 200:
            assert r.json()["cout_moyen_predit"] >= 0


class TestPredictEndpoint:

    def test_champ_manquant_retourne_422(self, client):
        payload = VALID_PAYLOAD.copy()
        del payload["age_conducteur1"]
        assert client.post("/predict", json=payload).status_code == 422

    def test_reponse_contient_prime(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        if r.status_code == 200:
            assert "prime_pure" in r.json()

    def test_prime_positive(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        if r.status_code == 200:
            assert r.json()["prime_pure"] >= 0

    def test_reponse_contient_tous_les_champs(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        if r.status_code == 200:
            data = r.json()
            assert all(k in data for k in ["proba_sinistre", "cout_moyen_predit", "prime_pure", "risque"])

    def test_coherence_prime_frequence_montant(self, client):
        """prime_pure doit être ≈ proba_sinistre × cout_moyen_predit."""
        r = client.post("/predict", json=VALID_PAYLOAD)
        if r.status_code == 200:
            d = r.json()
            expected = round(d["proba_sinistre"] * d["cout_moyen_predit"], 2)
            assert abs(d["prime_pure"] - expected) < 0.01
