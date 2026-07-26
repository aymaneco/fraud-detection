"""Service (inférence) : charge les artefacts figés et prédit, SANS jamais toucher au label.

    art = load_artifacts("v2")
    out = predict(raw_paniers_df, art)   # -> DataFrame [ID, fraud_proba]

Les features produit sont reconstruites par simple *lookup* dans les tables figées ;
les paniers synthétiques (is_synthetic==1) et toute valeur inconnue retombent sur la
base (règle métier / smoothing). Transform-only, déterministe.
"""
import os, json, joblib
import numpy as np, pandas as pd

from features_static import build_static, STATIC_COLS
from product_feats import load_baskets, apply_tables, PRODUCT_COLS

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_artifacts(version="v2", root=None):
    art = os.path.join(root or _ROOT, "PROD", "artifacts", version)
    with open(os.path.join(art, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(art, "product_tables.json"), encoding="utf-8") as f:
        pt = json.load(f)
    return {
        "config":  config,
        "encoder": joblib.load(os.path.join(art, "catboost_encoder.joblib")),
        "model":   joblib.load(os.path.join(art, "lgbm.joblib")),
        "tables":  pt["tables"],
        "base":    pt["base_rate"],
    }


def predict(raw_df, art):
    "raw_df : paniers bruts (format large item1..24…). Renvoie [ID, fraud_proba]."
    cfg = art["config"]; FEATURES = cfg["features"]

    # 1) features statiques (déterministe), vrais paniers seulement
    S = build_static(raw_df).reset_index(drop=True)
    ids_real = S["ID"].tolist()

    # 2) features produit par LOOKUP dans les tables figées (aucun label)
    B  = load_baskets(raw_df)
    pf = apply_tables(art["tables"], art["base"], ids_real, B).reset_index(drop=True)

    # 3) assemblage + encodage figé + prédiction
    X  = pd.concat([S[STATIC_COLS], pf], axis=1)[FEATURES]
    Xe = art["encoder"].transform(X)
    proba = art["model"].predict_proba(Xe)[:, 1]

    # 4) sortie sur TOUS les paniers d'entrée : réels -> proba, synthétiques/inconnus -> base 0
    out = pd.DataFrame({"ID": raw_df["ID"].to_numpy()})
    out["fraud_proba"] = out["ID"].map(dict(zip(ids_real, proba))).fillna(0.0)
    return out
