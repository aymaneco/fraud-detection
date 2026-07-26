"""Entraîne le modèle final sur TOUT le train et FIGE les artefacts de service.

Produit dans PROD/artifacts/<VERSION>/ :
  - catboost_encoder.joblib : encodeur dom_cat/dom_make (figé)
  - product_tables.json     : tables {SKU/model/cat/make -> taux} + base_rate (target encodings figés)
  - lgbm.joblib             : modèle LightGBM final
  - config.json             : features, hyperparamètres, version, seed, empreinte des données

Usage :  python PROD/train.py
"""
import os, sys, json, hashlib

# --- se placer à la racine et exposer le pipeline ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "PROD", "pipeline"))

import numpy as np, pandas as pd, joblib
import lightgbm as lgb
from category_encoders import CatBoostEncoder
from features_static import build_static, STATIC_COLS
from product_feats import load_baskets, fit_tables, apply_tables, PRODUCT_COLS

VERSION   = "v2"
SEED      = 42
ALPHA     = 20.0
CAT       = ["dom_cat", "dom_make"]
FEATURES  = STATIC_COLS + PRODUCT_COLS
TRAIN_CSV = "data/X_train_clean.csv"
LABEL_CSV = "data/Y_train.csv"

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_reference_profile(X, cat_cols, n_bins=10, top_cat=300):
    """Profil figé de la DONNÉE EN ENTRÉE, pour le monitoring.

    Trois familles, selon la cardinalité (même taxonomie que les sentinelles du
    notebook de sélection). Chacune couvre TOUTES les features, sans trou :
      - continue  (> n_bins valeurs) -> bornes des déciles          -> PSI par tranches
      - discrète  (<= n_bins)        -> fréquence de chaque valeur  -> PSI sur fréquences
      - catégorielle (texte)         -> fréquence des modalités     -> PSI sur fréquences

    Périmètre : la donnée d'ENTRÉE uniquement. On ne profile pas la sortie du modèle
    (hors sujet ici, et une référence de score calculée sur les données d'entraînement
    serait biaisée : in-sample 0,0528 vs hors-échantillon 0,0410 -> fausses alertes).
    """
    import numpy as np
    prof = {"numeric": {}, "discret": {}, "categorical": {}, "n_ref": int(len(X))}
    q = np.linspace(0, 1, n_bins + 1)
    for c in X.columns:
        s = X[c]
        if c in cat_cols:
            vc = s.astype(str).value_counts(normalize=True).head(top_cat)
            prof["categorical"][c] = {"freqs": {str(k): round(float(v), 6) for k, v in vc.items()}}
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique() > n_bins:                       # continue -> déciles
                v = s.to_numpy(dtype=float)
                edges = np.unique(np.quantile(v, q))
                # proportions stockées : les tranches ne font PAS 10 % quand des bornes se
                # confondent (ratios très concentrés : part_dom ~1, ratio_garantie ~0…).
                # Leur nombre sert aussi à choisir l'indicateur : < n_bins tranches = variable
                # trop concentrée pour un PSI -> on comparera les moyennes.
                idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
                props = np.bincount(idx, minlength=len(edges) - 1) / len(v)
                prof["numeric"][c] = {"edges": [float(e) for e in edges],
                                      "props": [round(float(p), 6) for p in props],
                                      "mean":  round(float(v.mean()), 6)}
            else:                                          # discrète (binaire / comptage) -> fréquences
                vc = s.value_counts(normalize=True)
                prof["discret"][c] = {"freqs": {str(k): round(float(v), 6) for k, v in vc.items()},
                                      "mean": round(float(s.mean()), 6)}
    couvert = len(prof["numeric"]) + len(prof["discret"]) + len(prof["categorical"])
    assert couvert == X.shape[1], f"profil incomplet : {couvert}/{X.shape[1]} features"
    return prof


def main():
    art = os.path.join("PROD", "artifacts", VERSION)
    os.makedirs(art, exist_ok=True)

    # 1) features statiques (déterministe) + labels
    S = build_static(TRAIN_CSV)
    Y = pd.read_csv(LABEL_CSV)[["ID", "fraud_flag"]].rename(columns={"fraud_flag": "fraud"})
    df = S.merge(Y, on="ID"); y = df["fraud"].astype(int); ids = df["ID"].tolist()

    # 2) tables produit FIGÉES (apprises sur tout le train)
    B = load_baskets(TRAIN_CSV)
    tables, base_rate = fit_tables(ids, y.to_numpy(), B, alpha=ALPHA)
    pf = apply_tables(tables, base_rate, ids, B).reset_index(drop=True)

    # 3) matrice complète (ordre FEATURES) + encodage figé
    X = pd.concat([df[STATIC_COLS].reset_index(drop=True), pf], axis=1)[FEATURES]
    enc = CatBoostEncoder(cols=CAT, random_state=SEED)
    Xe = enc.fit_transform(X, y)

    # 4) modèle final. params.json est une ENTRÉE obligatoire, pas une option : il porte
    # les hyperparamètres retenus par Optuna dans DEV/MODEL.ipynb (PR-AUC holdout 0,167).
    # Pas de repli codé en dur, qui entraînerait silencieusement un autre modèle.
    params_path = os.path.join(art, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"{params_path} introuvable. Les hyperparamètres retenus par Optuna "
            "(notebook DEV/MODEL.ipynb) sont une entrée obligatoire de l'entraînement."
        )
    params = json.load(open(params_path, encoding="utf-8"))["params"]
    print(f"hyperparamètres chargés depuis params.json ({len(params)} paramètres)")
    model = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbosity=-1).fit(Xe, y)

    # 4b) profil de référence pour le monitoring de la donnée EN ENTRÉE
    profile = build_reference_profile(X, CAT)

    # 5) sérialisation des artefacts
    joblib.dump(enc,   os.path.join(art, "catboost_encoder.joblib"))
    joblib.dump(model, os.path.join(art, "lgbm.joblib"))
    with open(os.path.join(art, "product_tables.json"), "w", encoding="utf-8") as f:
        json.dump({"tables": tables, "base_rate": base_rate, "alpha": ALPHA}, f)
    with open(os.path.join(art, "reference_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    config = {
        "version": VERSION, "seed": SEED,
        "features": FEATURES, "static_cols": STATIC_COLS,
        "product_cols": PRODUCT_COLS, "cat_cols": CAT,
        "params": params,
        "n_train": int(len(df)), "fraud_rate": round(float(y.mean()), 5),
        "train_data": TRAIN_CSV, "train_data_md5": _md5(TRAIN_CSV),
    }
    with open(os.path.join(art, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    n_keys = {k: len(v) for k, v in tables.items()}
    print(f"artefacts figés dans {art}/")
    print(f"  modèle       : lgbm.joblib ({len(FEATURES)} features)")
    print(f"  encodeur     : catboost_encoder.joblib ({CAT})")
    print(f"  tables prod. : product_tables.json  {n_keys}  base_rate={base_rate:.5f}")
    print(f"  config       : config.json (n_train={len(df):,}, md5={config['train_data_md5'][:8]}…)")
    print(f"  profil réf.  : reference_profile.json ({len(profile['numeric'])} continues, "
          f"{len(profile['discret'])} discrètes, {len(profile['categorical'])} catégorielles "
          f"= {len(profile['numeric'])+len(profile['discret'])+len(profile['categorical'])}/{len(FEATURES)} features)")


if __name__ == "__main__":
    main()
