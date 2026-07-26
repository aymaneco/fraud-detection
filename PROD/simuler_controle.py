"""Génère une base de contrôle SIMULÉE, pour la démonstration du dashboard.

7 jours de trafic construits à partir du jeu de test, avec une dérive **cumulative**
(chaque jour conserve les dérives des précédents), plus réaliste qu'un pic isolé et
surtout visible quelle que soit la fenêtre choisie :

  J-6, J-5 : régime normal                              -> tout vert (référence)
  J-4      : + prix −25 %                               -> PSI prix commence à monter
  J-3      : + prix −45 %                               -> PSI prix franchit le seuil
  J-2      : + nouveaux produits (35 %)                 -> couverture + PSI taux produit
  J-1      : + mix marques (SAMSUNG) + garanties retirées -> PSI dom_make, has_warranty
  J0       : + livraison forcée + paniers agrandis      -> logistique + structure

Plus quelques paniers non conformes chaque jour -> taux de conformité.

Chaque panier passe par le VRAI pipeline (portail -> features figées -> score) puis
par le VRAI échantillonnage (20 % au hash) : la base produite est de même nature que
celle qu'alimenterait la production.

Usage :  python PROD/simuler_controle.py
"""
import os, sys, json, shutil
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "PROD", "pipeline"))

import numpy as np, pandas as pd
from monitoring import check_schema, conformity_report, route, load_config, coverage
from features_static import build_static, STATIC_COLS
from product_feats import load_baskets, apply_tables
from inference import load_artifacts
import store

N_JOURS   = 7
PAR_JOUR  = 3000
ITEM_COLS = [f"item{i}" for i in range(1, 25)]
PRIX_COLS = [f"cash_price{i}" for i in range(1, 25)]
CODE_COLS = [f"goods_code{i}" for i in range(1, 25)]
MODEL_COLS= [f"model{i}" for i in range(1, 25)]
MAKE_COLS = [f"make{i}" for i in range(1, 25)]
QTY_COLS  = [f"Nbr_of_prod_purchas{i}" for i in range(1, 25)]


def _obj(lot, cols):
    "passe les colonnes en object : certaines sont lues en float64 quand elles sont vides."
    for c in cols:
        lot[c] = lot[c].astype(object)
    return lot


def casser(lot, n=3):
    "injecte n paniers non conformes (bruit d'intégration réaliste)."
    bad = _obj(lot.head(n).copy(), [PRIX_COLS[0]] + ITEM_COLS)
    bad["ID"] = [900000 + i for i in range(n)]
    bad.iloc[0, bad.columns.get_loc(PRIX_COLS[0])] = -49.9        # prix négatif
    if n > 1:
        bad.iloc[1, [bad.columns.get_loc(c) for c in ITEM_COLS]] = np.nan   # panier vide
    return pd.concat([lot, bad], ignore_index=True)


def derive(lot, jour_idx, rng):
    """Dérive CUMULATIVE : chaque jour ajoute une couche aux précédentes.
    Les fenêtres 1/3/7 jours montrent donc toutes quelque chose, d'intensité croissante."""
    lot = lot.copy()

    # ① prix : baisse progressive à partir de J-4
    facteur = {2: 0.75, 3: 0.55, 4: 0.55, 5: 0.55, 6: 0.55}.get(jour_idx)
    if facteur:
        for c in PRIX_COLS:
            lot[c] = lot[c] * facteur

    # ② nouveaux produits (SKU / modèles / marques inconnus) à partir de J-2
    if jour_idx >= 4:
        for cols, pref in [(CODE_COLS, "NEW"), (MODEL_COLS, "NOUVEAU"), (MAKE_COLS, "MARQUE_X")]:
            lot = _obj(lot, cols)
            for col in cols:
                m = rng.random(len(lot)) < 0.35
                lot.loc[m, col] = pref + "_" + lot.loc[m, col].astype(str)

    # ③ mix marques qui bascule + garanties qui disparaissent, à partir de J-1
    if jour_idx >= 5:
        lot = _obj(lot, MAKE_COLS + ITEM_COLS)
        for col in MAKE_COLS:
            m = rng.random(len(lot)) < 0.55
            lot.loc[m, col] = "SAMSUNG"
        for col in ITEM_COLS:
            lot.loc[lot[col] == "WARRANTY", col] = np.nan       # plus de garanties

    # ④ J0 : livraison systématique + paniers agrandis (structure)
    if jour_idx >= 6:
        lot = _obj(lot, ITEM_COLS + CODE_COLS + MODEL_COLS + MAKE_COLS)
        vide = lot[ITEM_COLS[1]].isna()
        lot.loc[vide, ITEM_COLS[1]] = "FULFILMENT CHARGE"       # frais de port partout
        lot.loc[vide, PRIX_COLS[1]] = 0.0
        lot.loc[vide, CODE_COLS[1]] = "SERV_LIV"
        lot.loc[vide, MAKE_COLS[1]] = "RETAILER"
        lot.loc[vide, MODEL_COLS[1]] = "RETAILER"
        lot.loc[vide, QTY_COLS[1]] = 1
        # on duplique le 1er bien sur 2 emplacements libres -> paniers plus gros
        for k in (2, 3):
            libre = lot[ITEM_COLS[k]].isna()
            for src, dst in [(ITEM_COLS, ITEM_COLS), (PRIX_COLS, PRIX_COLS), (CODE_COLS, CODE_COLS),
                             (MAKE_COLS, MAKE_COLS), (MODEL_COLS, MODEL_COLS), (QTY_COLS, QTY_COLS)]:
                lot.loc[libre, dst[k]] = lot.loc[libre, src[0]]
    return lot


def main():
    cfg = load_config()
    art = load_artifacts("v2")
    FEAT = art["config"]["features"]

    base_dir = os.path.join(ROOT, "control_base")
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)                  # repart d'une base propre

    te = pd.read_csv("data/X_test_clean.csv", low_memory=False)
    rng = np.random.default_rng(0)
    resume = []

    for j in range(N_JOURS):
        jour = datetime.now(timezone.utc) - timedelta(days=N_JOURS - 1 - j)
        src  = te.sample(PAR_JOUR, random_state=100 + j).reset_index(drop=True)
        src["ID"] = src["ID"] + j * 1_000_000    # IDs distincts d'un jour à l'autre
        lot  = casser(derive(src, j, rng))

        # 1) portail
        sc = check_schema(lot)
        ok = lot[sc["conforme"].to_numpy()].reset_index(drop=True)

        # 2) features figées + score (le vrai pipeline)
        S  = build_static(ok)
        B  = load_baskets(ok)
        pf = apply_tables(art["tables"], art["base"], S["ID"].tolist(), B).reset_index(drop=True)
        X  = pd.concat([S[STATIC_COLS], pf], axis=1)[FEAT]
        proba = art["model"].predict_proba(art["encoder"].transform(X))[:, 1]

        # 3) couverture du jour (indicateur de fraîcheur du catalogue)
        cov, _ = coverage(ok, art["tables"], B)

        # 4) routage sur le LOT COMPLET, conformes et rejets : c'est `route` qui décide
        # des deux flux, pas la simulation. Les non conformes n'ont pas de score.
        proba_par_id = dict(zip(S["ID"].to_numpy(), proba))
        r = route(lot["ID"], sc["conforme"], lot["ID"].map(proba_par_id), cfg)

        log = X.copy()
        log.insert(0, "ID", S["ID"].to_numpy())
        log["score"] = proba
        log = log[log["ID"].isin(r.loc[r.flux_a, "ID"])]
        store.append(log, "flux_a", ts=jour)

        err = pd.DataFrame({"ID": lot["ID"].to_numpy(), "raisons": sc["raisons"].to_numpy(),
                            "raison_b": r["raison_b"].to_numpy()})[r["flux_b"].to_numpy()]
        store.append(err, "erreurs", ts=jour)

        # 5) KPI du jour, archivés à côté (le job batch les recalculerait)
        rep = conformity_report(lot, sc)          # sc déjà calculé : pas de 2e validation
        kpi = {"date": jour.strftime("%Y-%m-%d"), "n_recus": rep["n"],
               "n_conformes": rep["conformes"], "taux_conformite": rep["taux_conformite"], **cov}
        store.append(pd.DataFrame([kpi]), "kpi", ts=jour)
        resume.append(kpi)

    print(f"base de contrôle simulée : {N_JOURS} jours\n")
    cols = ["date", "n_recus", "taux_conformite", "taux_sku_inconnus",
            "taux_make_inconnus", "taux_dom_inconnu"]
    print(pd.DataFrame(resume)[cols].to_string(index=False))
    print()
    for flux, s in store.stats().items():
        print(f"  {flux:8s} {s['lignes']:>6,} lignes | {s['partitions']} partitions | {s['taille_Mo']} Mo")


if __name__ == "__main__":
    main()
