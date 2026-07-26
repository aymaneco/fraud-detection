"""Monitoring de la donnée en entrée.

Niveau immédiat (1 panier, temps réel) :
  - check_schema(df)     : le panier est-il bien formé ? -> conforme + raisons
  - conformity_report(df): agrège le taux de conformité + les raisons de rejet (KPI)
  - route(...)           : quels paniers on garde dans la base de contrôle (flux A / flux B)

Le portail : un panier non conforme est LOGGÉ (avec sa raison) et NE PASSE PAS au modèle.

Deux flux indépendants, jamais mélangés :
  - flux A : échantillon UNIFORME (hash sur ID, taux configurable) -> seul flux lu par le PSI
  - flux B : flux enrichi (non conformes, scores élevés) -> diagnostic / futures étiquettes
Un panier peut appartenir aux deux : ce sont deux drapeaux, pas une catégorie.
"""
import os, json, hashlib
import numpy as np, pandas as pd
from contract import validate_wide


def check_schema(df):
    """Portail : valide chaque panier au format large.
    Renvoie [conforme (bool), raisons (str '|'-séparé)].

    Délègue au contrat Pydantic (`contract.validate_wide`) : une seule définition
    de la validité, partagée avec l'API. Les étiquettes de `raisons` sont stables
    (prix_negatif, panier_vide, colonnes_manquantes…) car les KPI s'appuient dessus.
    """
    return validate_wide(df)


def conformity_report(df, res=None):
    """KPI de conformité sur un lot : taux + décompte des raisons de rejet.

    `res` : résultat de `check_schema(df)` s'il a déjà été calculé. La validation
    est le poste le plus coûteux du portail, on évite de la refaire pour rien.
    """
    res = check_schema(df) if res is None else res
    n = len(res); nc = int(res["conforme"].sum())
    from collections import Counter
    raisons = Counter()
    for r in res.loc[~res["conforme"], "raisons"]:
        for tag in r.split("|"):
            if tag:
                raisons[tag] += 1
    return {"n": int(n), "conformes": nc, "non_conformes": int(n - nc),
            "taux_conformite": round(nc / n, 5) if n else 1.0,
            "raisons": dict(raisons)}


# ── Échantillonnage : quels paniers entrent dans la base de contrôle ────────

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "monitoring_config.json")

def load_config(path=None):
    "charge monitoring_config.json (taux d'échantillonnage, seuils…)."
    with open(path or _CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


def bucket(value):
    """ID -> entier 0..99, STABLE (md5 : identique entre processus et serveurs).
    On n'utilise pas hash() natif : il est randomisé par processus (PYTHONHASHSEED)."""
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % 100


def in_sample_a(ids, rate_pct):
    "masque booléen : le panier fait-il partie de l'échantillon uniforme (flux A) ?"
    return np.array([bucket(v) < rate_pct for v in ids])


def route(ids, conforme, proba=None, cfg=None):
    """Décide, pour chaque panier, s'il entre dans la base de contrôle.

    Renvoie [ID, flux_a, flux_b, raison_b] :
      - flux_a : tiré dans l'échantillon uniforme ET conforme (le PSI exige des features valides)
      - flux_b : non conforme, ou score >= seuil
    Les deux peuvent être vrais simultanément (drapeaux indépendants).
    """
    cfg = cfg or load_config()
    rate = cfg["flux_a"]["sample_rate_pct"]
    b_cfg = cfg["flux_b"]
    ids = np.asarray(ids); conforme = np.asarray(conforme, bool)

    flux_a = in_sample_a(ids, rate) & conforme

    flux_b = np.zeros(len(ids), bool)
    raison_b = np.array([""] * len(ids), dtype=object)
    if b_cfg.get("non_conformes", True):
        m = ~conforme
        flux_b |= m; raison_b[m] = "non_conforme"
    seuil = b_cfg.get("score_threshold")
    if seuil is not None and proba is not None:
        p = pd.to_numeric(pd.Series(proba), errors="coerce").fillna(-1).to_numpy()
        m = (p >= seuil) & conforme
        flux_b |= m
        raison_b[m] = np.where(raison_b[m] == "", "score_eleve", raison_b[m] + "|score_eleve")

    return pd.DataFrame({"ID": ids, "flux_a": flux_a, "flux_b": flux_b, "raison_b": raison_b})


# ── Couverture des tables produit (fraîcheur du catalogue) ──────────────────

def coverage(raw_df, tables, B=None):
    """Le modèle connaît-il les produits de ce lot ?

    Nos features les plus fortes viennent des tables produit figées (SKU / modèle).
    Un produit absent des tables retombe sur `base_rate` : le modèle perd son signal
    le plus discriminant sur ce panier. Un taux d'inconnus qui monte = le **catalogue
    a évolué** -> signal de réentraînement.

    Couvre les 4 granularités des tables (SKU, modèle, catégorie, marque). Plus la
    cardinalité d'une granularité est basse, plus un inconnu est parlant : un SKU
    inconnu est banal (14 296 références, nouveauté constante), une **catégorie**
    inconnue (135) est un événement majeur : le retailer entre dans un nouvel univers.

    Renvoie (metriques: dict, par_panier: DataFrame).
    """
    from product_feats import load_baskets, KEYS
    B = B if B is not None else load_baskets(raw_df)
    ids = [i for i in raw_df["ID"].tolist() if i in B]      # les synthétiques sont écartés par load_baskets

    tot = {k: 0 for k in KEYS}; inc = {k: 0 for k in KEYS}
    val_tot = val_inc = 0.0
    dom_inc = 0
    lignes = []
    for i in ids:
        b = B[i]
        inc_sku = None
        for k in KEYS:                                      # une boucle : plus de granularité codée en dur
            u = [v not in tables[k] for v in b[k]]
            tot[k] += len(u); inc[k] += sum(u)
            if k == "codes":
                inc_sku = u
        prices  = b["prices"]
        v_tot   = sum(prices)
        v_inc   = sum(p for p, u in zip(prices, inc_sku) if u)
        val_tot += v_tot; val_inc += v_inc
        d_inc = b["dom_model"] not in tables["models"]
        dom_inc += int(d_inc)
        lignes.append({"ID": i, "n_sku_inconnus": int(sum(inc_sku)),
                       "dom_model_inconnu": bool(d_inc),
                       "part_valeur_inconnue": round(v_inc / v_tot, 4) if v_tot > 0 else 0.0})

    n = max(len(ids), 1)
    met = {"n_paniers": len(ids)}
    for k, court in KEYS.items():                           # taux_sku/model/cat/make_inconnus
        met[f"taux_{court}_inconnus"] = round(inc[k] / max(tot[k], 1), 5)
    met["taux_dom_inconnu"]     = round(dom_inc / n, 5)     # produit PRINCIPAL inconnu -> alerte n°1
    met["part_valeur_inconnue"] = round(val_inc / val_tot, 5) if val_tot else 0.0
    return met, pd.DataFrame(lignes)
