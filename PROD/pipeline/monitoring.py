"""Monitoring : calcul des métriques sur la donnée en entrée.

Ce module ne décide rien et n'écrit rien. Il ne fait que **calculer des métriques**,
à deux moments distincts :

  - **au fil de l'eau**, sur un lot reçu, parce que ces métriques portent sur 100 %
    du trafic et ne seraient plus calculables ensuite (la base de contrôle ne garde
    que 20 % des paniers, et uniquement les conformes) :
        conformity_report(df)  taux de conformité + décompte des raisons de rejet
        coverage(df, tables)   fraîcheur du catalogue produit

  - **en différé**, en relisant la base de contrôle :
        fenetre(flux, jours)              lecture d'une fenêtre glissante
        rapport_drift(profile, seuils)    PSI et écarts de moyenne (délègue à drift)

L'échantillonnage, lui, n'est pas ici : c'est une décision prise au moment de
l'inférence (`inference.route`), pas une métrique.
"""
import os, json
import numpy as np, pandas as pd

from contract import validate_wide
import drift
import store


_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "monitoring_config.json")

def load_config(path=None):
    "charge monitoring_config.json (taux d'échantillonnage, seuils…)."
    with open(path or _CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Métriques calculées sur un lot reçu ─────────────────────────────────────

def conformity_report(df, res=None):
    """KPI de conformité sur un lot : taux + décompte des raisons de rejet.

    `res` : résultat de `contract.validate_wide(df)` s'il a déjà été calculé. La
    validation est le poste le plus coûteux du portail, on évite de la refaire.
    """
    res = validate_wide(df) if res is None else res
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


# ── Métriques calculées en différé, sur la base de contrôle ─────────────────

def fenetre(flux, jours=None, depuis=None):
    """Lecture d'une fenêtre de la base de contrôle.
    jours  : nb de jours en arrière depuis aujourd'hui (fenêtre glissante)
    depuis : date 'YYYY-MM-DD' incluse (prioritaire sur `jours`)"""
    return store.read_window(flux, jours=jours, depuis=depuis)


def taux_fenetre(kpi, col, poids="n_paniers"):
    """Moyenne d'un taux sur la fenêtre, PONDÉRÉE par le volume de chaque lot.

    `kpi` a une ligne par lot traité, pas par jour. Une moyenne simple donnerait le
    même poids à un lot de 50 paniers et à un lot de 3 000. Chaque ligne porte son
    propre dénominateur : `n_recus` pour la conformité, `n_paniers` pour les taux de
    couverture (les rejets n'ont pas de produits à confronter aux tables).
    """
    if kpi.empty:
        return 0.0
    p = kpi[poids].to_numpy(float)
    return float((kpi[col].to_numpy(float) * p).sum() / p.sum()) if p.sum() else 0.0


def rapport_drift(profile, seuils, jours=7, live=None):
    """Drift de la fenêtre contre le profil de référence figé.

    Se lit uniquement sur `flux_a`, l'échantillon uniforme : c'est le seul sous-ensemble
    représentatif de la population, et le seul qui porte les 45 variables calculées.
    """
    live = fenetre("flux_a", jours) if live is None else live
    if live.empty:
        return pd.DataFrame()
    return drift.rapport(live, profile, seuils)


def couverture_fenetre(seuils, jours=7, kpi=None):
    """Les 5 granularités de couverture face à leurs seuils calibrés.

    La lecture doit être CROISÉE : un taux de SKU inconnus qui explose seul peut
    n'être qu'une renumérotation de références, sans dégradation du modèle. Mais SKU,
    modèle et marque au-dessus du seuil en même temps, c'est l'assortiment qui a changé.
    """
    GRAINS = [("taux_cat_inconnus",   "taux_cat_inconnus_max",   "catégorie"),
              ("taux_make_inconnus",  "taux_make_inconnus_max",  "marque"),
              ("taux_dom_inconnu",    "taux_dom_inconnu_max",    "produit dominant"),
              ("taux_model_inconnus", "taux_model_inconnus_max", "modèle"),
              ("taux_sku_inconnus",   "taux_sku_inconnus_max",   "SKU")]
    kpi = fenetre("kpi", jours) if kpi is None else kpi
    if kpi.empty:
        return pd.DataFrame(), 0
    lignes, n_depasses = [], 0
    for col, cle, libelle in GRAINS:
        val, lim = taux_fenetre(kpi, col), seuils[cle]
        etat = "dépassé" if val > lim else ("proche" if val > lim / 2 else "normal")
        n_depasses += etat == "dépassé"
        lignes.append({"granularité": libelle, "inconnus": val, "seuil": lim, "état": etat})
    ordre = {"dépassé": 0, "proche": 1, "normal": 2}
    tab = pd.DataFrame(lignes).sort_values("état", key=lambda s: s.map(ordre)).reset_index(drop=True)
    return tab, n_depasses
