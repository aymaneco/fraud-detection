"""Service : l'acte d'inférence complet.

    art = load_artifacts("v2")
    out, kpi = serve(raw_paniers_df, art)     # -> [ID, fraud_proba] + KPI du lot

`serve` est le SEUL point de passage du scoring, quel que soit le canal (batch, API,
flux). Il enchaîne dans l'ordre :

  1. portail         : contrat Pydantic, un panier non conforme n'est PAS scoré
  2. score           : `predict`, transform-only, aucun label
  3. échantillonnage : quels paniers entrent dans la base de contrôle
  4. journalisation  : flux_a (features + score), erreurs, kpi

L'échantillonnage est ICI, dans l'inférence, et non dans un script appelant. Sinon
chaque nouveau point d'entrée devrait penser à le refaire, et le jour où l'un d'eux
l'oublie le monitoring devient aveugle sans que ça se voie.

`predict` reste une fonction PURE en dessous : aucun effet de bord, testable seule.
C'est elle qui sert à vérifier l'équivalence avec le notebook.
"""
import os, json, hashlib
import joblib
import numpy as np, pandas as pd

from contract import validate_wide
from features_static import build_static, STATIC_COLS
from product_feats import load_baskets, apply_tables, PRODUCT_COLS
from monitoring import load_config, conformity_report, coverage
import store

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


# ── Calcul pur : paniers -> probabilités ────────────────────────────────────

def predict(raw_df, art, with_features=False):
    """raw_df : paniers bruts (format large item1..24…). Renvoie [ID, fraud_proba].

    Avec `with_features=True`, renvoie aussi la matrice des 45 variables, indexée
    par ID. C'est ce dont `serve` a besoin pour alimenter la base de contrôle, sans
    avoir à recalculer les features une seconde fois.
    """
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

    # 4) sortie sur TOUS les paniers d'entrée : réels -> proba, synthétiques -> base 0
    out = pd.DataFrame({"ID": raw_df["ID"].to_numpy()})
    out["fraud_proba"] = out["ID"].map(dict(zip(ids_real, proba))).fillna(0.0)
    if with_features:
        X = X.copy(); X.index = ids_real
        return out, X
    return out


# ── Échantillonnage : quels paniers entrent dans la base de contrôle ────────

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
      - flux_a : tiré dans l'échantillon uniforme ET conforme (le PSI exige des
                 features valides). Cet échantillon doit rester UNIFORME : l'enrichir
                 des scores élevés le rendrait non représentatif.
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


# ── L'acte d'inférence : valider, scorer, échantillonner, journaliser ───────

def serve(raw_df, art, cfg=None, ts=None):
    """Traite un lot de paniers de bout en bout. Renvoie ([ID, fraud_proba], kpi).

    `ts` force l'horodatage des partitions écrites (utilisé par la simulation pour
    fabriquer plusieurs jours ; en production on laisse l'heure courante).
    """
    cfg = cfg or load_config()

    # 1) portail : la validation précède TOUT calcul
    sc   = validate_wide(raw_df)
    conf = sc["conforme"].to_numpy()
    ok   = raw_df[conf].reset_index(drop=True)

    # 2) score des seuls paniers conformes
    out, X = predict(ok, art, with_features=True)
    score_par_id = dict(zip(out["ID"], out["fraud_proba"]))

    # 3) échantillonnage, sur le lot COMPLET : c'est `route` qui tranche
    r = route(raw_df["ID"], sc["conforme"], raw_df["ID"].map(score_par_id), cfg)

    # 4) journalisation
    log = X.copy()
    log.insert(0, "ID", X.index.to_numpy())
    log["score"] = [score_par_id[i] for i in X.index]
    store.append(log[log["ID"].isin(r.loc[r.flux_a, "ID"])], "flux_a", ts=ts)

    err = pd.DataFrame({"ID": raw_df["ID"].to_numpy(), "raisons": sc["raisons"].to_numpy(),
                        "raison_b": r["raison_b"].to_numpy()})[r["flux_b"].to_numpy()]
    store.append(err, "erreurs", ts=ts)

    # 5) KPI du lot. Ils portent sur TOUS les paniers reçus, pas sur l'échantillon :
    # ils ne seraient pas recalculables depuis flux_a, d'où leur archivage ici.
    rep    = conformity_report(raw_df, sc)
    cov, _ = coverage(ok, art["tables"])
    jour   = (ts or store._now()).strftime("%Y-%m-%d")
    kpi = {"date": jour, "n_recus": rep["n"], "n_conformes": rep["conformes"],
           "taux_conformite": rep["taux_conformite"], **cov}
    store.append(pd.DataFrame([kpi]), "kpi", ts=ts)

    return out, kpi
