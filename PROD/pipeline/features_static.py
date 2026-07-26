"""Feature engineering statique (7 facettes EDA) : source de vérité dev ET prod.

Déterministe, SANS label : `build_static(path)` reproduit les variables statiques de
`data/features_train.csv` (hors `rand`, sentinelle de dev). Utilisé à l'entraînement
(`train.py`) et au service (`inference.py`) pour garantir zéro divergence.
"""
import numpy as np, pandas as pd

item_cols  = [f"item{i}"               for i in range(1, 25)]
price_cols = [f"cash_price{i}"         for i in range(1, 25)]
make_cols  = [f"make{i}"               for i in range(1, 25)]
qty_cols   = [f"Nbr_of_prod_purchas{i}" for i in range(1, 25)]
code_cols  = [f"goods_code{i}"         for i in range(1, 25)]

SERVICES    = {"FULFILMENT CHARGE", "SERVICE"}
WARRANTY    = "WARRANTY"
LIQUID_CATS = {"COMPUTERS", "TELEPHONES FAX MACHINES TWO WAY RADIOS"}
PRAM_MAKES  = {"SILVER CROSS", "BUGABOO"}


def _r(num, den):
    "ratio robuste : 0 quand le dénominateur est nul."
    den = np.asarray(den, float)
    return np.where(den > 0, np.asarray(num, float) / np.where(den > 0, den, 1), 0.0)

def _nunique_rows(V, mask):
    return np.array([len(set(V[i][mask[i]])) for i in range(len(V))])

def _prep(X):
    IT = X[item_cols].to_numpy(); P = X[price_cols].to_numpy(float)
    MK = X[make_cols].to_numpy(); Q = X[qty_cols].to_numpy(float); CD = X[code_cols].to_numpy()
    good = pd.notna(IT) & ~np.isin(IT, list(SERVICES)) & (IT != WARRANTY)
    return dict(ID=X.ID.to_numpy(), IT=IT, P=P, MK=MK, Q=Q, CD=CD, good=good, n=len(X))


def facette_enjeu(A):
    P, good = A["P"], A["good"]
    Pg = np.where(good, P, 0.0); total = Pg.sum(1)
    Pm = np.where(good, P, -np.inf); dom = Pm.argmax(1); r = np.arange(A["n"])
    ok = np.isfinite(Pm[r, dom]); dom_price = np.where(ok, P[r, dom], 0.0)
    return {"total": total, "dom_price": dom_price,
            "val_high": np.where(good & (P > 1000), P, 0.0).sum(1),
            "nb_biens_chers": ((good) & (P > 1000)).sum(1),
            "prix_rond": ((dom_price > 0) & (dom_price % 500 == 0)).astype(int)}

def facette_liquidite(A):
    P, MK, IT, good = A["P"], A["MK"], A["IT"], A["good"]
    Pm = np.where(good, P, -np.inf); dom = Pm.argmax(1); r = np.arange(A["n"]); ok = np.isfinite(Pm[r, dom])
    dom_cat = np.where(ok, IT[r, dom], "NONE"); dom_make = np.where(ok, MK[r, dom], "NONE")
    liq_mask = good & np.isin(IT, list(LIQUID_CATS)); total = np.where(good, P, 0.0).sum(1)
    return {"dom_cat": dom_cat, "dom_make": dom_make,
            "is_apple": (dom_make == "APPLE").astype(int),
            "is_pram_premium": np.isin(dom_make, list(PRAM_MAKES)).astype(int),
            "is_cat_liquide": np.isin(dom_cat, list(LIQUID_CATS)).astype(int),
            "part_liquide": _r(np.where(liq_mask, P, 0.0).sum(1), total)}

def facette_concentration(A):
    P, good = A["P"], A["good"]
    Pg = np.where(good, P, 0.0); total = Pg.sum(1); taille = good.sum(1)
    Psort = -np.sort(-Pg, axis=1); top1, top2 = Psort[:, 0], Psort[:, 1]
    Pn = np.where(good, P, np.nan); prix_std = np.nan_to_num(np.nanstd(Pn, axis=1))
    prix_moy = _r(total, taille)
    return {"part_dom": _r(top1, total), "part_top2": _r(top1 + top2, total),
            "ecart_dom": top1 - top2, "prix_std": prix_std, "prix_cv": _r(prix_std, prix_moy)}

def facette_structure(A):
    P, Q, CD, good = A["P"], A["Q"], A["CD"], A["good"]
    taille = good.sum(1); Qg = np.where(good, np.nan_to_num(Q), 0.0)
    return {"taille": taille, "is_mono": (taille == 1).astype(int), "is_duo": (taille == 2).astype(int),
            "is_3plus": (taille >= 3).astype(int), "nb_distinct": _nunique_rows(CD, good),
            "qty_total": Qg.sum(1), "max_qty": Qg.max(1), "n_multi": (Qg > 1).sum(1),
            "nb_petits": (good & (P < 50)).sum(1)}

def facette_rationalite(A):
    IT, good = A["IT"], A["good"]; taille = good.sum(1)
    nb_warr = (IT == WARRANTY).sum(1)
    acc = pd.Series(IT.ravel()).str.contains("ACCESSOR|CABLE", na=False).to_numpy().reshape(IT.shape)
    acc_mask = good & acc
    return {"has_warranty": (nb_warr > 0).astype(int), "nb_garanties": nb_warr,
            "ratio_garantie": _r(nb_warr, taille), "has_accessoires": (acc_mask.sum(1) > 0).astype(int)}

def facette_logistique(A):
    IT, good = A["IT"], A["good"]
    nb_serv = np.isin(IT, list(SERVICES)).sum(1); taille = good.sum(1)
    return {"has_delivery": (nb_serv > 0).astype(int), "nb_services": nb_serv,
            "ratio_service": _r(nb_serv, taille + nb_serv)}

def facette_coherence(A):
    IT, MK, good = A["IT"], A["MK"], A["good"]
    no_brand = good & (pd.isna(MK) | (MK == "RETAILER"))
    return {"nb_cat_distinctes": _nunique_rows(IT, good),
            "nb_make_distinctes": _nunique_rows(np.where(pd.isna(MK), "NONE", MK), good),
            "nb_sans_marque": no_brand.sum(1)}

FACETTES = [facette_enjeu, facette_liquidite, facette_concentration, facette_structure,
            facette_rationalite, facette_logistique, facette_coherence]


def build_static(source, exclude_synthetic=True):
    "source = chemin CSV ou DataFrame. Renvoie ID + 35 variables statiques (ordre facettes)."
    X = pd.read_csv(source, low_memory=False) if isinstance(source, str) else source.copy()
    if exclude_synthetic and "is_synthetic" in X.columns:
        X = X[X.is_synthetic == 0].reset_index(drop=True)
    A = _prep(X)
    out = {"ID": A["ID"]}
    for f in FACETTES:
        out.update(f(A))
    return pd.DataFrame(out)


STATIC_COLS = [
    "total", "dom_price", "val_high", "nb_biens_chers", "prix_rond",
    "dom_cat", "dom_make", "is_apple", "is_pram_premium", "is_cat_liquide", "part_liquide",
    "part_dom", "part_top2", "ecart_dom", "prix_std", "prix_cv",
    "taille", "is_mono", "is_duo", "is_3plus", "nb_distinct", "qty_total", "max_qty", "n_multi", "nb_petits",
    "has_warranty", "nb_garanties", "ratio_garantie", "has_accessoires",
    "has_delivery", "nb_services", "ratio_service",
    "nb_cat_distinctes", "nb_make_distinctes", "nb_sans_marque",
]
