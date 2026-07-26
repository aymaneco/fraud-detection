"""Features de propension-fraude produit, sans fuite.

Principe : on apprend une table `valeur -> taux de fraude` (lissée) sur un jeu
d'entraînement (`fit_tables`), puis on l'applique à n'importe quel jeu (`apply_tables`)
par simple lookup. Le jeu appliqué n'utilise jamais son propre label.

Granularités : goods_code (SKU), model, item (catégorie), make. Pour chaque panier,
on agrège les taux de ses articles (max / moyenne) + le taux du modèle dominant +
un comptage d'articles à haut risque.
"""
import numpy as np, pandas as pd
from collections import defaultdict

ITEM  = [f"item{i}"       for i in range(1, 25)]
PRICE = [f"cash_price{i}" for i in range(1, 25)]
MODEL = [f"model{i}"      for i in range(1, 25)]
CODE  = [f"goods_code{i}" for i in range(1, 25)]
MAKE  = [f"make{i}"       for i in range(1, 25)]
SERV  = {"FULFILMENT CHARGE", "SERVICE"}
KEYS  = {"codes": "sku", "models": "model", "cats": "cat", "makes": "make"}


def load_baskets(source):
    """dict ID -> {codes, models, cats, makes, prices, dom_model} pour les vrais biens
    (hors synthétiques). `prices` sert à pondérer la couverture par le montant.
    source = chemin CSV ou DataFrame."""
    X = pd.read_csv(source, low_memory=False) if isinstance(source, str) else source.copy()
    if "is_synthetic" in X.columns:
        X = X[X.is_synthetic == 0].reset_index(drop=True)
    IT = X[ITEM].to_numpy(); P = X[PRICE].to_numpy(float)
    MO = X[MODEL].to_numpy().astype(str); CD = X[CODE].to_numpy().astype(str)
    MK = X[MAKE].to_numpy().astype(str)
    good = pd.notna(IT) & ~np.isin(IT, list(SERV)) & (IT != "WARRANTY")
    Pm = np.where(good, P, -np.inf); dom = Pm.argmax(1); r = np.arange(len(X))
    ok = np.isfinite(Pm[r, dom])
    B = {}
    for i in range(len(X)):
        g = good[i]; sel = [j for j in range(24) if g[j]]
        B[X.ID.iloc[i]] = dict(
            codes = [CD[i, j] for j in sel],
            models= [MO[i, j] for j in sel],
            cats  = [IT[i, j] for j in sel],
            makes = [MK[i, j] for j in sel],
            prices= [float(P[i, j]) for j in sel],      # pour pondérer la couverture par le montant
            dom_model = MO[r[i], dom[i]] if ok[i] else "NONE",
        )
    return B


def fit_tables(train_ids, train_y, B, alpha=20.0):
    "apprend les tables valeur->taux (lissées) à partir de (train_ids, train_y)."
    train_y = np.asarray(train_y, float); g = float(train_y.mean())
    tables = {}
    for key in KEYS:
        tot = defaultdict(float); fr = defaultdict(float)
        for idx, i in enumerate(train_ids):
            yi = train_y[idx]
            for v in set(B[i][key]):
                tot[v] += 1; fr[v] += yi
        tables[key] = {v: (fr[v] + alpha * g) / (tot[v] + alpha) for v in tot}
    return tables, g


def apply_tables(tables, g, apply_ids, B):
    "applique les tables à apply_ids -> DataFrame de features (lookup, sans label)."
    feats = {}
    for key, s in KEYS.items():
        rate = tables[key]; mx = []; mn = []
        for i in apply_ids:
            rs = [rate.get(v, g) for v in B[i][key]] or [g]
            mx.append(max(rs)); mn.append(float(np.mean(rs)))
        feats[f"{s}_fraude_max"]  = mx
        feats[f"{s}_fraude_mean"] = mn
    feats["dom_model_fraude"]   = [tables["models"].get(B[i]["dom_model"], g) for i in apply_ids]
    feats["nb_articles_risque"] = [int(sum(tables["codes"].get(v, g) > 2 * g for v in B[i]["codes"]))
                                   for i in apply_ids]
    return pd.DataFrame(feats, index=list(apply_ids))


PRODUCT_COLS = [f"{s}_fraude_max" for s in KEYS.values()] + \
               [f"{s}_fraude_mean" for s in KEYS.values()] + \
               ["dom_model_fraude", "nb_articles_risque"]
