"""Contrat d'entrée : UNIQUE définition de « qu'est-ce qu'un panier valide ».

Une seule définition de la validité, quel que soit le format d'arrivée :
  - JSON imbriqué  {ID, articles:[…]}   -> Panier(**payload)
  - CSV format large (item1..24, …)     -> validate_wide(df), qui construit les mêmes
                                           objets Panier emplacement par emplacement

C'est `validate_wide` qui garde la porte du pipeline (`monitoring.check_schema`).

Les contraintes sont calibrées sur les données observées (163 357 emplacements du
train : qty >= 1 toujours, prix jamais négatif ni manquant, goods_code toujours
présent, make/model parfois absents).
"""
from typing import Optional, List
import numpy as np, pandas as pd
from pydantic import BaseModel, Field, ConfigDict, ValidationError, model_validator

SERVICES     = {"FULFILMENT CHARGE", "SERVICE"}
WARRANTY     = "WARRANTY"
NON_PRODUITS = SERVICES | {WARRANTY}
MAX_SLOTS    = 24


class Article(BaseModel):
    """Une ligne du panier : un produit, ou une ligne annexe (service / garantie)."""
    model_config = ConfigDict(extra="forbid")          # une clé inattendue = erreur d'intégration

    item:       str   = Field(min_length=1)            # catégorie (ou FULFILMENT CHARGE / SERVICE / WARRANTY)
    cash_price: float = Field(ge=0, allow_inf_nan=False)
    goods_code: str   = Field(min_length=1)
    qty:        int   = Field(ge=1)
    make:       Optional[str] = None                   # absent dans ~0,8 % des lignes
    model:      Optional[str] = None


class Panier(BaseModel):
    """Un panier soumis au financement."""
    model_config = ConfigDict(extra="forbid")

    ID:        int
    articles:  List[Article] = Field(min_length=1, max_length=MAX_SLOTS)

    @model_validator(mode="after")
    def au_moins_un_bien(self):
        """Règle métier : un panier doit contenir au moins un VRAI produit.
        Sans bien réel, le produit dominant n'existe pas et les 45 features sont
        dégénérées (dom_price=0, dom_cat='NONE'…), on refuse de scorer du vide."""
        if not any(a.item not in NON_PRODUITS for a in self.articles):
            raise ValueError("panier_vide")
        return self


# ── Traduction erreurs Pydantic -> étiquettes stables du monitoring ─────────
# Les KPI et alertes s'appuient sur ces étiquettes : elles ne doivent pas changer.

def _tag(err):
    t   = err.get("type", "")
    loc = err.get("loc", ())
    champ = next((str(x) for x in reversed(loc) if isinstance(x, str)), "")
    if champ == "cash_price":
        return "prix_negatif" if t == "greater_than_equal" else "prix_non_numerique"
    if t == "value_error":
        return "panier_vide" if "panier_vide" in str(err.get("msg", "")) else "regle_metier"
    if t == "too_short":     return "panier_vide"
    if t == "too_long":      return "trop_d_articles"
    if champ == "ID":        return "id_invalide"
    if t == "missing":       return f"champ_manquant({champ})"
    if t == "extra_forbidden": return f"champ_inattendu({champ})"
    return f"{champ or t}_invalide"


def raisons_de(exc: ValidationError):
    "liste ordonnée et dédupliquée des étiquettes d'erreur."
    vus, out = set(), []
    for e in exc.errors():
        tag = _tag(e)
        if tag not in vus:
            vus.add(tag); out.append(tag)
    return out


# ── Validation du format large (CSV) ────────────────────────────────────────

def _cols(prefix):
    return [f"{prefix}{i}" for i in range(1, MAX_SLOTS + 1)]

WIDE_COLS = (["ID"] + _cols("item") + _cols("cash_price") + _cols("make")
             + _cols("model") + _cols("goods_code") + _cols("Nbr_of_prod_purchas"))


def _none(v):
    "NaN/NaT -> None ; sinon str."
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v)


def validate_wide(df):
    """Valide un lot au format large via le contrat Pydantic.
    Renvoie un DataFrame [conforme (bool), raisons (str '|'-séparé)].
    Un emplacement compte comme article dès que `item` est renseigné."""
    manquantes = [c for c in WIDE_COLS if c not in df.columns]
    if manquantes:
        n = len(df)
        return pd.DataFrame({"conforme": [False] * n,
                             "raisons": [f"colonnes_manquantes({len(manquantes)})"] * n})

    IT = df[_cols("item")].to_numpy()
    P  = df[_cols("cash_price")].to_numpy()
    MK = df[_cols("make")].to_numpy()
    MO = df[_cols("model")].to_numpy()
    CD = df[_cols("goods_code")].to_numpy()
    Q  = df[_cols("Nbr_of_prod_purchas")].to_numpy()
    ids = df["ID"].to_numpy()

    conforme, raisons = [], []
    for i in range(len(df)):
        arts = []
        for j in range(MAX_SLOTS):
            if pd.isna(IT[i, j]):
                continue                                    # emplacement vide
            arts.append({"item": IT[i, j], "cash_price": P[i, j],
                         "goods_code": _none(CD[i, j]), "qty": Q[i, j],
                         "make": _none(MK[i, j]), "model": _none(MO[i, j])})
        try:
            Panier.model_validate({"ID": ids[i], "articles": arts})
            conforme.append(True); raisons.append("")
        except ValidationError as e:
            conforme.append(False); raisons.append("|".join(raisons_de(e)))
    return pd.DataFrame({"conforme": conforme, "raisons": raisons})


