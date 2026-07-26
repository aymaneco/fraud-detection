"""Mesure du drift de la donnée en entrée : fenêtre live vs profil de référence.

Deux indicateurs, choisis selon la nature de la variable :

  - **PSI** (Population Stability Index) pour les variables dont la distribution a
    une *forme* à surveiller : continues bien étalées (prix, montants, taux produit)
    et catégorielles (mix de catégories / marques). Une moyenne pourrait rester
    stable alors que la population s'est réorganisée : le PSI, lui, le voit.
        PSI = Σ (p_live − p_ref) × ln(p_live / p_ref)
        < 0,1 stable | 0,1–0,25 à surveiller | > 0,25 drift significatif

  - **Écart de moyenne** pour les variables trop concentrées pour un découpage utile :
    binaires (une loi de Bernoulli est ENTIÈREMENT décrite par sa moyenne : le PSI
    n'apporterait rien), comptages, et ratios bimodaux (`part_liquide`, `part_dom`…).
    Plus lisible : « 8,7 % → 4,1 % (−4,6 pts) ».

Le choix est automatique : une continue dont les déciles se sont *effondrés* (moins de
tranches que demandé) est trop concentrée -> écart de moyenne.
"""
import numpy as np, pandas as pd

EPS = 1e-4          # remplace les proportions nulles : ln(0) = -inf
N_BINS = 10


def psi(ref_props, live_props, eps=EPS):
    "PSI entre deux vecteurs de proportions (même découpage, même ordre)."
    r = np.clip(np.asarray(ref_props, float), eps, None)
    l = np.clip(np.asarray(live_props, float), eps, None)
    return float(np.sum((l - r) * np.log(l / r)))


def props_continue(values, edges):
    "range les valeurs live dans les tranches FIGÉES du profil -> proportions."
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.zeros(len(edges) - 1)
    idx = np.clip(np.searchsorted(np.asarray(edges), v, side="right") - 1, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1) / len(v)


def props_modalites(serie, modalites):
    "fréquences live des modalités de référence (+ 'AUTRES' pour les inconnues)."
    vc = serie.astype(str).value_counts(normalize=True)
    connues = np.array([float(vc.get(m, 0.0)) for m in modalites])
    return np.append(connues, max(0.0, 1.0 - connues.sum()))       # dernière case = modalités hors référence


def contributeur_principal(ref_p, live_p, labels):
    """Quelle tranche / modalité explique le PSI ? -> « libellé : ref % → live % ».

    Un PSI seul ne dit pas QUOI a bougé. On renvoie le poste qui pèse le plus dans
    la somme, ce qui rend l'alerte directement actionnable.
    """
    r = np.clip(np.asarray(ref_p, float), EPS, None)
    l = np.clip(np.asarray(live_p, float), EPS, None)
    c = (l - r) * np.log(l / r)
    i = int(np.argmax(c))
    part = c[i] / c.sum() if c.sum() else 0.0
    return f"{labels[i]} : {ref_p[i]:.1%} → {live_p[i]:.1%}  ({part:.0%} du PSI)"


def libelles_tranches(edges):
    "libellés lisibles des tranches d'une variable continue."
    e = np.asarray(edges, float)
    return [f"{e[i]:,.0f}–{e[i+1]:,.0f}".replace(",", " ") for i in range(len(e) - 1)]


STATUTS = ("drift", "surveiller", "stable")     # du plus grave au plus sain


PLANCHER_ABS = 0.01     # un écart plus petit que ça n'a aucune portée métier


def statut(valeur, indicateur, seuils, ecart_abs=None):
    """Statut en texte simple. La mise en couleur appartient à la présentation
    (dashboard), pas au calcul : ça garde ce module utilisable en console/log.

    Pour l'écart de moyenne, on exige un écart RELATIF *et* ABSOLU significatifs :
    sans le plancher absolu, une variable très rare (`is_pram_premium`, taux de base
    0,32 %) passerait de 0,32 % à 0,16 %, soit −50 % en relatif mais 0,16 **point**
    en absolu, et déclencherait une fausse alerte à chaque fenêtre.
    """
    if indicateur == "PSI":
        if valeur >= seuils["psi_alerte"]:      return "drift"
        if valeur >= seuils["psi_surveiller"]:  return "surveiller"
        return "stable"
    if ecart_abs is not None and abs(ecart_abs) < PLANCHER_ABS:
        return "stable"
    if abs(valeur) >= 0.50: return "drift"
    if abs(valeur) >= 0.20: return "surveiller"
    return "stable"


def rapport(live, profile, seuils):
    """Un indicateur par feature. Renvoie un DataFrame trié par sévérité.

    colonnes : feature | famille | indicateur | ref | live | valeur | cause | statut
      - `valeur` : PSI, ou écart relatif de moyenne selon l'indicateur
      - `cause`  : ce qui a bougé (tranche ou modalité principale), pour rendre
                   l'alerte actionnable sans avoir à ouvrir la donnée
    """
    lignes = []

    # --- continues : PSI si les déciles tiennent, sinon écart de moyenne ---
    for f, d in profile.get("numeric", {}).items():
        if f not in live.columns:
            continue
        v = live[f].to_numpy(float)
        if len(d["props"]) >= N_BINS:
            live_p = props_continue(v, d["edges"])
            val = psi(d["props"], live_p)
            lignes.append({"feature": f, "famille": "continue", "indicateur": "PSI",
                           "ref": d["mean"], "live": float(np.nanmean(v)), "valeur": val,
                           "cause": contributeur_principal(d["props"], live_p,
                                                           libelles_tranches(d["edges"])),
                           "statut": statut(val, "PSI", seuils)})
        else:
            ref, liv = d["mean"], float(np.nanmean(v))
            rel = (liv - ref) / abs(ref) if ref else 0.0
            lignes.append({"feature": f, "famille": "concentrée", "indicateur": "écart moyenne",
                           "ref": ref, "live": liv, "valeur": rel,
                           "cause": f"moyenne : {ref:,.4g} → {liv:,.4g}",
                           "statut": statut(rel, "écart", seuils, ecart_abs=liv - ref)})

    # --- discrètes (binaires, comptages) : écart de moyenne ---
    for f, d in profile.get("discret", {}).items():
        if f not in live.columns:
            continue
        ref, liv = d["mean"], float(np.nanmean(live[f].to_numpy(float)))
        rel = (liv - ref) / abs(ref) if ref else 0.0
        lignes.append({"feature": f, "famille": "discrète", "indicateur": "écart moyenne",
                       "ref": ref, "live": liv, "valeur": rel,
                       "cause": f"taux : {ref:.1%} → {liv:.1%}",
                       "statut": statut(rel, "écart", seuils, ecart_abs=liv - ref)})

    # --- catégorielles : PSI sur les fréquences des modalités ---
    for f, d in profile.get("categorical", {}).items():
        if f not in live.columns:
            continue
        mods = list(d["freqs"].keys())
        ref_p = np.append(np.array([d["freqs"][m] for m in mods], float), EPS)  # + case 'AUTRES'
        live_p = props_modalites(live[f], mods)
        val = psi(ref_p, live_p)
        lignes.append({"feature": f, "famille": "catégorielle", "indicateur": "PSI",
                       "ref": np.nan, "live": np.nan, "valeur": val,
                       "cause": contributeur_principal(ref_p, live_p, mods + ["<inconnues>"]),
                       "statut": statut(val, "PSI", seuils)})

    df = pd.DataFrame(lignes)
    ordre = {s: i for i, s in enumerate(STATUTS)}
    return df.sort_values(["statut", "valeur"], key=lambda s: s.map(ordre) if s.name == "statut" else s.abs(),
                          ascending=[True, False]).reset_index(drop=True)
