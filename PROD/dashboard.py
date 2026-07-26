"""Dashboard de monitoring de la donnée en entrée (Streamlit, minimaliste).

    streamlit run PROD/dashboard.py

Cinq blocs, une seule page :
  1. Indicateurs du jour (volume, conformité, alertes)
  2. Couverture du catalogue : les 5 granularités face à leurs seuils calibrés
  3. Évolution jour par jour (conformité + fraîcheur du catalogue)
  4. Drift des variables suivies sur la fenêtre
  5. Drift jour par jour d'une variable au choix, indispensable car une dérive d'un
     seul jour se dilue dans une fenêtre large et peut passer inaperçue.
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "PROD", "pipeline"))

# le dashboard ne lit la base de contrôle qu'à travers `monitoring` : il affiche,
# il ne calcule pas.
import numpy as np, pandas as pd, streamlit as st
import monitoring

st.set_page_config(page_title="Monitoring de la donnée en entrée", layout="wide")

COULEUR = {"drift": "#c0392b", "surveiller": "#e67e22", "stable": "#27ae60"}
PUCE    = {"drift": "🔴", "surveiller": "🟠", "stable": "🟢"}

# ── Définitions affichées dans les infobulles « ? » ────────────────────────
# Un dashboard de monitoring est lu par des gens qui ne connaissent pas le modèle.
# Chaque indicateur porte donc sa définition, son PÉRIMÈTRE et son seuil : un taux
# sans son périmètre n'est pas interprétable.
AIDE = {
"fenetre":
    "Nombre de jours agrégés. Les compteurs se somment et les taux sont repondérés par "
    "le volume de chaque lot. Attention : une fenêtre large **lisse** un incident d'une "
    "seule journée, d'où la vue jour par jour en bas de page.",

"recus":
    "Nombre total de paniers arrivés sur la fenêtre, et nombre retenu dans l'échantillon "
    "de contrôle (tirage stable par `md5(ID) % 100`).\n\n"
    "Périmètres différents selon l'indicateur : la **conformité** porte sur 100 % des "
    "paniers reçus, la **couverture** sur 100 % des paniers conformes, le **drift** sur "
    "l'échantillon uniforme seulement.",

"conformite":
    "Part des paniers qui passent le contrat d'entrée : types corrects, prix positif ou "
    "nul, quantité au moins égale à 1, et **au moins un vrai bien** (hors frais de port, "
    "service et garantie).\n\n"
    "Calculée sur **100 % des paniers reçus**. Un panier non conforme est journalisé avec "
    "sa raison et **n'est pas scoré**. Seuil d'alerte : 98 %.",

"dom":
    "Part des paniers dont le **bien le plus cher** a un modèle absent des tables produit "
    "figées à l'entraînement.\n\n"
    "Conséquence : le modèle retombe sur le taux de fraude de base (1,42 %) et perd son "
    "signal le plus discriminant sur ce panier. Calculée sur les paniers conformes. "
    "Seuil : 3 %.",

"drift":
    "Nombre de variables dont l'indicateur dépasse le seuil de drift, sur les "
    "**45 calculées**.\n\n"
    "Le compteur porte volontairement sur les 45 et non sur les 8 affichées : sinon une "
    "dérive hors du tableau passerait totalement inaperçue.",

"couverture":
    "Les 4 granularités des tables produit (SKU, modèle, catégorie, marque) plus le "
    "produit dominant, face à des seuils **calibrés sur des niveaux mesurés** sur données "
    "neuves.\n\n"
    "Plus la cardinalité est basse, plus un inconnu est parlant, donc plus le seuil est "
    "serré : 135 catégories contre 14 296 SKU.\n\n"
    "À lire **en croisé** : le SKU seul peut exploser par simple renumérotation de "
    "références sans que le modèle se dégrade, mais trois granularités au-dessus du seuil "
    "en même temps signifient que l'assortiment a changé.",

"conf_courbe":
    "Un point par jour : somme des paniers conformes divisée par la somme des paniers "
    "reçus sur **tous les lots du jour**.",

"inconnus_courbe":
    "Trois taux, un par granularité :\n\n"
    "- `taux_sku_inconnus` : part des articles dont le code produit est absent des "
    "**14 296** références connues. Nouveauté constante, seuil 15 %.\n"
    "- `taux_make_inconnus` : part des articles dont la marque est absente des **808** "
    "marques connues. Seuil 2 %.\n"
    "- `taux_dom_inconnu` : part des **paniers** dont le bien le plus cher a un modèle "
    "inconnu. Seuil 3 %.\n\n"
    "Les deux premiers comptent des **articles**, le troisième des **paniers**. Les trois "
    "qui montent ensemble signalent un changement d'assortiment, pas une renumérotation.",

"suivies":
    "Les 8 variables de plus forte importance **SHAP** du modèle.\n\n"
    "**PSI** pour les distributions étalées : il compare la forme de la distribution "
    "tranche par tranche, et voit une réorganisation même si la moyenne ne bouge pas.\n\n"
    "**Écart de moyenne** pour les variables trop concentrées pour un découpage utile "
    "(binaires, comptages, ratios tassés). Une binaire est entièrement décrite par sa "
    "moyenne, un PSI n'apporterait rien.\n\n"
    "Le choix entre les deux est automatique.",

"jour_par_jour":
    "Le drift d'une variable au choix, jour par jour, sur les **45 calculées** et pas "
    "seulement les 8 suivies. C'est ce qui permet d'aller inspecter une variable le jour "
    "précis où elle dérive.",
}

COL_AIDE = {
"feature":    "Nom de la variable du modèle, telle qu'elle est calculée par le pipeline.",
"famille":    "Comment la variable a été profilée à l'entraînement : continue (déciles), "
              "concentrée (déciles effondrés), discrète (peu de valeurs), catégorielle (texte).",
"indicateur": "PSI ou écart de moyenne, choisi automatiquement selon la forme de la distribution.",
"valeur":     "PSI (sans unité, seuil 0,25), ou écart relatif de la moyenne par rapport à "
              "la référence (seuil 50 %, avec un plancher de 1 point en absolu).",
"cause":      "La tranche ou la modalité qui contribue le plus à l'écart, avec sa part du total. "
              "Un score seul ne dit pas ce qui a bougé.",
"statut":     "🟢 stable, 🟠 à surveiller, 🔴 drift. Seuils dans monitoring_config.json.",
"granularité":"Le niveau auquel on cherche le produit dans les tables figées.",
"inconnus":   "Part des articles (ou des paniers pour le produit dominant) absents des tables.",
"seuil":      "Seuil d'alerte, calibré sur le niveau naturellement observé sur données neuves.",
"état":       "🟢 sous la moitié du seuil, 🟠 entre la moitié et le seuil, 🔴 au-dessus.",
}


@st.cache_data
def charger_config():
    cfg = monitoring.load_config()
    prof = json.load(open(os.path.join(ROOT, "PROD", "artifacts", "v2",
                                       "reference_profile.json"), encoding="utf-8"))
    return cfg, prof


@st.cache_data
def charger(flux, jours):
    return monitoring.fenetre(flux, jours=jours)


cfg, prof = charger_config()
seuils = cfg["seuils_alerte"]
TOP    = cfg["features_affichees"]["top"]
MIN_N  = cfg["fenetre"]["min_lignes_psi"]

st.title("Monitoring de la donnée en entrée")
st.caption("Modèle **v2** · référence : profil figé à l'entraînement "
           f"({prof['n_ref']:,} paniers, {len(TOP)} variables suivies sur 45 calculées)")

jours = st.radio("Fenêtre d'analyse", [1, 3, 7], index=2, horizontal=True,
                 format_func=lambda j: f"{j} jour" + ("s" if j > 1 else ""),
                 help=AIDE["fenetre"])

live = charger("flux_a", jours)
kpi  = charger("kpi", jours)
if live.empty:
    st.warning("Base de contrôle vide. Lancer d'abord : `python PROD/simuler_controle.py`")
    st.stop()

rap = monitoring.rapport_drift(prof, seuils, live=live)
suivies = rap[rap.feature.isin(TOP)]
# le compteur d'alerte porte sur les 45 variables CALCULÉES, pas sur les 8 affichées :
# sinon une dérive sur une variable non affichée (p. ex. une chute des prix, que le
# top 8 ne contient pas) passerait totalement inaperçue.
en_drift  = rap.loc[rap.statut == "drift", "feature"].tolist()
n_drift   = len(en_drift)
hors_top  = [f for f in en_drift if f not in TOP]

# ── 1. indicateurs ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
taux_ech = cfg["flux_a"]["sample_rate_pct"]
c1.metric("Paniers reçus", f"{int(kpi['n_recus'].sum()):,}" if not kpi.empty else "n/a",
          delta=f"{len(live):,} échantillonnés ({taux_ech} %)", delta_color="off",
          help=AIDE["recus"])
if not kpi.empty:
    conf = kpi["n_conformes"].sum() / kpi["n_recus"].sum()
    c2.metric("Conformité", f"{conf:.2%}",
              delta=f"{(conf - seuils['taux_conformite_min'])*100:+.2f} pts vs seuil",
              help=AIDE["conformite"])
    dom = monitoring.taux_fenetre(kpi, "taux_dom_inconnu")
    c3.metric("Produit dominant inconnu", f"{dom:.2%}",
              delta=f"{(dom - seuils['taux_dom_inconnu_max'])*100:+.2f} pts vs seuil",
              delta_color="inverse", help=AIDE["dom"])
c4.metric("Variables en drift", f"{n_drift} / {len(rap)}", help=AIDE["drift"])

if hors_top:
    st.error("**Drift hors des variables affichées** : " + ", ".join(f"`{f}`" for f in hors_top)
             + ". Visibles dans « les 45 variables calculées » ci-dessous.")

if len(live) < MIN_N:
    st.info(f"Échantillon de contrôle : **{len(live):,} lignes** ({taux_ech} % des "
            f"{int(kpi['n_recus'].sum()):,} paniers reçus), en dessous des {MIN_N:,} "
            "recommandées. Le PSI est peu fiable sur si peu de volume : élargir la fenêtre.")

# ── 2. couverture du catalogue ─────────────────────────────────────────────
# Les 5 seuils de couverture sont calibrés dans monitoring_config.json : ils doivent
# TOUS être confrontés à la mesure, sinon un dépassement passe inaperçu. Et c'est la
# lecture CROISÉE qui porte le signal : le SKU seul peut exploser par simple
# renumérotation de références sans que le modèle se dégrade, mais SKU + modèle +
# marque ensemble, c'est l'assortiment du retailer qui a changé.
ETAT = {"dépassé": "🔴", "proche": "🟠", "normal": "🟢"}

if not kpi.empty:
    st.subheader("Couverture du catalogue", help=AIDE["couverture"])
    couv, n_depasses = monitoring.couverture_fenetre(seuils, kpi=kpi)
    aff_couv = couv.assign(inconnus=couv["inconnus"].map("{:.2%}".format),
                           seuil=couv["seuil"].map("{:.2%}".format),
                           **{"état": couv["état"].map(lambda e: f"{ETAT[e]} {e}")})
    st.dataframe(aff_couv, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.TextColumn(c, help=COL_AIDE[c])
                                for c in aff_couv.columns})

    # pondéré par le nombre de paniers. Le poids exact serait la valeur totale du lot,
    # qui n'est pas archivée : c'est donc une approximation, suffisante pour un ordre
    # de grandeur affiché à côté d'une alerte.
    val_inc = monitoring.taux_fenetre(kpi, "part_valeur_inconnue")
    detail = f"**{val_inc:.1%}** de la valeur financée porte sur des produits inconnus du modèle."
    if n_depasses >= 3:
        st.error(f"**{n_depasses} granularités sur 5 au-dessus du seuil.** Ce n'est pas une "
                 f"renumérotation de références, c'est l'assortiment qui a changé. {detail} "
                 "Réentraînement à programmer.")
    elif n_depasses:
        st.warning(f"{n_depasses} granularité sur 5 au-dessus du seuil. {detail}")
    else:
        st.caption(f"Les 5 granularités sont sous leur seuil. {detail}")

# ── 3. évolution jour par jour ──────────────────────────────────────────────
if not kpi.empty:
    st.subheader("Évolution jour par jour")
    # plusieurs lots peuvent arriver le même jour (chaque appel à serve écrit une ligne
    # kpi) : on agrège par jour. Les compteurs se somment, les taux se repondèrent par
    # le volume du lot. Sans ça, deux lots produiraient deux points à la même date, et
    # un lot de 50 paniers pèserait autant qu'un lot de 3 000.
    # la conformité se pondère par n_recus, les taux de couverture par n_paniers :
    # les rejets n'ont pas de produits à confronter aux tables.
    TAUX = ["taux_sku_inconnus", "taux_make_inconnus", "taux_dom_inconnu"]
    g = kpi.copy()
    for c in TAUX:
        g[c] = g[c] * g["n_paniers"]
    k = g.groupby("date")[["n_recus", "n_conformes", "n_paniers"] + TAUX].sum().sort_index()
    k["taux_conformite"] = k["n_conformes"] / k["n_recus"]
    for c in TAUX:
        k[c] = k[c] / k["n_paniers"]

    g1, g2 = st.columns(2)
    with g1:
        st.caption("Taux de conformité", help=AIDE["conf_courbe"])
        st.line_chart(k[["taux_conformite"]], height=200)
    with g2:
        st.caption("Produits inconnus (fraîcheur du catalogue)", help=AIDE["inconnus_courbe"])
        st.line_chart(k[["taux_sku_inconnus", "taux_make_inconnus", "taux_dom_inconnu"]], height=200)

# ── 4. drift des variables suivies ─────────────────────────────────────────
st.subheader(f"Variables suivies, fenêtre {jours} jour" + ("s" if jours > 1 else ""),
             help=AIDE["suivies"])
aff = suivies.copy()
aff["statut"] = aff["statut"].map(lambda s: f"{PUCE[s]} {s}")
aff["valeur"] = aff.apply(
    lambda r: f"{r['valeur']:.3f}" if r["indicateur"] == "PSI" else f"{r['valeur']:+.1%}", axis=1)
COLS = ["feature", "indicateur", "valeur", "cause", "statut"]
st.dataframe(aff[COLS], hide_index=True, use_container_width=True,
             column_config={
                 "cause": st.column_config.TextColumn("qu'est-ce qui a bougé",
                                                      width="large", help=COL_AIDE["cause"]),
                 **{c: st.column_config.TextColumn(c, help=COL_AIDE[c])
                    for c in COLS if c != "cause"}})
st.caption("PSI : <0,10 stable · 0,10–0,25 à surveiller · >0,25 drift  |  "
           "écart de moyenne : écart relatif, avec plancher de 1 point")

with st.expander(f"Voir les {len(rap)} variables calculées ({n_drift} en drift)"):
    tout = rap.copy()
    tout["statut"] = tout["statut"].map(lambda s: f"{PUCE[s]} {s}")
    COLS_T = ["feature", "famille", "indicateur", "valeur", "cause", "statut"]
    st.dataframe(tout[COLS_T], hide_index=True, use_container_width=True,
                 column_config={
                     "cause": st.column_config.TextColumn("qu'est-ce qui a bougé",
                                                          width="large", help=COL_AIDE["cause"]),
                     **{c: st.column_config.TextColumn(c, help=COL_AIDE[c])
                        for c in COLS_T if c != "cause"}})

# ── 5. drift jour par jour d'une variable ──────────────────────────────────
st.subheader("Drift jour par jour", help=AIDE["jour_par_jour"])
st.caption("Une dérive d'une seule journée se **dilue** dans une fenêtre large : "
           "la vue par jour la fait ressortir.")
# toutes les variables sont proposées (les 8 suivies en tête) : sans ça, impossible
# d'aller inspecter une variable non affichée le jour précis où elle dérive.
options = TOP + [f for f in rap.feature.tolist() if f not in TOP]
choix = st.selectbox("Variable", options, index=0,
                     format_func=lambda f: f + ("  ·  suivie" if f in TOP else ""),
                     help="Les 8 variables suivies sont en tête, les 37 autres suivent. "
                          "Une variable en drift signalée par la bannière rouge se trouve ici.")

par_jour = []
for d in sorted(monitoring.fenetre("flux_a", jours=7)["ts"].str[:10].unique()):
    w = monitoring.fenetre("flux_a", depuis=d)
    w = w[w["ts"].str[:10] == d]
    r = monitoring.rapport_drift(prof, seuils, live=w)
    ligne = r[r.feature == choix]
    if not ligne.empty:
        par_jour.append({"date": d, "valeur": float(ligne["valeur"].iloc[0]),
                         "statut": ligne["statut"].iloc[0]})
pj = pd.DataFrame(par_jour)
if not pj.empty:
    ind = rap.loc[rap.feature == choix, "indicateur"].iloc[0]
    st.bar_chart(pj.set_index("date")[["valeur"]], height=240,
                 color=[COULEUR[pj.statut.iloc[-1]]])
    st.caption(f"`{choix}`, indicateur : **{ind}**. "
               + ("Seuil de drift : 0,25." if ind == "PSI" else "Seuil de drift : ±50 % en relatif."))
