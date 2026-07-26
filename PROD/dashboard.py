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
                 format_func=lambda j: f"{j} jour" + ("s" if j > 1 else ""))

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
          help="Les indicateurs de conformité et de couverture portent sur TOUS les paniers reçus. "
               "Le drift (PSI) est calculé sur l'échantillon de contrôle uniforme.")
if not kpi.empty:
    conf = kpi["n_conformes"].sum() / kpi["n_recus"].sum()
    c2.metric("Conformité", f"{conf:.2%}",
              delta=f"{(conf - seuils['taux_conformite_min'])*100:+.2f} pts vs seuil")
    dom = monitoring.taux_fenetre(kpi, "taux_dom_inconnu")
    c3.metric("Produit dominant inconnu", f"{dom:.2%}",
              delta=f"{(dom - seuils['taux_dom_inconnu_max'])*100:+.2f} pts vs seuil",
              delta_color="inverse")
c4.metric("Variables en drift", f"{n_drift} / {len(rap)}")

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
    st.subheader("Couverture du catalogue")
    couv, n_depasses = monitoring.couverture_fenetre(seuils, kpi=kpi)
    aff_couv = couv.assign(inconnus=couv["inconnus"].map("{:.2%}".format),
                           seuil=couv["seuil"].map("{:.2%}".format),
                           **{"état": couv["état"].map(lambda e: f"{ETAT[e]} {e}")})
    st.dataframe(aff_couv, hide_index=True, use_container_width=True)

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
        st.caption("Taux de conformité")
        st.line_chart(k[["taux_conformite"]], height=200)
    with g2:
        st.caption("Produits inconnus (fraîcheur du catalogue)")
        st.line_chart(k[["taux_sku_inconnus", "taux_make_inconnus", "taux_dom_inconnu"]], height=200)

# ── 4. drift des variables suivies ─────────────────────────────────────────
st.subheader(f"Variables suivies, fenêtre {jours} jour" + ("s" if jours > 1 else ""))
aff = suivies.copy()
aff["statut"] = aff["statut"].map(lambda s: f"{PUCE[s]} {s}")
aff["valeur"] = aff.apply(
    lambda r: f"{r['valeur']:.3f}" if r["indicateur"] == "PSI" else f"{r['valeur']:+.1%}", axis=1)
st.dataframe(aff[["feature", "indicateur", "valeur", "cause", "statut"]],
             hide_index=True, use_container_width=True,
             column_config={"cause": st.column_config.TextColumn(
                 "qu'est-ce qui a bougé", width="large")})
st.caption("PSI : <0,10 stable · 0,10–0,25 à surveiller · >0,25 drift  |  "
           "écart de moyenne : écart relatif, avec plancher de 1 point")

with st.expander(f"Voir les {len(rap)} variables calculées ({n_drift} en drift)"):
    tout = rap.copy()
    tout["statut"] = tout["statut"].map(lambda s: f"{PUCE[s]} {s}")
    st.dataframe(tout[["feature", "famille", "indicateur", "valeur", "cause", "statut"]],
                 hide_index=True, use_container_width=True,
                 column_config={"cause": st.column_config.TextColumn(
                     "qu'est-ce qui a bougé", width="large")})

# ── 5. drift jour par jour d'une variable ──────────────────────────────────
st.subheader("Drift jour par jour")
st.caption("Une dérive d'une seule journée se **dilue** dans une fenêtre large : "
           "la vue par jour la fait ressortir.")
# toutes les variables sont proposées (les 8 suivies en tête) : sans ça, impossible
# d'aller inspecter une variable non affichée le jour précis où elle dérive.
options = TOP + [f for f in rap.feature.tolist() if f not in TOP]
choix = st.selectbox("Variable", options, index=0,
                     format_func=lambda f: f + ("  ·  suivie" if f in TOP else ""))

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
