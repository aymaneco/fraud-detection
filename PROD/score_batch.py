"""Scoring par lot : un CSV de paniers -> un CSV de probabilités (format soumission).

Même enchaînement qu'en temps réel : PORTAIL (contrat Pydantic) puis modèle. Un panier
non conforme est journalisé dans `control_base/erreurs/` avec sa raison et ne passe PAS
au modèle. Il reçoit malgré tout une ligne en sortie, à 0, pour que le fichier produit
couvre exactement les ID reçus.

Usage :  python PROD/score_batch.py <input.csv> <output.csv> [version=v2]
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "PROD", "pipeline"))

import pandas as pd
from inference import load_artifacts, predict
from monitoring import check_schema, conformity_report
import store


def main(inp, outp, version="v2"):
    art = load_artifacts(version)
    raw = pd.read_csv(inp, low_memory=False)

    # 1) portail : on valide AVANT de scorer, et on journalise les rejets
    sc  = check_schema(raw)
    rep = conformity_report(raw, sc)
    conf = sc["conforme"].to_numpy()
    if rep["non_conformes"]:
        # même schéma que le flux B temps réel (ID, raisons, raison_b) : la partition
        # erreurs/ se lit comme une seule table quelle que soit sa provenance.
        store.append(pd.DataFrame({"ID": raw.loc[~conf, "ID"].to_numpy(),
                                   "raisons": sc.loc[~conf, "raisons"].to_numpy(),
                                   "raison_b": "non_conforme"}), "erreurs")

    # 2) modèle, sur les seuls paniers conformes
    res = predict(raw[conf].reset_index(drop=True), art)

    # 3) sortie sur TOUS les ID reçus : rejeté ou synthétique -> 0
    sub = pd.DataFrame({"ID": raw["ID"].to_numpy()})
    sub["fraud_flag"] = sub["ID"].map(dict(zip(res.ID, res.fraud_proba))).fillna(0.0)
    sub.to_csv(outp)                              # colonne d'index -> format soumission

    n_synth = int((raw.get("is_synthetic", pd.Series([], dtype=int)) == 1).sum())
    print(f"conformité : {rep['taux_conformite']:.2%} ({rep['non_conformes']:,} rejeté(s))"
          + (f" -> {rep['raisons']}" if rep["raisons"] else ""))
    print(f"scoré {len(res):,} paniers sur {len(sub):,} reçus "
          f"(dont {n_synth} synthétiques à 0) -> {outp}")
    print(f"proba : min {res.fraud_proba.min():.4f} | médiane {res.fraud_proba.median():.4f} "
          f"| max {res.fraud_proba.max():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python PROD/score_batch.py <input.csv> <output.csv> [version]"); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "v2")
