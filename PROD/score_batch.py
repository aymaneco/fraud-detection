"""Scoring par lot : un CSV de paniers -> un CSV de probabilités (format soumission).

Point d'entrée mince : il ne fait que lire un fichier, appeler `inference.serve` et
écrire le résultat. Tout le reste (portail, score, échantillonnage, journalisation
dans la base de contrôle) appartient à `serve` et vaut donc pour n'importe quel canal.

Un panier non conforme n'est pas scoré. Il reçoit malgré tout une ligne en sortie,
à 0, pour que le fichier produit couvre exactement les ID reçus.

Usage :  python PROD/score_batch.py <input.csv> <output.csv> [version=v2]
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "PROD", "pipeline"))

import pandas as pd
from inference import load_artifacts, serve


def main(inp, outp, version="v2"):
    art = load_artifacts(version)
    raw = pd.read_csv(inp, low_memory=False)

    out, kpi = serve(raw, art)

    sub = pd.DataFrame({"ID": raw["ID"].to_numpy()})
    sub["fraud_flag"] = sub["ID"].map(dict(zip(out.ID, out.fraud_proba))).fillna(0.0)
    sub.to_csv(outp)                              # colonne d'index -> format soumission

    n_rejetes = kpi["n_recus"] - kpi["n_conformes"]
    n_synth   = int((raw.get("is_synthetic", pd.Series([], dtype=int)) == 1).sum())
    print(f"conformité : {kpi['taux_conformite']:.2%} ({n_rejetes:,} rejeté(s))")
    print(f"scoré {kpi['n_conformes']:,} paniers sur {kpi['n_recus']:,} reçus "
          f"(dont {n_synth} synthétiques à 0) -> {outp}")
    print(f"catalogue : SKU inconnus {kpi['taux_sku_inconnus']:.2%} | "
          f"marques {kpi['taux_make_inconnus']:.2%} | dominant {kpi['taux_dom_inconnu']:.2%}")
    print(f"proba : min {out.fraud_proba.min():.4f} | médiane {out.fraud_proba.median():.4f} "
          f"| max {out.fraud_proba.max():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python PROD/score_batch.py <input.csv> <output.csv> [version]"); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "v2")
