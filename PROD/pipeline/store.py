"""Base de contrôle : où atterrissent les paniers échantillonnés (flux A) et les rejets.

Stockage Parquet **partitionné par jour** : c'est ce qu'on ferait en production
(S3/HDFS lus par Spark/DuckDB), et le job batch n'a qu'à charger les partitions de
sa fenêtre au lieu de tout relire.

    control_base/
      flux_a/date=2026-07-25/part-<horodatage>.parquet   <- features + score (PSI, drift)
      erreurs/date=2026-07-25/part-<horodatage>.parquet  <- paniers rejetés + raison
      kpi/date=2026-07-25/part-<horodatage>.parquet      <- agrégats du jour (conformité, couverture)

Les KPI sont archivés parce qu'ils portent sur TOUS les paniers reçus, pas seulement
sur l'échantillon de contrôle : ils ne sont donc pas recalculables depuis `flux_a`.

On écrit en append (un fichier par lot), jamais de réécriture : c'est immuable,
concurrent-safe, et un incident reste tracé.
"""
import os, glob
from datetime import datetime, timezone
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "control_base")
FLUX = ("flux_a", "erreurs", "kpi")     # les trois partitions de la base


def _part_dir(flux, jour):
    return os.path.join(BASE, flux, f"date={jour}")


def _now():
    return datetime.now(timezone.utc)


def append(df, flux="flux_a", ts=None):
    """Ajoute un lot à la partition du jour. Renvoie le chemin écrit (ou None si vide)."""
    if df is None or len(df) == 0:
        return None
    ts = ts or _now()
    df = df.copy()
    df["ts"] = ts.isoformat()
    d = _part_dir(flux, ts.strftime("%Y-%m-%d"))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"part-{ts.strftime('%Y%m%dT%H%M%S%f')}.parquet")
    df.to_parquet(path, index=False)
    return path


def read_window(flux="flux_a", jours=None, depuis=None):
    """Lit une fenêtre de la base.
    jours  : nb de jours en arrière depuis aujourd'hui (fenêtre glissante)
    depuis : date 'YYYY-MM-DD' incluse (prioritaire sur `jours`)
    """
    parts = sorted(glob.glob(os.path.join(BASE, flux, "date=*", "*.parquet")))
    if depuis is None and jours is not None:
        depuis = (_now() - pd.Timedelta(days=jours - 1)).strftime("%Y-%m-%d")
    if depuis:
        parts = [p for p in parts if _jour_de(p) >= depuis]
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _jour_de(path):
    for seg in path.replace("\\", "/").split("/"):
        if seg.startswith("date="):
            return seg[5:]
    return ""


def stats():
    "inventaire de la base : partitions et volumes par flux."
    out = {}
    for flux in FLUX:
        parts = glob.glob(os.path.join(BASE, flux, "date=*", "*.parquet"))
        jours = sorted({_jour_de(p) for p in parts})
        n = sum(pd.read_parquet(p, columns=["ts"]).shape[0] for p in parts) if parts else 0
        mo = sum(os.path.getsize(p) for p in parts) / 1e6
        out[flux] = {"partitions": len(jours), "jours": jours, "lignes": int(n), "taille_Mo": round(mo, 2)}
    return out


def purge(retention_jours):
    "supprime les partitions plus vieilles que la rétention (RGPD / coût de stockage)."
    limite = (_now() - pd.Timedelta(days=retention_jours)).strftime("%Y-%m-%d")
    supprimes = []
    for flux in FLUX:
        for d in glob.glob(os.path.join(BASE, flux, "date=*")):
            if _jour_de(d) < limite:
                for f in glob.glob(os.path.join(d, "*")):
                    os.remove(f)
                os.rmdir(d); supprimes.append(d)
    return supprimes
