"""Paso 4: índices por entidad, con encogimiento bayesiano (Beta-Binomial, Bayes empírico).

- Recambio electoral: de las personas con órdenes de servicio de la entidad entre julio y diciembre, qué parte sigue
  entre enero y junio del año siguiente. Se compara el año en que entra la autoridad (2019 y 2023) con los demás.
- Familiares: probabilidad esperada de parientes (suma de las probabilidades del paso 2) entre las personas que contrata
  la entidad en el año.
Las tasas de entidades con pocos contratados son ruidosas: se encogen hacia la media de su tipo de entidad, con un
previo Beta estimado por momentos. Prueba de que ayuda: el índice de un mismo lugar debe repetirse entre dos periodos
distintos (2019 frente a 2023, o 2023 frente a 2024) más con encogimiento que sin él. Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 04_indices.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, entidades  # noqa: E402


def encoger(k, n):
    """Media posterior Beta-Binomial con previo estimado por momentos (k puede no ser entero)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    mu = k.sum() / n.sum()
    r = k / n
    var = np.average((r - mu) ** 2, weights=n)
    ruido = mu * (1 - mu) * np.mean(1 / n)
    var_real = max(var - ruido, 1e-6)
    phi = float(np.clip(mu * (1 - mu) / var_real - 1, 1, 1e4))
    return (k + mu * phi) / (n + phi), phi


def correlacion(a, b):
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan


ent = entidades().set_index("RUC")["NOMBRE_DE_ENTIDAD"].str.upper()
tipo = pd.Series(np.select([ent.str.contains("MUNICIPALIDAD", na=False),
                            ent.str.contains("GOBIERNO REGIONAL|REGION ", na=False, regex=True)],
                           ["municipalidad", "gobierno regional"], "otra entidad"), index=ent.index)

o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "anio", "servicio"])
p = o[o["ruc"].str.startswith("10") & o["servicio"]].copy()
p["sem"] = (p["fecha"].dt.month > 6).astype(int) + 1
act = p[["ruc_entidad", "ruc", "anio", "sem"]].drop_duplicates()

# ---------------------------------------------------------------- recambio electoral
filas = []
for y in range(2019, 2026):
    antes = act[(act["anio"] == y - 1) & (act["sem"] == 2)][["ruc_entidad", "ruc"]]
    despues = act[(act["anio"] == y) & (act["sem"] == 1)][["ruc_entidad", "ruc"]].assign(sigue=1)
    j = antes.merge(despues, on=["ruc_entidad", "ruc"], how="left").fillna({"sigue": 0})
    g = j.groupby("ruc_entidad")["sigue"].agg(k="sum", n="size").reset_index().assign(anio=y)
    filas.append(g)
r = pd.concat(filas, ignore_index=True)
r["tipo"] = r["ruc_entidad"].map(tipo).fillna("otra entidad")
r = r[r["n"] >= 1].reset_index(drop=True)
r["tasa"] = r["k"] / r["n"]
r["tasa_enc"] = np.nan
for (y, t), g in r.groupby(["anio", "tipo"]):
    r.loc[g.index, "tasa_enc"] = encoger(g["k"], g["n"])[0]
normal = r[~r["anio"].isin([2019, 2023])].groupby("ruc_entidad")
base_raw = normal.apply(lambda g: np.average(g["tasa"], weights=g["n"]), include_groups=False)
base_enc = normal.apply(lambda g: np.average(g["tasa_enc"], weights=g["n"]), include_groups=False)
idx = pd.DataFrame({"tipo": r.groupby("ruc_entidad")["tipo"].first()})
for y in (2019, 2023):
    ry = r[r["anio"] == y].set_index("ruc_entidad")
    idx[f"caida_{y}"] = base_raw - ry["tasa"]
    idx[f"caida_{y}_enc"] = base_enc - ry["tasa_enc"]
    idx[f"n_{y}"] = ry["n"]
print("RECAMBIO ELECTORAL: caída de la retención de contratados cuando entra la autoridad (puntos)")
for t, g in idx.groupby("tipo"):
    print(f"  {t:<18} entidades {len(g):,}: caída mediana 2019 {g['caida_2019_enc'].median() * 100:+.1f}, "
          f"2023 {g['caida_2023_enc'].median() * 100:+.1f}")
mun = idx[idx["tipo"] == "municipalidad"]
print(f"  ¿la misma municipalidad cae en las dos elecciones? correlación 2019-2023: sin encoger "
      f"{correlacion(mun['caida_2019'], mun['caida_2023']):.3f}, encogida {correlacion(mun['caida_2019_enc'], mun['caida_2023_enc']):.3f}")

# ---------------------------------------------------------------- familiares por entidad
fa = pd.read_parquet(SALIDA / "alerta_familiar.parquet").merge(o[["id", "ruc_entidad", "ruc", "anio"]], on="id")
k = fa.groupby(["ruc_entidad", "anio", "ruc"])["prob"].max().groupby(level=[0, 1]).sum().rename("k")
n = p.groupby(["ruc_entidad", "anio"])["ruc"].nunique().rename("n")
fam = pd.concat([n, k], axis=1).fillna({"k": 0}).reset_index()
fam = fam[fam["n"] >= 1].reset_index(drop=True)
fam["tipo"] = fam["ruc_entidad"].map(tipo).fillna("otra entidad")
fam["tasa"] = fam["k"] / fam["n"]
fam["tasa_enc"] = np.nan
for (y, t), g in fam.groupby(["anio", "tipo"]):
    fam.loc[g.index, "tasa_enc"] = encoger(g["k"], g["n"])[0]
a23, a24 = fam[fam["anio"] == 2023].set_index("ruc_entidad"), fam[fam["anio"] == 2024].set_index("ruc_entidad")
print("\nFAMILIARES POR ENTIDAD: parientes esperados por cada 100 contratados")
print(f"  ¿la misma entidad repite de 2023 a 2024? correlación: sin encoger {correlacion(a23['tasa'], a24['tasa']):.3f}, "
      f"encogida {correlacion(a23['tasa_enc'], a24['tasa_enc']):.3f}")
top_raw = set(a24["tasa"].nlargest(50).index)
top_enc = set(a24["tasa_enc"].nlargest(50).index)
print(f"  las 50 primeras de 2024 tienen en mediana {a24.loc[list(top_raw), 'n'].median():.0f} contratados sin encoger y "
      f"{a24.loc[list(top_enc), 'n'].median():.0f} encogidas")

idx = idx.reset_index()
fam.to_parquet(SALIDA / "indice_familiares.parquet", index=False)
idx.to_parquet(SALIDA / "indice_recambio.parquet", index=False)
