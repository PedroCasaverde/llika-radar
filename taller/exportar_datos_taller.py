"""Arma las tablas livianas del taller (una fila por proveedor) a partir de los datos procesados del radar.

Se corre una sola vez, en la máquina que tiene los datos completos:

    python taller/exportar_datos_taller.py --origen RUTA/datos/procesados/radar

Las variables se calculan igual que en radar/12_evaluacion_e5.py (función ventana). El RUC del proveedor se cambia por
un número correlativo, en el mismo orden, para que los empates se resuelvan igual que en el script original. Salen:
- aprendizaje.parquet: alertas 2018-2021, etiqueta = sanción que empieza en 2022-2023
- examen.parquet:      alertas 2022-2023, etiqueta = sanción desde 2024 (el conjunto congelado del E5)
- examen_entidades.parquet: pares proveedor-entidad del examen, para medir la carga por entidad
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]

ap = argparse.ArgumentParser()
ap.add_argument("--origen", required=True, help="carpeta datos/procesados/radar con los parquet completos")
ap.add_argument("--destino", default=str(Path(__file__).parent / "datos"))
args = ap.parse_args()
R, D = Path(args.origen), Path(args.destino)
D.mkdir(parents=True, exist_ok=True)

o = pd.read_parquet(R / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto", "fecha"])
al = pd.read_parquet(R / "radar_alertas.parquet", columns=["id", "ruc", "anio", "tarjeta", "puntos", "fecha"])
s = pd.read_parquet(R / "sanciones.parquet", columns=["ruc", "inicio", "motivo"])
esp = pd.read_parquet(R / "alerta_empresa_espejo.parquet", columns=["id", "ruc_sanc"])
ant = pd.read_parquet(R / "antecedentes.parquet")
print("datos leídos", flush=True)


def ventana(desde, hasta, etq_desde, etq_hasta=None):
    # copia de ventana() en radar/12_evaluacion_e5.py
    fin = pd.Timestamp(f"{hasta}-12-31")
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta)]
    X = v.groupby("ruc").agg(monto=("monto", "sum"), ordenes=("id", "size"), entidades=("ruc_entidad", "nunique"))
    X["log_monto"], X["log_ordenes"] = np.log1p(X["monto"]), np.log1p(X["ordenes"])
    X["persona"] = X.index.str.startswith("10").astype(int)
    aa = al[(al["anio"] >= desde) & (al["anio"] <= hasta) & al["tarjeta"].isin(CARTAS)]
    piv = aa.pivot_table(index="ruc", columns="tarjeta", values="puntos", aggfunc="max")
    for t in CARTAS:
        X[t] = piv[t].reindex(X.index).fillna(0) if t in piv else 0.0
    a = ant[ant["fecha"] <= fin]
    for tipo, col in [("multa", "multas"), ("penalidad", "penalidades")]:
        X[col] = a[a["tipo"].eq(tipo)].groupby("ruc").size().reindex(X.index).fillna(0)
        X["log_" + col] = np.log1p(X[col])
    X["antecedentes"] = X["multas"] + X["penalidades"]
    primera = o[o["fecha"] <= fin].groupby("ruc")["fecha"].min()
    e = aa[aa["tarjeta"].eq("empresa_espejo")][["id", "ruc", "fecha"]].merge(esp, on="id")
    ini = e.merge(s[["ruc", "inicio"]].rename(columns={"ruc": "ruc_sanc"}), on="ruc_sanc")
    ini = ini[ini["inicio"] <= ini["fecha"]].groupby(["id", "ruc"])["inicio"].max().reset_index()
    ini["primera"] = ini["ruc"].map(primera)
    ev = ini[(ini["primera"] >= ini["inicio"]) & (ini["primera"] >= pd.Timestamp("2018-07-01"))]
    X["evasion"] = X.index.isin(set(ev["ruc"])).astype(int)
    X["reciente"] = X.index.isin(set(aa.loc[aa["anio"].eq(hasta), "ruc"])).astype(int)
    corte = pd.Timestamp(f"{etq_desde}-01-01")
    X = X[~X.index.isin(set(s.loc[s["inicio"] < corte, "ruc"]))]
    lab = s["inicio"] >= corte
    if etq_hasta is not None:
        lab &= s["inicio"] < pd.Timestamp(f"{etq_hasta}-01-01")
    X["y"] = X.index.isin(set(s.loc[lab, "ruc"])).astype(int)
    return X


def categoria(m):
    m = str(m).lower()
    if "impedido" in m:
        return "impedimento"
    if "fals" in m or "inexact" in m:
        return "falsedad"
    if "resuelva el contrato" in m or "perfeccionar" in m or "mantener" in m or "incumpl" in m:
        return "incumplimiento"
    return "otros"


def anonimo(X):
    # el orden de filas se conserva tal cual: así los empates salen igual que en el E5
    T = X.reset_index(drop=True)
    T.insert(0, "id", np.arange(len(T), dtype=np.int32))
    for c in T.columns:
        if T[c].dtype == "float64" and c not in ("monto", "log_monto", "log_ordenes", "log_multas", "log_penalidades"):
            T[c] = T[c].astype("float32")
    return T


Xtr = ventana(2018, 2021, 2022, 2024)
Xte = ventana(2022, 2023, 2024)

mot = s[s["inicio"] >= "2024-01-01"].sort_values("inicio").drop_duplicates("ruc").set_index("ruc")["motivo"].map(categoria)
Xte["motivo"] = np.where(Xte["y"].eq(1), Xte.index.map(lambda r: mot.get(r, "otros")), None)

ids = pd.Series(np.arange(len(Xte), dtype=np.int32), index=Xte.index)
pares = o[(o["anio"] >= 2022) & (o["anio"] <= 2023) & o["ruc"].isin(ids.index)][["ruc", "ruc_entidad"]].drop_duplicates()
ent = {r: i for i, r in enumerate(sorted(pares["ruc_entidad"].unique()))}
pares = pd.DataFrame({"id": pares["ruc"].map(ids).astype("int32"),
                      "entidad": pares["ruc_entidad"].map(ent).astype("int32")})

anonimo(Xtr).to_parquet(D / "aprendizaje.parquet", index=False)
anonimo(Xte).to_parquet(D / "examen.parquet", index=False)
pares.to_parquet(D / "examen_entidades.parquet", index=False)

huellas = {}
for f in ("aprendizaje.parquet", "examen.parquet", "examen_entidades.parquet"):
    p = D / f
    huellas[f] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "MB": round(p.stat().st_size / 1e6, 1)}
resumen = {"aprendizaje": {"n": int(len(Xtr)), "con_sancion": int(Xtr["y"].sum())},
           "examen": {"n": int(len(Xte)), "con_sancion": int(Xte["y"].sum())},
           "pares_examen": int(len(pares)), "archivos": huellas}
(D / "huellas.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(resumen, ensure_ascii=False, indent=1))
