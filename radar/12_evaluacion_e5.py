"""Paso 12: evaluación del E5. Baselines contra el modelo v0, con rango por bootstrap y error por segmento.

El modelo v0 es el que ganó en las rondas de mejora (HALLAZGOS 50 a 53): una regresión logística L1 con los puntos de
cada alerta del radar y los antecedentes del proveedor (multas del Tribunal y penalidades de contratos, paso 14), más
evasión y recencia. Se compara con:
- no hacer nada (1 000 proveedores al azar);
- lo que se hace hoy (lo más caro primero);
- un modelo simple (regresión logística con 4 variables de tamaño);
- la regla del radar (puntaje legal por puntos, solo con las alertas de vínculo; empate por monto).

Ventanas sin fuga. Los modelos se entrenan con alertas de 2018-2021 y sanciones que empiezan en 2022-2023; la prueba
usa alertas de 2022-2023 y sanciones desde 2024, sin quien ya tenía una sanción antes. La confirmación se reentrena con
alertas de 2020-2023 y sanciones de 2024-2025, y se prueba con alertas de 2024-2025 y sanciones de 2026.

- Métrica principal: de los 1 000 primeros proveedores de cada lista, el porcentaje con una sanción posterior.
- Contrapeso: la carga para el OCI, cuántos proveedores de la lista le tocan a cada entidad (mediana y percentil 90).
- Rango: intervalo del 95 % por bootstrap de proveedores (pesos de Poisson, 1 000 repeticiones), pareado entre listas.

Solo imprime agregados y los guarda en datos/procesados/radar/evaluacion_e5.json.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 -W ignore 12_evaluacion_e5.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, wilson  # noqa: E402

t0 = time.time()
K, B = 1000, 1000
CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
ANTEC = ["log_multas", "log_penalidades", "evasion", "reciente"]
TAM = ["log_monto", "log_ordenes", "entidades", "persona"]
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto", "fecha"])
al = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["id", "ruc", "anio", "tarjeta", "puntos", "fecha"])
s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio", "motivo"])
esp = pd.read_parquet(SALIDA / "alerta_empresa_espejo.parquet", columns=["id", "ruc_sanc"])
ant = pd.read_parquet(SALIDA / "antecedentes.parquet")
print(f"datos leídos en {time.time() - t0:.0f} s", flush=True)


def ventana(desde, hasta, etq_desde, etq_hasta=None):
    """Una fila por proveedor (como en el paso 15); las variables solo usan información hasta el 31/12 de `hasta`."""
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
    y = X.index.isin(set(s.loc[lab, "ruc"])).astype(int)
    return X, y


def entrenar(X, y, cols, l1=True):
    """Todos los positivos y 100 000 negativos al azar; con L1, el C se elige por validación cruzada."""
    rng = np.random.default_rng(7)
    neg = np.flatnonzero(y == 0)
    fil = np.sort(np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))
    sc = StandardScaler().fit(X.iloc[fil][cols])
    if l1:
        g = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000), {"C": [0.001, 0.01, 0.1, 1.0]},
                         cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc")
        m = g.fit(sc.transform(X.iloc[fil][cols]), y[fil]).best_estimator_
    else:
        m = LogisticRegression(max_iter=500).fit(sc.transform(X.iloc[fil][cols]), y[fil])
    coef = {c: round(float(v), 3) for c, v in zip(cols, m.coef_[0])}
    return (lambda Z: m.predict_proba(sc.transform(Z[cols]))[:, 1]), coef


def orden(*claves):
    return np.lexsort(tuple(-np.asarray(c, dtype=float) for c in reversed(claves)))


def carga(X, idx, desde, hasta):
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta) & o["ruc"].isin(set(X.index[idx]))]
    por_ent = v.drop_duplicates(["ruc_entidad", "ruc"]).groupby("ruc_entidad").size()
    return {"entidades": int(len(por_ent)), "mediana": float(por_ent.median()), "p90": float(por_ent.quantile(0.9))}


def bootstrap(y, ordenes, rng, k=K, b=B, largo=4 * K):
    res = {n: np.empty(b) for n in ordenes}
    for r in range(b):
        w = rng.poisson(1.0, size=len(y))
        for n, idx in ordenes.items():
            ww = w[idx[:largo]]
            cw = np.cumsum(ww)
            j = int(np.searchsorted(cw, k))
            res[n][r] = (ww[: j + 1] * y[idx[: j + 1]]).sum() / cw[j]
    return res


def ic(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def categoria(m):
    m = str(m).lower()
    if "impedido" in m:
        return "impedimento"
    if "fals" in m or "inexact" in m:
        return "falsedad"
    if "resuelva el contrato" in m or "perfeccionar" in m or "mantener" in m or "incumpl" in m:
        return "incumplimiento"
    return "otros"


def listas_de(X, f_tam, f_v0):
    p = X[CARTAS].sum(axis=1).to_numpy()
    m = X["monto"].to_numpy()
    return {"hoy": orden(m), "simple": orden(f_tam(X), m), "regla": orden(p, m), "v0": orden(f_v0(X), m)}


def evaluar(X, y, listas, desde, hasta, semilla, cat):
    bs = bootstrap(y, listas, np.random.default_rng(semilla))
    base = float(y.mean())
    azar = np.random.default_rng(semilla + 1).binomial(K, base, size=10_000) / K
    filas = {"azar": {"precision": base, "rango": ic(azar)}}
    for n, idx in listas.items():
        top = idx[:K]
        k = int(y[top].sum())
        filas[n] = {"precision": k / K, "con_sancion": k, "rango": ic(bs[n]), "wilson": list(wilson(k, K)),
                    "carga": carga(X, top, desde, hasta),
                    "por_motivo": X.index[top][y[top] == 1].map(cat).value_counts().to_dict()}
    dif = {f"v0-{b_}": {"dif": filas["v0"]["precision"] - filas[b_]["precision"], "rango": ic(bs["v0"] - bs[b_])}
           for b_ in ("hoy", "simple", "regla")}
    return filas, dif


salida = {}
mot = s[s["inicio"] >= "2024-01-01"].sort_values("inicio").drop_duplicates("ruc").set_index("ruc")["motivo"].map(categoria)
# ---------------------------------------------------------------- prueba principal
Xtr, ytr = ventana(2018, 2021, 2022, 2024)
Xte, yte = ventana(2022, 2023, 2024)
f_tam, c_tam = entrenar(Xtr, ytr, TAM, l1=False)
f_v0, c_v0 = entrenar(Xtr, ytr, CARTAS + ANTEC)
listas = listas_de(Xte, f_tam, f_v0)
filas, dif = evaluar(Xte, yte, listas, 2022, 2023, 11, lambda r: mot.get(r, "otros"))
top = listas["v0"][:K]
T = Xte.iloc[top].assign(y=yte[top])
con_alerta = T[CARTAS].sum(axis=1).gt(0)
seg = {}
for nombre, m in {"con alerta de vínculo": con_alerta, "solo por antecedentes": ~con_alerta,
                  "persona natural": T["persona"].eq(1), "empresa": T["persona"].eq(0),
                  "vende a una sola entidad": T["entidades"].eq(1)}.items():
    n, k = int(m.sum()), int(T.loc[m, "y"].sum())
    seg[nombre] = {"n": n, "con_sancion": k, "precision": k / n if n else None, "wilson": list(wilson(k, n))}
bases = {"persona natural": Xte["persona"].eq(1), "vende a una sola entidad": Xte["entidades"].eq(1)}
for nombre, m in bases.items():
    seg[nombre]["base_del_grupo"] = float(yte[m.to_numpy()].mean())
salida["prueba"] = {"proveedores": int(len(Xte)), "con_sancion": int(yte.sum()), "base": float(yte.mean()),
                    "entrenamiento": int(len(Xtr)), "entrenamiento_con_sancion": int(ytr.sum()),
                    "coef_v0": c_v0, "coef_simple": c_tam, "filas": filas, "diferencias": dif, "segmentos_v0": seg,
                    "solape_v0_regla": len(set(top) & set(listas["regla"][:K]))}
print("\nPRUEBA 2022-2023 -> sanciones desde 2024")
for n, f in filas.items():
    print(f"  {n:<7} {f['precision']:.2%} [{f['rango'][0]:.2%} a {f['rango'][1]:.2%}] carga {f.get('carga')} "
          f"motivos {f.get('por_motivo')}")
for n, d in dif.items():
    print(f"  diferencia {n}: {d['dif']:+.2%} [{d['rango'][0]:+.2%} a {d['rango'][1]:+.2%}]")
print("  coeficientes del v0:", c_v0)
print("  segmentos del v0:", {k: (v["n"], v["con_sancion"]) for k, v in seg.items()}, flush=True)

# ---------------------------------------------------------------- confirmación con 2026
Xtr3, ytr3 = ventana(2020, 2023, 2024, 2026)
X26, y26 = ventana(2024, 2025, 2026)
f_tam3, _ = entrenar(Xtr3, ytr3, TAM, l1=False)
f_v03, _ = entrenar(Xtr3, ytr3, CARTAS + ANTEC)
mot26 = s[s["inicio"] >= "2026-01-01"].sort_values("inicio").drop_duplicates("ruc").set_index("ruc")["motivo"].map(categoria)
f26, d26 = evaluar(X26, y26, listas_de(X26, f_tam3, f_v03), 2024, 2025, 17, lambda r: mot26.get(r, "otros"))
salida["confirmacion_2026"] = {"proveedores": int(len(X26)), "con_sancion": int(y26.sum()), "base": float(y26.mean()),
                               "filas": f26, "diferencias": d26}
print("\nCONFIRMACIÓN 2024-2025 -> sanciones de 2026")
for n, f in f26.items():
    print(f"  {n:<7} {f['precision']:.2%} [{f['rango'][0]:.2%} a {f['rango'][1]:.2%}]")
for n, d in d26.items():
    print(f"  diferencia {n}: {d['dif']:+.2%} [{d['rango'][0]:+.2%} a {d['rango'][1]:+.2%}]")
(SALIDA / "evaluacion_e5.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nlisto en {time.time() - t0:.0f} s")
