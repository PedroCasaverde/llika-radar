"""Paso 13: ¿se puede mejorar el modelo v0? Variantes y regla de selección pre-registradas en HALLAZGOS 50.

- A: v0 actual (puntos; empate por monto total).
- B: v0 con intensidad en el desempate (puntos; empate por órdenes alertadas y luego por monto alertado).
- C: pesos aprendidos con solo los puntos de cada tipo de alerta (regresión logística L1).
- D: C más intensidad (órdenes, monto y entidades con alerta) y detalle de la empresa espejo (sancionados vinculados
  distintos y si alguno tiene una sanción definitiva vigente al cierre de la ventana).
- E: D más las 4 variables de tamaño.

Ventanas sin fuga (las etiquetas de entrenamiento y de selección no se cruzan con las de la prueba):
- entrenamiento para elegir: alertas 2018-2019, sanciones que empiezan en 2020-2021;
- validación (elige): alertas 2020-2021, sanciones que empiezan en 2022-2023;
- prueba del E5 (una sola vez): se reentrena con alertas 2018-2021 y sanciones de 2022-2023, y se prueba con alertas
  2022-2023 y sanciones desde 2024;
- confirmación: se reentrena con alertas 2020-2023 y sanciones de 2024-2025, y se prueba con 2024-2025 frente a 2026.
Solo imprime agregados y guarda datos/procesados/radar/mejoras_v0.json.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 -W ignore 13_mejoras_v0.py
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
INTENS = ["log_alertadas", "log_monto_alertado", "log_ent_alertadas", "log_sanc_vinculados", "vinculo_definitivo"]
TAM = ["log_monto", "log_ordenes", "entidades", "persona"]
VARIANTES = {"C": CARTAS, "D": CARTAS + INTENS, "E": CARTAS + INTENS + TAM}
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto"])
al = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "tarjeta", "puntos",
                                                                 "monto"])
s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio", "definitiva"])
esp = pd.read_parquet(SALIDA / "alerta_empresa_espejo.parquet", columns=["id", "ruc_sanc"])
print(f"datos leídos en {time.time() - t0:.0f} s", flush=True)


def ventana(desde, hasta, etq_desde, etq_hasta=None):
    """Una fila por proveedor con tamaño, puntos por alerta, intensidad y etiqueta (sanción que empieza en la ventana
    de etiqueta). Las variables solo usan información hasta el 31/12 de `hasta`."""
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta)]
    X = v.groupby("ruc").agg(monto=("monto", "sum"), ordenes=("id", "size"), entidades=("ruc_entidad", "nunique"))
    X["log_monto"], X["log_ordenes"] = np.log1p(X["monto"]), np.log1p(X["ordenes"])
    X["persona"] = X.index.str.startswith("10").astype(int)
    aa = al[(al["anio"] >= desde) & (al["anio"] <= hasta)]
    piv = aa.pivot_table(index="ruc", columns="tarjeta", values="puntos", aggfunc="max")
    for t in CARTAS:
        X[t] = piv[t].reindex(X.index).fillna(0) if t in piv else 0.0
    u = aa[aa["tarjeta"].isin(CARTAS)].drop_duplicates("id")
    g = u.groupby("ruc").agg(alertadas=("id", "size"), monto_alertado=("monto", "sum"),
                             ent_alertadas=("ruc_entidad", "nunique"))
    for c in ["alertadas", "monto_alertado", "ent_alertadas"]:
        X["log_" + c] = np.log1p(g[c].reindex(X.index).fillna(0))
    e = aa[aa["tarjeta"].eq("empresa_espejo")][["id", "ruc"]].merge(esp, on="id")
    fin = pd.Timestamp(f"{hasta}-12-31")
    defin = set(s.loc[s["definitiva"] & (s["inicio"] <= fin), "ruc"])
    X["log_sanc_vinculados"] = np.log1p(e.groupby("ruc")["ruc_sanc"].nunique().reindex(X.index).fillna(0))
    X["vinculo_definitivo"] = e.assign(d=e["ruc_sanc"].isin(defin)).groupby("ruc")["d"].any().reindex(X.index) \
        .fillna(False).astype(int)
    X["alertadas"] = g["alertadas"].reindex(X.index).fillna(0)
    X["monto_alertado"] = g["monto_alertado"].reindex(X.index).fillna(0)
    corte = pd.Timestamp(f"{etq_desde}-01-01")
    X = X[~X.index.isin(set(s.loc[s["inicio"] < corte, "ruc"]))]
    lab = s["inicio"] >= corte
    if etq_hasta is not None:
        lab &= s["inicio"] < pd.Timestamp(f"{etq_hasta}-01-01")
    y = X.index.isin(set(s.loc[lab, "ruc"])).astype(int)
    return X, y


def entrenar(X, y, cols, semilla=7):
    """Igual que el paso 8: todos los positivos y 100 000 negativos al azar; L1 con C elegido por validación cruzada."""
    rng = np.random.default_rng(semilla)
    neg = np.flatnonzero(y == 0)
    fil = np.sort(np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))
    sc = StandardScaler().fit(X.iloc[fil][cols])
    g = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000), {"C": [0.001, 0.01, 0.1, 1.0]},
                     cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc")
    g.fit(sc.transform(X.iloc[fil][cols]), y[fil])
    coef = {c: round(float(v), 3) for c, v in zip(cols, g.best_estimator_.coef_[0])}
    return (lambda Z: g.predict_proba(sc.transform(Z[cols]))[:, 1]), g.best_params_["C"], coef


def orden(*claves):
    """Índices de mayor a menor por la primera clave y, en empate, por las siguientes."""
    return np.lexsort(tuple(-np.asarray(c, dtype=float) for c in reversed(claves)))


def carga(X, idx, desde, hasta):
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta) & o["ruc"].isin(set(X.index[idx]))]
    por_ent = v.drop_duplicates(["ruc_entidad", "ruc"]).groupby("ruc_entidad").size()
    return float(por_ent.median())


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


def evaluar(X, y, listas, desde, hasta, semilla):
    bs = bootstrap(y, listas, np.random.default_rng(semilla))
    out = {}
    for n, idx in listas.items():
        k = int(y[idx[:K]].sum())
        out[n] = {"precision": k / K, "con_sancion": k, "rango": ic(bs[n]), "carga_mediana": carga(X, idx[:K], desde, hasta)}
        if n != "A":
            out[n]["dif_vs_A"] = {"dif": out[n]["precision"] - int(y[listas["A"][:K]].sum()) / K, "rango": ic(bs[n] - bs["A"])}
    return out


def listas_de(X, modelos):
    p = X[CARTAS].sum(axis=1).to_numpy()
    li = {"A": orden(p, X["monto"]), "B": orden(p, X["alertadas"], X["monto_alertado"])}
    for n, f in modelos.items():
        li[n] = orden(f(X), X["monto"])
    return li


salida = {}
# ---------------------------------------------------------------- 1) selección en validación
Xtr, ytr = ventana(2018, 2019, 2020, 2022)
Xva, yva = ventana(2020, 2021, 2022, 2024)
print(f"entrenamiento {len(Xtr):,} ({ytr.sum():,} positivos); validación {len(Xva):,} ({yva.sum():,} positivos, base "
      f"{yva.mean():.3%}) ({time.time() - t0:.0f} s)", flush=True)
modelos, info = {}, {}
for n, cols in VARIANTES.items():
    f, c, coef = entrenar(Xtr, ytr, cols)
    modelos[n], info[n] = f, {"C": c, "coeficientes": coef}
    print(f"  {n}: C={c} {coef}", flush=True)
val = evaluar(Xva, yva, listas_de(Xva, modelos), 2020, 2021, 21)
print("\nVALIDACIÓN (alertas 2020-2021, sanciones 2022-2023)")
for n, r in val.items():
    extra = f"; frente a A {r['dif_vs_A']['dif']:+.2%} {[round(x * 100, 2) for x in r['dif_vs_A']['rango']]}" if n != "A" else ""
    print(f"  {n}: {r['precision']:.2%} {[round(x * 100, 2) for x in r['rango']]} carga {r['carga_mediana']}{extra}")
candidatas = [n for n, r in val.items() if n != "A" and r["dif_vs_A"]["rango"][0] > 0 and r["carga_mediana"] <= 5]
elegida = max(candidatas, key=lambda n: val[n]["precision"]) if candidatas else "A"
print(f"  regla: candidatas {candidatas}; elegida {elegida}", flush=True)
salida["validacion"] = {"filas": val, "modelos": info, "elegida": elegida}

# ---------------------------------------------------------------- 2) prueba del E5, una sola vez, A frente a la elegida
Xtr2, ytr2 = ventana(2018, 2021, 2022, 2024)
Xte, yte = ventana(2022, 2023, 2024)
mod2 = {}
if elegida in VARIANTES:
    f, c, coef = entrenar(Xtr2, ytr2, VARIANTES[elegida])
    mod2[elegida] = f
    salida["prueba_modelo"] = {"C": c, "coeficientes": coef}
li = {k: v for k, v in listas_de(Xte, mod2).items() if k in ("A", elegida)}
salida["prueba"] = evaluar(Xte, yte, li, 2022, 2023, 23)
print(f"\nPRUEBA (alertas 2022-2023, sanciones desde 2024): {salida['prueba']}")
if elegida != "A":
    top = li[elegida][:K]
    T = Xte.iloc[top].assign(y=yte[top])
    seg = {}
    for nombre, m in {"persona natural": T["persona"].eq(1), "empresa": T["persona"].eq(0),
                      "vende a una sola entidad": T["entidades"].eq(1)}.items():
        n, k = int(m.sum()), int(T.loc[m, "y"].sum())
        seg[nombre] = {"n": n, "con_sancion": k, "wilson": list(wilson(k, n))}
    salida["segmentos_elegida"] = seg
    print("  segmentos de la elegida:", seg)

# ---------------------------------------------------------------- 3) confirmación con 2026
Xtr3, ytr3 = ventana(2020, 2023, 2024, 2026)
X26, y26 = ventana(2024, 2025, 2026)
mod3 = {}
if elegida in VARIANTES:
    mod3[elegida] = entrenar(Xtr3, ytr3, VARIANTES[elegida])[0]
li = {k: v for k, v in listas_de(X26, mod3).items() if k in ("A", elegida)}
salida["confirmacion_2026"] = evaluar(X26, y26, li, 2024, 2025, 27)
print(f"\nCONFIRMACIÓN 2026: {salida['confirmacion_2026']}")
(SALIDA / "mejoras_v0.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nlisto en {time.time() - t0:.0f} s")
