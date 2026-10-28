"""Paso 12: evaluación del E5. Baselines contra el modelo v0, con rango por bootstrap y error por segmento.

Mismo corte que el paso 8. Las alertas de 2022-2023 se prueban contra las sanciones del Tribunal que empiezan desde
2024, y quien ya tenía una sanción que empezó antes de 2024 sale de la prueba. El modelo v0 (puntaje por puntos) no se
ajusta con sanciones: sus pesos son la gravedad legal por la probabilidad. La regresión de tamaño y el Lasso se entrenan
con las alertas de 2018-2021 contra sanciones desde 2022, igual que en el paso 8. La confirmación usa las alertas de
2024-2025 contra sanciones desde 2026, como el paso 9.

- Métrica principal: de los 1 000 primeros proveedores de cada lista, el porcentaje con una sanción posterior.
- Contrapeso: la carga para el OCI, es decir, cuántos proveedores de la lista le tocan a cada entidad (mediana y
  percentil 90 entre las entidades que reciben alguno).
- Rango: intervalo del 95 % por bootstrap de proveedores (pesos de Poisson, 1 000 repeticiones), pareado entre listas.

Solo imprime agregados y los guarda en datos/procesados/radar/evaluacion_e5.json.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 12_evaluacion_e5.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, wilson  # noqa: E402

t0 = time.time()
K, B = 1000, 1000
CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
TAM = ["log_monto", "log_ordenes", "entidades", "persona"]
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto"])
al = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["ruc", "anio", "tarjeta", "puntos"])
s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio"])
print(f"datos leídos en {time.time() - t0:.0f} s", flush=True)


def ventana(desde, hasta, etiqueta_desde):
    """Igual que en el paso 8: una fila por proveedor, con su tamaño, los puntos de cada alerta y la etiqueta."""
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta)]
    X = v.groupby("ruc").agg(monto=("monto", "sum"), ordenes=("id", "size"), entidades=("ruc_entidad", "nunique"))
    X["log_monto"], X["log_ordenes"] = np.log1p(X["monto"]), np.log1p(X["ordenes"])
    X["persona"] = X.index.str.startswith("10").astype(int)
    aa = al[(al["anio"] >= desde) & (al["anio"] <= hasta)]
    piv = aa.pivot_table(index="ruc", columns="tarjeta", values="puntos", aggfunc="max")
    for t in CARTAS:
        X[t] = piv[t].reindex(X.index).fillna(0) if t in piv else 0.0
    corte = pd.Timestamp(f"{etiqueta_desde}-01-01")
    X = X[~X.index.isin(set(s.loc[s["inicio"] < corte, "ruc"]))]
    y = X.index.isin(set(s.loc[s["inicio"] >= corte, "ruc"])).astype(int)
    return X, y


def orden(primero, desempate):
    """Índices de mayor a menor por la primera clave y, si empatan, por la segunda (la regla del radar: mayor monto)."""
    return np.lexsort((-np.asarray(desempate), -np.asarray(primero)))


def carga(X, idx, desde, hasta):
    """Proveedores de la lista por entidad que les compró en la ventana: mediana y percentil 90."""
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta) & o["ruc"].isin(set(X.index[idx]))]
    por_ent = v.drop_duplicates(["ruc_entidad", "ruc"]).groupby("ruc_entidad").size()
    return {"entidades": int(len(por_ent)), "mediana": float(por_ent.median()), "p90": float(por_ent.quantile(0.9))}


def bootstrap(y, ordenes, rng, k=K, b=B, largo=4 * K):
    """Precisión en los k primeros con pesos de Poisson por proveedor; los mismos pesos para todas las listas."""
    res = {n: np.empty(b) for n in ordenes}
    for r in range(b):
        w = rng.poisson(1.0, size=len(y))
        for n, idx in ordenes.items():
            ww = w[idx[:largo]]
            cw = np.cumsum(ww)
            j = int(np.searchsorted(cw, k))  # primera posición donde el peso acumulado llega a k
            res[n][r] = (ww[: j + 1] * y[idx[: j + 1]]).sum() / cw[j]
    return res


def resumen(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


salida = {}
# ---------------------------------------------------------------- prueba principal: alertas 2022-2023, sanciones desde 2024
Xtr, ytr = ventana(2018, 2021, 2022)
Xte, yte = ventana(2022, 2023, 2024)
print(f"entrenamiento {len(Xtr):,} proveedores ({ytr.sum():,} con sanción posterior); prueba {len(Xte):,} "
      f"({yte.sum():,}, base {yte.mean():.3%}) ({time.time() - t0:.0f} s)", flush=True)
rng = np.random.default_rng(7)
neg = np.flatnonzero(ytr == 0)
fil = np.sort(np.concatenate([np.flatnonzero(ytr == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))
Xs, ys = Xtr.iloc[fil], ytr[fil]
sc_t = StandardScaler().fit(Xs[TAM])
p_tam = LogisticRegression(max_iter=500).fit(sc_t.transform(Xs[TAM]), ys).predict_proba(sc_t.transform(Xte[TAM]))[:, 1]
cols = CARTAS + TAM
sc_l = StandardScaler().fit(Xs[cols])
g = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", max_iter=500), {"C": [0.001, 0.01, 0.1, 1.0]},
                 cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc").fit(sc_l.transform(Xs[cols]), ys)
p_las = g.predict_proba(sc_l.transform(Xte[cols]))[:, 1]
puntos = Xte[CARTAS].sum(axis=1).to_numpy()
monto = Xte["monto"].to_numpy()
listas = {
    "hoy": orden(monto, np.zeros(len(monto))),               # lo que se hace hoy: lo más caro primero
    "simple": orden(p_tam, monto),                           # regresión logística con 4 variables de tamaño
    "v0": orden(puntos, monto),                              # puntaje por puntos; empate: mayor monto
    "lasso": orden(p_las, monto),                            # Lasso con alertas y tamaño (paso 8)
}
print(f"modelos listos ({time.time() - t0:.0f} s); Lasso C={g.best_params_['C']}", flush=True)
bs = bootstrap(yte, listas, np.random.default_rng(11))
base = float(yte.mean())
azar = np.random.default_rng(13).binomial(K, base, size=10_000) / K  # 1 000 proveedores al azar
filas = {"azar": {"precision": base, "rango": resumen(azar), "auc": 0.5}}
puntajes = {"hoy": monto, "simple": p_tam, "v0": puntos, "lasso": p_las}
for n, idx in listas.items():
    top = idx[:K]
    k = int(yte[top].sum())
    filas[n] = {"precision": k / K, "con_sancion": k, "rango": resumen(bs[n]), "wilson": list(wilson(k, K)),
                "auc": float(roc_auc_score(yte, puntajes[n])), "carga": carga(Xte, top, 2022, 2023)}
# carga de la lista al azar: promedio de 20 sorteos
cargas = [carga(Xte, np.random.default_rng(100 + i).choice(len(Xte), K, replace=False), 2022, 2023) for i in range(20)]
filas["azar"]["carga"] = {c: float(np.mean([x[c] for x in cargas])) for c in ("entidades", "mediana", "p90")}
dif = {f"{a}-{b_}": {"dif": filas[a]["precision"] - filas[b_]["precision"], "rango": resumen(bs[a] - bs[b_])}
       for a, b_ in [("v0", "hoy"), ("v0", "simple"), ("lasso", "hoy"), ("lasso", "v0"), ("simple", "hoy")]}
top_v0, top_hoy = set(listas["v0"][:K]), set(listas["hoy"][:K])
solape = {"proveedores_en_ambas": len(top_v0 & top_hoy),
          "sancionados_solo_v0": int(yte[list(top_v0 - top_hoy)].sum()),
          "sancionados_solo_hoy": int(yte[list(top_hoy - top_v0)].sum()),
          "sancionados_en_ambas": int(yte[list(top_v0 & top_hoy)].sum())}
salida["prueba"] = {"proveedores": int(len(Xte)), "con_sancion": int(yte.sum()), "base": base,
                    "entrenamiento": int(len(Xtr)), "entrenamiento_con_sancion": int(ytr.sum()),
                    "lasso_C": g.best_params_["C"], "filas": filas, "diferencias": dif, "solape_v0_hoy": solape}
print("\nPRUEBA 2022-2023 -> sanciones desde 2024")
for n, f in filas.items():
    print(f"  {n:<7} {f['precision']:.2%} [{f['rango'][0]:.2%} a {f['rango'][1]:.2%}]  AUC {f['auc']:.3f}  carga {f['carga']}")
for n, d in dif.items():
    print(f"  diferencia {n}: {d['dif']:+.2%} [{d['rango'][0]:+.2%} a {d['rango'][1]:+.2%}]")
print(f"  solape v0 y hoy: {solape}", flush=True)

# ---------------------------------------------------------------- error por segmento del modelo v0
top = listas["v0"][:K]
T = Xte.iloc[top].assign(y=yte[top])
principal = T[CARTAS].idxmax(axis=1).map({"empresa_espejo": "vínculo societario", "mismo_dueno": "vínculo societario",
                                           "funcionario_propio": "funcionario", "doble_percepcion": "funcionario",
                                           "familiar": "posible familiar"})
segmentos = {"persona natural": T["persona"].eq(1), "empresa": T["persona"].eq(0),
             "vende a una sola entidad": T["entidades"].eq(1), "vende a varias entidades": T["entidades"].gt(1)}
for nombre in ["vínculo societario", "funcionario", "posible familiar"]:
    segmentos[f"alerta principal: {nombre}"] = principal.eq(nombre)
segmentos["alerta principal: funcionario o posible familiar"] = principal.isin(["funcionario", "posible familiar"])
# base de cada segmento en toda la prueba, para leer la precisión frente a lo esperable en ese grupo
bases = {"persona natural": Xte["persona"].eq(1), "empresa": Xte["persona"].eq(0),
         "vende a una sola entidad": Xte["entidades"].eq(1), "vende a varias entidades": Xte["entidades"].gt(1)}
seg = {}
print("\nSEGMENTOS DEL MODELO V0 (sus 1 000 primeros)")
for nombre, m in segmentos.items():
    n, k = int(m.sum()), int(T.loc[m, "y"].sum())
    lo, hi = wilson(k, n)
    seg[nombre] = {"n": n, "con_sancion": k, "precision": k / n if n else None, "wilson": [lo, hi]}
    extra = ""
    if nombre in bases:
        b = float(yte[bases[nombre].to_numpy()].mean())
        seg[nombre]["base_del_grupo"] = b
        extra = f"; base del grupo {b:.3%}, {k / max(n, 1) / b:.1f} veces"
    print(f"  {nombre:<48}{n:>5} proveedores, {k:>3} con sanción ({k / max(n, 1):.1%}; IC {lo:.1%} a {hi:.1%}{extra})")
salida["segmentos_v0"] = seg

# ---------------------------------------------------------------- confirmación: alertas 2024-2025, sanciones desde 2026
X26, y26 = ventana(2024, 2025, 2026)
m26, p26 = X26["monto"].to_numpy(), X26[CARTAS].sum(axis=1).to_numpy()
l26 = {"hoy": orden(m26, np.zeros(len(m26))), "v0": orden(p26, m26)}
bs26 = bootstrap(y26, l26, np.random.default_rng(17))
conf = {"proveedores": int(len(X26)), "con_sancion": int(y26.sum()), "base": float(y26.mean())}
for n, idx in l26.items():
    k = int(y26[idx[:K]].sum())
    conf[n] = {"precision": k / K, "con_sancion": k, "rango": resumen(bs26[n])}
conf["v0-hoy"] = {"dif": conf["v0"]["precision"] - conf["hoy"]["precision"], "rango": resumen(bs26["v0"] - bs26["hoy"])}
conf["proveedores_en_ambas"] = len(set(l26["v0"][:K]) & set(l26["hoy"][:K]))
salida["confirmacion_2026"] = conf
print(f"\nCONFIRMACIÓN 2024-2025 -> sanciones desde 2026: {conf}")

(SALIDA / "evaluacion_e5.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nlisto en {time.time() - t0:.0f} s: {SALIDA / 'evaluacion_e5.json'}")
