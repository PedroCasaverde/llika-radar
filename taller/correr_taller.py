# %% [markdown]
# # LLIKA · Taller de evaluación: el experimento mínimo
#
# Este notebook repite la evaluación del E5 con tablas livianas (una fila por proveedor, sin RUC), para que cualquiera
# pueda correrla sin bajar los datos completos del radar.
#
# **Cómo correrlo:** en Colab, *Entorno de ejecución → Ejecutar todas*. En una laptop: `pip install pandas pyarrow scikit-learn`
# y ejecutar las celdas de arriba abajo. Tarda entre 2 y 5 minutos.
#
# Lo que mide, en el orden del taller:
# 1. **Pre-registro**: referencia, conjunto, métrica, repeticiones y semillas, antes de correr nada.
# 2. **La referencia** (lo que se hace hoy: revisar primero lo más caro) con su número y su rango. Es la tarea del carril A.
# 3. **El candidato v0** (opcional): el modelo del E5, entrenado con la misma receta.
# 4. **Contrapeso, segmentos y comprobación** contra las cifras del E5.
# 5. **Registro de la corrida** en un JSON, para que el auditor compare.
#
# Si la descarga falla, poner `MODO_DEMO = True`: usa el 10 % de los proveedores y los números cambian (hay que declararlo).

# %%
MODO_DEMO = False     # True solo si no se pueden cargar las tablas completas
SEMILLA_BOOT = 11     # la misma del E5
K, B = 1000, 1000     # los 1000 primeros de cada lista y 1000 repeticiones del bootstrap
URL = "https://raw.githubusercontent.com/PedroCasaverde/llika-radar/main/taller/datos/"

import hashlib, json, platform, time, urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
ANTEC = ["log_multas", "log_penalidades", "evasion", "reciente"]

# %%
# Busca las tablas junto al notebook; si no están, las baja del repositorio
ARCHIVOS = ["aprendizaje.parquet", "examen.parquet", "examen_entidades.parquet", "huellas.json"]
DATOS = next((d for d in (Path("datos"), Path("taller/datos")) if (d / "examen.parquet").exists()), Path("datos"))
DATOS.mkdir(exist_ok=True)
for f in ARCHIVOS:
    if not (DATOS / f).exists():
        print("bajando", f)
        urllib.request.urlretrieve(URL + f, DATOS / f)

huellas = json.loads((DATOS / "huellas.json").read_text(encoding="utf-8"))
for f, h in huellas["archivos"].items():
    real = hashlib.sha256((DATOS / f).read_bytes()).hexdigest()
    print(f"{f:<26} {'OK, es el conjunto congelado' if real == h['sha256'] else 'OJO: no coincide la huella'}")

tr = pd.read_parquet(DATOS / "aprendizaje.parquet")
te = pd.read_parquet(DATOS / "examen.parquet")
pares = pd.read_parquet(DATOS / "examen_entidades.parquet")

if MODO_DEMO:
    # muestra al azar del 10 % de los proveedores; K baja en la misma proporción
    rng_demo = np.random.default_rng(2026)
    te = te[rng_demo.random(len(te)) < 0.10].reset_index(drop=True)
    K = 100
    print("MODO_DEMO: 10 % de los proveedores del examen y K = 100. Declararlo en la bitácora.")

y = te["y"].to_numpy()
print(f"\nexamen: {len(te):,} proveedores, {y.sum():,} con sanción posterior (base {y.mean():.3%})")
print(f"aprendizaje: {len(tr):,} proveedores, {tr['y'].sum():,} con sanción en 2022-2023")

# %% [markdown]
# ## 1. Pre-registro (08:15)
# Se declara antes de correr. La celda lo guarda en `bitacora/preregistro.json`.

# %%
PRE_REGISTRO = {
    "equipo": "G5 · LLIKA (carril A: medir la referencia)",
    "referencia": "lo que se hace hoy: los proveedores ordenados por monto total, de mayor a menor",
    "candidato": "modelo v0 del E5: regresión logística L1 con alertas de vínculo y antecedentes (opcional el sábado)",
    "conjunto": {"descripcion": "proveedores con órdenes en 2022-2023, sin sanción previa a 2024; etiqueta = sanción desde 2024",
                 "n": int(len(te)), "con_sancion": int(y.sum()), "sha256": huellas["archivos"]["examen.parquet"]["sha256"]},
    "metrica": f"precisión en los {K} primeros: de los {K} primeros de la lista, qué porcentaje tuvo sanción posterior",
    "contrapeso": "carga por entidad: cuántos de esos proveedores le tocan a cada entidad (mediana y percentil 90)",
    "incertidumbre": f"bootstrap de proveedores con pesos de Poisson, {B} repeticiones, semilla {SEMILLA_BOOT}, pareado entre listas",
    "modo_demo": MODO_DEMO,
    "fecha": datetime.now().isoformat(timespec="seconds"),
}
Path("bitacora").mkdir(exist_ok=True)
Path("bitacora/preregistro.json").write_text(json.dumps(PRE_REGISTRO, ensure_ascii=False, indent=1), encoding="utf-8")
for k_, v_ in PRE_REGISTRO.items():
    print(f"{k_:>13}: {v_}")

# %%
# Herramientas: las mismas fórmulas que radar/12_evaluacion_e5.py
def orden(*claves):
    # ordena de mayor a menor por la primera clave; las siguientes desempatan
    return np.lexsort(tuple(-np.asarray(c, dtype=float) for c in reversed(claves)))


def bootstrap(y, listas, rng, k, b, largo=None):
    largo = largo or 4 * k
    res = {n: np.empty(b) for n in listas}
    for r in range(b):
        w = rng.poisson(1.0, size=len(y))          # cada proveedor entra 0, 1, 2... veces
        for n, idx in listas.items():
            ww = w[idx[:largo]]
            cw = np.cumsum(ww)
            j = int(np.searchsorted(cw, k))
            res[n][r] = (ww[: j + 1] * y[idx[: j + 1]]).sum() / cw[j]
    return res


def ic(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - m), float(c + m)]


def carga(top_ids):
    por_ent = pares[pares["id"].isin(top_ids)].groupby("entidad").size()
    return {"entidades": int(len(por_ent)), "mediana": float(por_ent.median()), "p90": float(por_ent.quantile(0.9))}


def medir(idx):
    top = idx[:K]
    k = int(y[top].sum())
    return {"precision": k / K, "con_sancion": k, "wilson": wilson(k, K), "carga": carga(te["id"].to_numpy()[top])}

# %% [markdown]
# ## 2. La referencia (Bloque 1, carril A)
# Lo que se hace hoy: revisar primero lo más caro. Sin modelo, sin entrenar nada.

# %%
listas = {"hoy": orden(te["monto"])}
res = {"hoy": medir(listas["hoy"])}
r = res["hoy"]
print(f"Referencia (lo más caro primero): {r['precision']:.1%} de los {K} primeros tuvo sanción posterior "
      f"({r['con_sancion']} proveedores). Al azar sería {y.mean():.2%}.")
print(f"Carga por entidad: mediana {r['carga']['mediana']:.0f}, percentil 90 {r['carga']['p90']:.0f}")

# %% [markdown]
# ## 3. Comparaciones (opcionales el sábado)
# La regla por puntos no se entrena. El v0 se entrena con la ventana de aprendizaje y se mide en el examen.

# %%
CORRER_CANDIDATO = True

listas["regla"] = orden(te[CARTAS].sum(axis=1), te["monto"])
res["regla"] = medir(listas["regla"])

coef = None
if CORRER_CANDIDATO:
    t0 = time.time()
    ytr = tr["y"].to_numpy()
    rng = np.random.default_rng(7)                    # todos los positivos y 100 000 negativos, como en el E5
    neg = np.flatnonzero(ytr == 0)
    fil = np.sort(np.concatenate([np.flatnonzero(ytr == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))
    cols = CARTAS + ANTEC
    sc = StandardScaler().fit(tr.iloc[fil][cols])
    g = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000), {"C": [0.001, 0.01, 0.1, 1.0]},
                     cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc")
    m = g.fit(sc.transform(tr.iloc[fil][cols]), ytr[fil]).best_estimator_
    coef = {c: round(float(v), 3) for c, v in zip(cols, m.coef_[0])}
    puntaje = m.predict_proba(sc.transform(te[cols]))[:, 1]
    listas["v0"] = orden(puntaje, te["monto"])
    res["v0"] = medir(listas["v0"])
    print(f"v0 entrenado en {time.time() - t0:.0f} s; pesos: {coef}")

for n_, r_ in res.items():
    print(f"{n_:<6} {r_['precision']:.1%}  ({r_['con_sancion']} de {K})  carga mediana {r_['carga']['mediana']:.0f}")

# %%
# Incertidumbre: bootstrap pareado (todas las listas con los mismos sorteos)
t0 = time.time()
bs = bootstrap(y, listas, np.random.default_rng(SEMILLA_BOOT), K, B)
for n_ in res:
    res[n_]["rango"] = ic(bs[n_])
dif = {f"{c}-hoy": {"dif": res[c]["precision"] - res["hoy"]["precision"], "rango": ic(bs[c] - bs["hoy"])}
       for c in res if c != "hoy"}
print(f"bootstrap de {B} repeticiones en {time.time() - t0:.0f} s")
for n_, r_ in res.items():
    print(f"{n_:<6} {r_['precision']:.1%}  rango 95 % [{r_['rango'][0]:.1%} a {r_['rango'][1]:.1%}]")
for n_, d_ in dif.items():
    print(f"diferencia {n_}: {d_['dif'] * 100:+.1f} puntos porcentuales [{d_['rango'][0] * 100:+.1f} a {d_['rango'][1] * 100:+.1f}]")

# %%
# Error por segmento: dónde acierta y dónde falla cada lista
def segmentos(idx):
    T = te.iloc[idx[:K]]
    yy = T["y"]
    con_alerta = T[CARTAS].sum(axis=1).gt(0)
    grupos = {"con alerta de vínculo": con_alerta, "sin alerta de vínculo": ~con_alerta,
              "persona natural": T["persona"].eq(1), "empresa": T["persona"].eq(0),
              "vende a una sola entidad": T["entidades"].eq(1)}
    return {g_: {"n": int(mm.sum()), "con_sancion": int(yy[mm].sum()),
                 "precision": (float(yy[mm].mean()) if mm.sum() else None)} for g_, mm in grupos.items()}


seg = {n_: segmentos(idx) for n_, idx in listas.items()}
print(pd.DataFrame({n_: {g_: f"{v_['con_sancion']}/{v_['n']}" for g_, v_ in s_.items()} for n_, s_ in seg.items()}))
motivos = te.iloc[listas["hoy"][:K]].query("y == 1")["motivo"].value_counts().to_dict()
print("\nmotivo de la sanción en la referencia:", motivos)

# %%
# Comprobación contra el E5 (solo con las tablas completas)
ESPERADO = {"hoy": 0.037, "regla": 0.071, "v0": 0.143}
if MODO_DEMO:
    print("MODO_DEMO: no se compara con el E5.")
else:
    for n_, e_ in ESPERADO.items():
        if n_ in res:
            ok = abs(res[n_]["precision"] - e_) < 0.0015
            print(f"{n_:<6} E5 {e_:.1%} · ahora {res[n_]['precision']:.1%} → {'coincide' if ok else 'NO coincide, revisar'}")

# %% [markdown]
# ## 4. La línea de la plenaria y el registro de la corrida
# El JSON queda en `bitacora/`; el auditor corre el notebook y compara su JSON con este.

# %%
cand = "v0" if "v0" in res else "regla"
linea = (f"Referencia (lo más caro primero) {res['hoy']['precision']:.1%} · candidato ({cand}) {res[cand]['precision']:.1%} · "
         f"diferencia {dif[cand + '-hoy']['dif'] * 100:+.1f} p.p. · dispersión: rango 95 % de la diferencia "
         f"[{dif[cand + '-hoy']['rango'][0] * 100:+.1f} a {dif[cand + '-hoy']['rango'][1] * 100:+.1f}] p.p. "
         f"({B} repeticiones, semilla {SEMILLA_BOOT})")
print(linea)

corrida = {"fecha": datetime.now().isoformat(timespec="seconds"), "modo_demo": MODO_DEMO,
           "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "sklearn": sklearn.__version__, "semillas": {"bootstrap": SEMILLA_BOOT, "negativos": 7, "cv": 1},
           "k": K, "repeticiones": B, "huellas": {f: h["sha256"] for f, h in huellas["archivos"].items()},
           "resultados": res, "diferencias": dif, "segmentos": seg, "pesos_v0": coef, "linea_plenaria": linea}
nombre = f"bitacora/corrida_{datetime.now():%Y%m%d_%H%M%S}.json"
Path(nombre).write_text(json.dumps(corrida, ensure_ascii=False, indent=1), encoding="utf-8")
print("guardado:", nombre)
