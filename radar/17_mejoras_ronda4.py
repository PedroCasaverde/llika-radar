"""Paso 17: ronda 4 (HALLAZGOS 53): bayesiana con previa legal, riesgo de entidad, fusión de rangos y boosting.

Base tomada del paso 15, mejoras del modelo v0, ronda 2 (HALLAZGOS 51): antecedentes de incumplimiento, evasión y recencia.

- A: v0 actual.
- F: v0; empate por número de antecedentes (multas del Tribunal y penalidades de contratos) y luego por monto.
- G: v0 más 1 punto si tiene antecedentes (solo para quien ya tiene alguna alerta del radar).
- H: G más 1 punto si es una empresa espejo que empezó a vender después de la sanción de su vinculado.
- I: regresión logística L1 con los puntos de cada alerta, los antecedentes, la evasión y la recencia.
- J: gradient boosting con todo lo anterior, más intensidad y tamaño.

Mismas ventanas y regla que el paso 13: se elige en validación (alertas 2020-2021, sanciones 2022-2023) y la elegida se
prueba una sola vez en la prueba del E5 (alertas 2022-2023, sanciones desde 2024) y en 2026. Solo imprime agregados y
guarda datos/procesados/radar/mejoras_ronda2.json.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 -W ignore 15_mejoras_ronda2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, wilson  # noqa: E402

t0 = time.time()
K, B = 1000, 1000
CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
ANTEC = ["log_multas", "log_penalidades", "evasion", "reciente"]
INTENS = ["log_alertadas", "log_monto_alertado", "log_ent_alertadas"]
TAM = ["log_monto", "log_ordenes", "entidades", "persona"]
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto", "fecha"])
al = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "tarjeta", "puntos",
                                                                 "monto", "fecha"])
s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio"])
esp = pd.read_parquet(SALIDA / "alerta_empresa_espejo.parquet", columns=["id", "ruc_sanc"])
ant = pd.read_parquet(SALIDA / "antecedentes.parquet")
print(f"datos leídos en {time.time() - t0:.0f} s", flush=True)


def ventana(desde, hasta, etq_desde, etq_hasta=None):
    """Una fila por proveedor; las variables solo usan información hasta el 31/12 de `hasta`."""
    fin = pd.Timestamp(f"{hasta}-12-31")
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta)]
    X = v.groupby("ruc").agg(monto=("monto", "sum"), ordenes=("id", "size"), entidades=("ruc_entidad", "nunique"))
    X["log_monto"], X["log_ordenes"] = np.log1p(X["monto"]), np.log1p(X["ordenes"])
    X["persona"] = X.index.str.startswith("10").astype(int)
    aa = al[(al["anio"] >= desde) & (al["anio"] <= hasta) & al["tarjeta"].isin(CARTAS)]
    piv = aa.pivot_table(index="ruc", columns="tarjeta", values="puntos", aggfunc="max")
    for t in CARTAS:
        X[t] = piv[t].reindex(X.index).fillna(0) if t in piv else 0.0
    u = aa.drop_duplicates("id")
    g = u.groupby("ruc").agg(alertadas=("id", "size"), monto_alertado=("monto", "sum"),
                             ent_alertadas=("ruc_entidad", "nunique"))
    for c in ["alertadas", "monto_alertado", "ent_alertadas"]:
        X["log_" + c] = np.log1p(g[c].reindex(X.index).fillna(0))
    # antecedentes hasta el cierre de la ventana
    a = ant[ant["fecha"] <= fin]
    for tipo, col in [("multa", "multas"), ("penalidad", "penalidades")]:
        n = a[a["tipo"].eq(tipo)].groupby("ruc").size()
        X[col] = n.reindex(X.index).fillna(0)
        X["log_" + col] = np.log1p(X[col])
    X["antecedentes"] = X["multas"] + X["penalidades"]
    # evasión: empresa espejo que empezó a vender después del inicio de la sanción de su vinculado
    primera = o[o["fecha"] <= fin].groupby("ruc")["fecha"].min()
    e = aa[aa["tarjeta"].eq("empresa_espejo")][["id", "ruc", "fecha"]].merge(esp, on="id")
    ini = e.merge(s.rename(columns={"ruc": "ruc_sanc"}), on="ruc_sanc")
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


def muestra(y, semilla=7):
    rng = np.random.default_rng(semilla)
    neg = np.flatnonzero(y == 0)
    return np.sort(np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))


def entrenar_l1(X, y, cols):
    fil = muestra(y)
    sc = StandardScaler().fit(X.iloc[fil][cols])
    g = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000), {"C": [0.001, 0.01, 0.1, 1.0]},
                     cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc")
    g.fit(sc.transform(X.iloc[fil][cols]), y[fil])
    coef = {c: round(float(v), 3) for c, v in zip(cols, g.best_estimator_.coef_[0])}
    return (lambda Z: g.predict_proba(sc.transform(Z[cols]))[:, 1]), {"C": g.best_params_["C"], "coeficientes": coef}


def entrenar_gb(X, y, cols):
    fil = muestra(y)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, early_stopping=True,
                                       validation_fraction=0.2, random_state=1).fit(X.iloc[fil][cols], y[fil])
    return (lambda Z: m.predict_proba(Z[cols])[:, 1]), {"iteraciones": int(m.n_iter_)}


APRENDIDAS = {"I": (entrenar_l1, CARTAS + ANTEC), "J": (entrenar_gb, CARTAS + ANTEC + INTENS + TAM)}


def orden(*claves):
    return np.lexsort(tuple(-np.asarray(c, dtype=float) for c in reversed(claves)))


def carga(X, idx, desde, hasta):
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta) & o["ruc"].isin(set(X.index[idx]))]
    return float(v.drop_duplicates(["ruc_entidad", "ruc"]).groupby("ruc_entidad").size().median())


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


def listas_de(X, modelos, solo=None):
    p = X[CARTAS].sum(axis=1).to_numpy()
    tiene = (p > 0).astype(float)
    g = p + tiene * (X["antecedentes"].to_numpy() > 0)
    li = {"A": orden(p, X["monto"]), "F": orden(p, X["antecedentes"], X["monto"]), "G": orden(g, X["monto"]),
          "H": orden(g + X["evasion"].to_numpy(), X["monto"])}
    for n, f in modelos.items():
        li[n] = orden(f(X), X["monto"])
    return {n: v for n, v in li.items() if solo is None or n in solo}


def evaluar(X, y, listas, desde, hasta, semilla):
    bs = bootstrap(y, listas, np.random.default_rng(semilla))
    pa = int(y[listas["A"][:K]].sum()) / K
    out = {}
    for n, idx in listas.items():
        k = int(y[idx[:K]].sum())
        out[n] = {"precision": k / K, "con_sancion": k, "rango": ic(bs[n]), "carga_mediana": carga(X, idx[:K], desde, hasta)}
        if n != "A":
            out[n]["dif_vs_A"] = {"dif": k / K - pa, "rango": ic(bs[n] - bs["A"])}
    return out


def mostrar(titulo, res):
    print(f"\n{titulo}")
    for n, r in res.items():
        extra = (f"; frente a A {r['dif_vs_A']['dif']:+.2%} [{r['dif_vs_A']['rango'][0]:+.2%} a "
                 f"{r['dif_vs_A']['rango'][1]:+.2%}]") if n != "A" else ""
        print(f"  {n}: {r['precision']:.2%} [{r['rango'][0]:.2%} a {r['rango'][1]:.2%}] carga {r['carga_mediana']}{extra}",
              flush=True)




def agregar_riesgo(X, desde, hasta):
    """Riesgo de las entidades a las que vende: tasa de proveedores con alguna sanción hasta el cierre de la ventana,
    encogida con Bayes empírico (Beta-Binomial por momentos); por proveedor, la máxima entre sus entidades."""
    fin = pd.Timestamp(f"{hasta}-12-31")
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta)][["ruc_entidad", "ruc"]].drop_duplicates()
    v["sanc"] = v["ruc"].isin(set(s.loc[s["inicio"] <= fin, "ruc"])).astype(int)
    e = v.groupby("ruc_entidad")["sanc"].agg(["sum", "size"])
    r = e.loc[e["size"] >= 5, "sum"] / e.loc[e["size"] >= 5, "size"]
    m, var = r.mean(), r.var()
    k = max(m * (1 - m) / var - 1, 1.0)
    a, b = m * k, (1 - m) * k
    e["riesgo"] = (e["sum"] + a) / (e["size"] + a + b)
    X["riesgo_entidad"] = v.merge(e["riesgo"], left_on="ruc_entidad", right_index=True).groupby("ruc")["riesgo"].max() \
        .reindex(X.index).fillna(m)
    return X


def entrenar_bayes(X, y, cols_alerta, cols_otras):
    """Logística con previa normal: media en la gravedad legal (pendiente común b por punto) para las alertas y 0 para
    lo demás. La precisión de la previa (lam) se elige con validación cruzada de 3 partes dentro del entrenamiento."""
    fil = muestra(y)
    Za = X.iloc[fil][cols_alerta].to_numpy(float)
    sc = StandardScaler().fit(X.iloc[fil][cols_otras])
    Zo = sc.transform(X.iloc[fil][cols_otras])
    Z = np.hstack([Za, Zo])
    yy = y[fil]
    b = LogisticRegression(max_iter=500).fit(Za.sum(axis=1, keepdims=True), yy).coef_[0][0]  # pendiente por punto
    w0 = np.concatenate([np.full(Za.shape[1], b), np.zeros(Zo.shape[1])])

    def ajustar(Zt, yt, lam):
        def f(w):
            z = w[0] + Zt @ w[1:]
            nll = np.sum(np.logaddexp(0, z) - yt * z)
            d = w[1:] - w0
            g0 = 1 / (1 + np.exp(-z)) - yt
            return nll + lam / 2 * d @ d, np.concatenate([[g0.sum()], Zt.T @ g0 + lam * d])
        return minimize(f, np.concatenate([[0.0], w0]), jac=True, method="L-BFGS-B").x

    folds = StratifiedKFold(3, shuffle=True, random_state=1)
    mejor = max([0.1, 1.0, 10.0, 100.0, 1000.0], key=lambda lam: np.mean([
        roc_auc_score(yy[te], Z[te] @ ajustar(Z[tr], yy[tr], lam)[1:]) for tr, te in folds.split(Z, yy)]))
    w = ajustar(Z, yy, mejor)
    coef = {c: round(float(v), 3) for c, v in zip(cols_alerta + cols_otras, w[1:])}

    def pred(Q):
        return Q[cols_alerta].to_numpy(float) @ w[1:1 + len(cols_alerta)] + sc.transform(Q[cols_otras]) @ w[1 + len(cols_alerta):]
    return pred, {"lam": mejor, "pendiente_previa": round(float(b), 3), "coeficientes": coef}


def ajustar_todo(Xtr, ytr):
    mods, info = {}, {}
    mods["I"], info["I"] = entrenar_l1(Xtr, ytr, CARTAS + ANTEC)
    mods["M"], info["M"] = entrenar_bayes(Xtr, ytr, CARTAS, ANTEC)
    mods["N"], info["N"] = entrenar_l1(Xtr, ytr, CARTAS + ANTEC + ["riesgo_entidad"])
    mods["P"], info["P"] = entrenar_gb(Xtr, ytr, CARTAS + ANTEC + INTENS + TAM + ["riesgo_entidad"])
    return mods, info


def listas4(X, mods):
    p = X[CARTAS].sum(axis=1).to_numpy()
    li = {"A": orden(p, X["monto"])}
    for n, f in mods.items():
        li[n] = orden(f(X), X["monto"])
    rango = {n: np.empty(len(X)) for n in ("A", "I")}
    for n in rango:
        rango[n][li[n]] = np.arange(len(X))
    li["O"] = orden(1 / (60 + rango["A"]) + 1 / (60 + rango["I"]), X["monto"])  # fusión de rangos recíprocos
    return li


def evaluar_ref(X, y, listas, desde, hasta, semilla, ref="I"):
    bs = bootstrap(y, listas, np.random.default_rng(semilla))
    pr = int(y[listas[ref][:K]].sum()) / K
    out = {}
    for n, idx in listas.items():
        k = int(y[idx[:K]].sum())
        out[n] = {"precision": k / K, "con_sancion": k, "rango": ic(bs[n]), "carga_mediana": carga(X, idx[:K], desde, hasta)}
        if n != ref:
            out[n]["dif_vs_ref"] = {"dif": k / K - pr, "rango": ic(bs[n] - bs[ref])}
    return out


def mostrar4(titulo, res):
    print(f"\n{titulo}")
    for n, r in res.items():
        d = r.get("dif_vs_ref")
        extra = f"; frente a I {d['dif']:+.2%} [{d['rango'][0]:+.2%} a {d['rango'][1]:+.2%}]" if d else ""
        print(f"  {n}: {r['precision']:.2%} [{r['rango'][0]:.2%} a {r['rango'][1]:.2%}] carga {r['carga_mediana']}{extra}",
              flush=True)


salida = {}
Xtr, ytr = ventana(2018, 2019, 2020, 2022)
Xva, yva = ventana(2020, 2021, 2022, 2024)
Xtr, Xva = agregar_riesgo(Xtr, 2018, 2019), agregar_riesgo(Xva, 2020, 2021)
mods, info = ajustar_todo(Xtr, ytr)
for n, v in info.items():
    print(f"  {n}: {v}", flush=True)
val = evaluar_ref(Xva, yva, listas4(Xva, mods), 2020, 2021, 41)
mostrar4("VALIDACIÓN (alertas 2020-2021, sanciones 2022-2023)", val)
cand = [n for n, r in val.items() if n in ("M", "N", "O", "P") and r["dif_vs_ref"]["rango"][0] > 0 and r["carga_mediana"] <= 5]
elegida = max(cand, key=lambda n: val[n]["precision"]) if cand else "I"
print(f"  regla: candidatas {cand}; elegida {elegida}", flush=True)
salida["validacion"] = {"filas": val, "modelos": info, "elegida": elegida}

Xtr2, ytr2 = ventana(2018, 2021, 2022, 2024)
Xte, yte = ventana(2022, 2023, 2024)
Xtr2, Xte = agregar_riesgo(Xtr2, 2018, 2021), agregar_riesgo(Xte, 2022, 2023)
mods2, info2 = ajustar_todo(Xtr2, ytr2)
li = {n: v for n, v in listas4(Xte, mods2).items() if n in ("A", "I", elegida)}
salida["prueba"] = evaluar_ref(Xte, yte, li, 2022, 2023, 43)
salida["prueba_modelos"] = {n: info2[n] for n in info2 if n in ("I", elegida)}
mostrar4("PRUEBA (alertas 2022-2023, sanciones desde 2024)", salida["prueba"])

Xtr3, ytr3 = ventana(2020, 2023, 2024, 2026)
X26, y26 = ventana(2024, 2025, 2026)
Xtr3, X26 = agregar_riesgo(Xtr3, 2020, 2023), agregar_riesgo(X26, 2024, 2025)
mods3, _ = ajustar_todo(Xtr3, ytr3)
li = {n: v for n, v in listas4(X26, mods3).items() if n in ("A", "I", elegida)}
salida["confirmacion_2026"] = evaluar_ref(X26, y26, li, 2024, 2025, 47)
mostrar4("CONFIRMACIÓN (alertas 2024-2025, sanciones de 2026)", salida["confirmacion_2026"])
(SALIDA / "mejoras_ronda4.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nlisto en {time.time() - t0:.0f} s")
