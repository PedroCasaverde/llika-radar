"""Prueba de modelos 4 (HALLAZGOS 160, sección 24): los modelos simples con la data nueva, en las cuatro tareas. Fijado antes de correr.

Modelos: Isolation Forest (sin etiquetas), índice simple de banderas (CRI), Bayes ingenuo con tramos y regresión logística condicional con los estadísticos forenses.
Cada uno solo y fusionado (RRF) con el monto y con el tamaño; en impedimentos, además, dentro de la alerta.
Tareas y referencia: señal adversa y sanciones, D1 vigente; hallazgos por caso, el monto; impedimentos, la alerta y luego el monto.

    python -I -X utf8 m4_simples.py elegir ; python -I -X utf8 m4_simples.py confirmar
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alerta_impedimentos import impedimento_real  # noqa: E402
from d10_regresion import variables  # noqa: E402
from d12_simples2 import BayesTramos, anomalia, banderas  # noqa: E402
from d15_variables import NUEVAS, lista_121, nivel  # noqa: E402
from d17_configuracion import alerta  # noqa: E402
from etiquetas import HOY_EXPLORACION, K, Hechos  # noqa: E402
from evaluacion import armar_corte, cargar_x, familia_a_la_decision, universo  # noqa: E402
from m1_d1_configuracion import ajustar as ajustar_d1  # noqa: E402
from m2_objetivo import etiqueta_ajuste  # noqa: E402
from medidas import medir  # noqa: E402

warnings.filterwarnings("ignore")
V4 = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4"
OUT = V4 / "resultados"
CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
TAREAS = ("y", "y_S", "y_rev", "y_imp_real")
ESTAD = ["s_obj_frac_parte", "s_redondo_100", "s_redondo_1000", "s_monto_repetido", "s_precio_z", "s_precio_atipico", "s_fecha_inusual", "s_benford_1d"]
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def con_simples(x):
    a = sorted(x["anio"].astype(int).unique())
    s = pd.read_parquet(V4 / "x_simples.parquet", filters=[("anio", "in", [int(v) for v in a])])
    s = s.astype({"ruc_entidad": str, "ruc": str, "anio": int})
    x = x.astype({"ruc_entidad": str, "ruc": str, "anio": int}).merge(s, on=K, how="left")
    return x


def fx(d):
    X = variables(d)
    for c in ESTAD:
        X[c] = d[c].astype(float).fillna(float(d[c].median()) if c == "s_benford_1d" else 0.0)
    return X


def logit_condicional(tr, y, cont=20, l2=1e-3):
    con = pd.Series(y).groupby(tr["grupo"].to_numpy()).transform("any").to_numpy()
    d, yy = tr[con], y[con]
    X = fx(d)
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).to_numpy()
    rng = np.random.default_rng(0)
    filas = []
    for _, idx in pd.Series(np.arange(len(d))).groupby(d["grupo"].to_numpy()).groups.items():
        idx = np.asarray(idx)
        pos, neg = idx[yy[idx]], idx[~yy[idx]]
        if len(neg):
            for p in pos:
                filas.append(np.concatenate([[p], rng.choice(neg, size=cont, replace=len(neg) < cont)]))
    S = np.stack(filas)
    Zs = Z[S]
    n = len(S)

    def perdida(b):
        s = Zs @ b
        m = s.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
        p = np.exp(s - lse[:, None])
        return -(s[:, 0] - lse).sum() + l2 * n * (b @ b), -(Zs[:, 0, :] - (p[:, :, None] * Zs).sum(axis=1)).sum(axis=0) + 2 * l2 * n * b
    r = minimize(perdida, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
    return lambda t: ((fx(t) - mu) / sd).to_numpy() @ r.x


def cri(tr, y):
    """Promedio de las banderas que, en el ajuste y dentro de la entidad-año, están más encendidas en los positivos (t > 1,64)."""
    B = banderas(tr)
    D = B - B.groupby(tr["grupo"].to_numpy()).transform("mean")
    d = D[y]
    t = d.mean() / (d.std().replace(0, 1) / np.sqrt(max(len(d), 1)))
    usa = [c for c in B.columns if t[c] > 1.64] or list(B.columns)
    return (lambda te: banderas(te)[usa].mean(axis=1).to_numpy()), usa


def rrf2(s, b, g, k=60):
    r = pd.DataFrame({"g": g, "s": s, "b": b})
    return (1 / (k + r.groupby("g")["s"].rank(ascending=False, method="average")) + 1 / (k + r.groupby("g")["b"].rank(ascending=False, method="average"))).to_numpy()


def rango(s, g):
    return pd.Series(np.asarray(s)).groupby(np.asarray(g)).rank(pct=True).to_numpy()


def modelos(tr, ys, te):
    """Puntajes de la prueba por tarea y modelo."""
    out = {}
    s_if = -IsolationForest(n_estimators=200, max_samples=256, random_state=0, n_jobs=-1).fit(anomalia(te)).score_samples(anomalia(te))
    for t, y in ys.items():
        f_cri, usa = cri(tr, y)
        con = pd.Series(y).groupby(tr["grupo"].to_numpy()).transform("any").to_numpy()
        byn = BayesTramos(fx(tr[con]), y[con])
        lc = logit_condicional(tr, y)
        out[t] = {"if": s_if, "cri": f_cri(te), "bayes": byn(fx(te)), "logit": lc(te)}
        paso(f"   {t}: {int(y.sum()):,} positivos de ajuste; banderas del CRI {len(usa)}")
    return out


def variantes(s, te, al):
    g, m, ta = te["grupo"].to_numpy(), te["s_monto"].to_numpy(dtype=float), te["s_tamano"].to_numpy(dtype=float)
    return {"solo": s, "con_monto": rrf2(s, m, g), "con_tamano": rrf2(s, ta, g), "alerta_luego": 100 * al + rango(s, g)}


def tablas(cortes, t, ref, nombres):
    return medir(cortes, t, {n: f"s_{n}" for n in nombres} | {"ref": ref}, ["ref"], n_boot=1000 if len(cortes) > 1 else 200)["metodos"]


def elegir():
    H = Hechos(conocimiento="publicacion")
    c116 = [c for c in lista_121() if c not in NUEVAS]
    x = familia_a_la_decision(cargar_x([2018, 2019]), 2019)
    hist = H.historia(x)
    x = con_simples(pd.concat([x, hist], axis=1)[lambda d: universo(d, d[hist.columns])])
    x["grupo"] = x["ruc_entidad"] + "|" + x["anio"].astype(str)
    tr, te = x[x["anio"] == 2018].copy(), x[x["anio"] == 2019].copy()
    tr = tr[~H.etiqueta_amplia(tr, "2020-12-31")["conocido_al_cierre"].to_numpy()].reset_index(drop=True)
    e = H.etiqueta_amplia(te, "2021-12-31")
    te = te[~e["conocido_al_cierre"].to_numpy()].reset_index(drop=True)
    e = H.etiqueta_amplia(te, "2021-12-31")
    te["y"] = e["y"].to_numpy()
    te["y_S"] = te["y"].to_numpy() & (nivel(e["fuentes"]) == 2)
    fu = Hechos.etiqueta(H, te, fecha_corte=HOY_EXPLORACION)["fuentes"].fillna("").astype(str)
    te["y_rev"] = fu.str.contains("A|B").to_numpy()
    te["y_imp_real"] = fu.str.contains("C").to_numpy() & impedimento_real(H, te)
    te["s_monto"] = te["c_monto"].to_numpy(dtype=float)
    te["s_tamano"] = np.log1p(te[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1)).to_numpy()
    al = alerta(te)
    te["s_L0"] = 100 * al + np.log1p(te["c_monto"].astype(float).to_numpy())
    ey, ep = etiqueta_ajuste(H, tr, "T2", "2021-12-31")
    te["s_D1"] = ajustar_d1(tr.assign(y=ey, peso=ep), te, c116, (50,), True)[50].mean(axis=0)
    fu_tr = Hechos.etiqueta(H, tr, fecha_corte="2021-12-31")["fuentes"].fillna("").astype(str)
    et2 = H.etiqueta_amplia(tr, "2021-12-31", ventana_anios=None)
    ys = {"y": ey, "y_S": nivel(et2["fuentes"]) >= 2, "y_rev": fu_tr.str.contains("A|B").to_numpy(), "y_imp_real": fu_tr.str.contains("C").to_numpy() & impedimento_real(H, tr)}
    M = modelos(tr, ys, te)
    REF = {"y": "s_D1", "y_S": "s_D1", "y_rev": "s_monto", "y_imp_real": "s_L0"}
    res, elegidos = {}, {}
    for t in TAREAS:
        nombres = []
        for mod, s in M[t].items():
            for v, sv in variantes(s, te, al).items():
                n = f"{mod}_{v}"
                te[f"s_{n}"] = sv
                nombres.append(n)
        r = tablas({2019: te}, t, REF[t], nombres)
        p0 = r["ref"]["precision"]
        filas = {n: {"precision": r[n]["precision"], "aciertos": r[n]["aciertos"], "frente_ref": r[n]["precision"] / max(p0, 1e-12)} for n in nombres}
        mejor = max(filas, key=lambda n: filas[n]["precision"])
        res[t] = {"referencia": {"precision": p0, "aciertos": r["ref"]["aciertos"]}, "modelos": filas, "mejor": mejor}
        if filas[mejor]["frente_ref"] >= 1.05:
            elegidos[t] = mejor
        top = sorted(filas, key=lambda n: -filas[n]["precision"])[:4]
        paso(f"{t}: referencia {p0:.4f} ({r['ref']['aciertos']:.0f}); " + "; ".join(f"{n} {filas[n]['precision']:.4f}" for n in top) + f"; pasa a confirmar: {elegidos.get(t, 'ninguno')}")
    (OUT / "m4_elegir.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "m4_elegido.json").write_text(json.dumps(elegidos), encoding="utf-8")
    paso(f"ELEGIDOS: {elegidos or 'ninguno pasa a confirmar'}")


def confirmar():
    el = json.loads((OUT / "m4_elegido.json").read_text(encoding="utf-8"))
    if not el:
        paso("nada que confirmar")
        (OUT / "m4_confirmar.json").write_text("{}", encoding="utf-8")
        return
    assert not (OUT / "m4_confirmar.json").exists(), "la confirmación se corre una sola vez"
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    REF = {"y": "s_m2", "y_S": "s_m2", "y_rev": "s_monto", "y_imp_real": "s_alerta_monto"}
    cortes = {}
    for a, anios in CORTES.items():
        tr, te = armar_corte(H, anios, a)
        tr, te = con_simples(tr.reset_index(drop=True)), con_simples(te.reset_index(drop=True))
        corte = f"{a}-12-31"
        ey, _ = etiqueta_ajuste(H, tr, "T2", corte)
        et2 = H.etiqueta_amplia(tr, corte, ventana_anios=None)
        fu = Hechos.etiqueta(H, tr, fecha_corte=corte)["fuentes"].fillna("").astype(str)
        ys = {"y": ey, "y_S": nivel(et2["fuentes"]) >= 2, "y_rev": fu.str.contains("A|B").to_numpy(), "y_imp_real": fu.str.contains("C").to_numpy() & impedimento_real(H, tr)}
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        b = b.merge(pd.read_parquet(OUT / f"puntajes_m2_{a}.parquet", columns=K + ["s_m2"]).astype({"ruc_entidad": str, "ruc": str, "anio": int}), on=K, how="left")
        te = te.merge(b[K + ["s_monto", "s_tamano_proveedor", "alerta"]].rename(columns={"s_tamano_proveedor": "s_tamano"}), on=K, how="inner")
        M = modelos(tr, {t: ys[t] for t in el}, te)
        for t, n in el.items():
            mod, var = n.split("_", 1)
            s = variantes(M[t][mod], te, te["alerta"].to_numpy(dtype=bool))[var]
            b = b.merge(te[K].assign(**{f"s_{t}": s}), on=K, how="left")
        b["y"] = b["y"].astype(bool)
        b["y_S"] = b["y"] & (b["nivel"] == 2)
        b["y_rev"], b["y_imp_real"] = b["y_rev"].astype(bool), b["y_imp_real"].astype(bool)
        cortes[a] = b
        paso(f"corte {a} listo")
    res = {}
    for t, n in el.items():
        r = medir(cortes, t, {"m": f"s_{t}", "ref": REF[t]}, ["ref"])["metodos"]
        g = r["m"]["frente_a"]["ref"]
        ic = g["ic90_dos_vias"] if t in ("y", "y_S") else g["ic90_entidad"]
        sube = sum(1 for x in g["por_anio"].values() if x["razon"] and x["razon"] > 1)
        res[t] = {"modelo": n, "precision": r["m"]["precision"], "precision_ref": r["ref"]["precision"], "razon": g["razon"], "ic90": ic, "por_anio": {k: v["razon"] for k, v in g["por_anio"].items()},
                  "se_adopta": bool(g["razon"] >= 1.05 and ic[0] > 1 and sube >= 2)}
        paso(f"{t}: {res[t]}")
    (OUT / "m4_confirmar.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"elegir": elegir, "confirmar": confirmar}[sys.argv[1]]()
    paso("listo")
