"""Prueba de modelos 1 (HALLAZGOS 160, sección 19): D1 con número de árboles fijo y promedio de semillas. Fijado antes de correr.

    python -I -X utf8 m1_d1_configuracion.py elegir       # solo 2018 → 2019: resultados/m1_elegir.json
    python -I -X utf8 m1_d1_configuracion.py confirmar    # una vez, en 2020, 2021 y 2023: puntajes_m1_<año> y m1_confirmar.md
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_linea_base import exposicion, monotonia  # noqa: E402
from d15_variables import NUEVAS, lista_121, nivel  # noqa: E402
from etiquetas import K, Hechos  # noqa: E402
from evaluacion import CATEGORICAS, Transformador, armar_corte, cargar_x, familia_a_la_decision, universo  # noqa: E402
from medidas import fmt, medir  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
ARBOLES = (50, 100, 200, 300, 600)
SEMILLAS = range(10)
CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def ajustar(tr, te, cols, arboles, categoricas, semillas=SEMILLAS):
    """Puntaje de la prueba por semilla y número de árboles: {arboles: matriz (semillas × casos)}. Margen de exposición, pesos y
    parámetros iguales a d1; sin parada temprana."""
    w = np.where(tr["y"], tr["peso"].clip(lower=0.25), 1.0)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(exposicion(tr), tr["y"], sample_weight=w)
    m_tr, m_te = lr.decision_function(exposicion(tr)), lr.decision_function(exposicion(te))
    T = Transformador(tr, cols)
    Xtr, Xte = T(tr), T(te)
    cat = [c for c in CATEGORICAS if c in Xtr.columns] if categoricas else "auto"
    params = dict(objective="binary", learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, bagging_fraction=0.5,
                  bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, monotone_constraints=monotonia(Xtr.columns),
                  monotone_constraints_method="advanced", verbose=-1, num_threads=12)
    ds = lgb.Dataset(Xtr, tr["y"].to_numpy(), weight=w, init_score=m_tr, categorical_feature=cat, free_raw_data=False)
    out = {a: [] for a in arboles}
    for s in semillas:
        b = lgb.train({**params, "seed": s}, ds, num_boost_round=max(arboles))
        for a in arboles:
            out[a].append(m_te + b.predict(Xte, raw_score=True, num_iteration=a))
        paso(f"      semilla {s} lista")
    return {a: np.vstack(v) for a, v in out.items()}


def preparar_te(te):
    te = te.assign(grupo=te["ruc_entidad"].astype(str) + "|" + te["anio"].astype(str), ruc_entidad=te["ruc_entidad"].astype(str), ruc=te["ruc"].astype(str))
    n = nivel(te["fuentes"])
    te["y_P"], te["y_S"], te["y_ABC"] = te["y"].to_numpy() & (n == 1), te["y"].to_numpy() & (n == 2), te["y"].to_numpy() & (n == 3)
    te["y"] = te["y"].astype(bool)
    te["s_tamano"] = np.log1p(te[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1)).to_numpy()
    return te


def elegir():
    H = Hechos(conocimiento="publicacion")
    c116 = [c for c in lista_121() if c not in NUEVAS]
    x = familia_a_la_decision(cargar_x([2018, 2019]), 2019)
    hist = H.historia(x)
    x = pd.concat([x, hist], axis=1)
    x = x[universo(x, hist)].copy()
    tr, te = x[x["anio"] == 2018].copy(), x[x["anio"] == 2019].copy()
    tr = tr.join(H.etiqueta_amplia(tr, "2020-12-31"))
    te = te.join(H.etiqueta_amplia(te, "2021-12-31"))
    tr, te = tr[~tr["conocido_al_cierre"]].reset_index(drop=True), te[~te["conocido_al_cierre"]].reset_index(drop=True)
    te = preparar_te(te)
    paso(f"ajuste 2018: {len(tr):,} casos, {int(tr['y'].sum()):,} positivos; prueba 2019: {len(te):,} casos, {int(te['y'].sum()):,} positivos")
    filas = {}
    for cat in (False, True):
        paso(f"categóricas declaradas: {cat}")
        P = ajustar(tr, te, c116, ARBOLES, cat)
        for a in ARBOLES:
            te["s_m"] = P[a].mean(axis=0)
            d = {y: medir({2019: te}, y, {"m": "s_m", "tamano": "s_tamano"}, ["tamano"], n_boot=500) for y in ("y", "y_S", "y_P", "y_ABC")}
            por_sem = []
            for i in range(P[a].shape[0]):
                te["s_u"] = P[a][i]
                por_sem.append(medir({2019: te}, "y", {"u": "s_u", "tamano": "s_tamano"}, ["tamano"], n_boot=1)["metodos"]["u"]["frente_a"]["tamano"]["razon"])
            f = {"precision": d["y"]["metodos"]["m"]["precision"], "frente_tamano": fmt(d["y"]["metodos"]["m"]["frente_a"]["tamano"]),
                 "S": fmt(d["y_S"]["metodos"]["m"]["frente_a"]["tamano"]), "P": fmt(d["y_P"]["metodos"]["m"]["frente_a"]["tamano"]),
                 "ABC": fmt(d["y_ABC"]["metodos"]["m"]["frente_a"]["tamano"]), "semilla_min": min(por_sem), "semilla_max": max(por_sem)}
            filas[f"{a}|{cat}"] = f
            paso(f"   {a} árboles, categóricas {cat}: {f}")
    (OUT / "m1_elegir.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    sin = {a: filas[f"{a}|False"]["precision"] for a in ARBOLES}
    con = {a: filas[f"{a}|True"]["precision"] for a in ARBOLES}
    usa_cat = max(con.values()) >= 1.01 * max(sin.values())
    base = con if usa_cat else sin
    arb = min(a for a, v in base.items() if v >= 0.99 * max(base.values()))
    paso(f"ELEGIDO por la regla: {arb} árboles; categóricas declaradas: {usa_cat}")
    (OUT / "m1_elegido.json").write_text(json.dumps({"arboles": arb, "categoricas": bool(usa_cat)}), encoding="utf-8")


def confirmar():
    assert not (OUT / "m1_confirmar.json").exists(), "la confirmación se corre una sola vez"
    el = json.loads((OUT / "m1_elegido.json").read_text(encoding="utf-8"))
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cortes = {}
    for a, anios in CORTES.items():
        tr, te = armar_corte(H, anios, a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        paso(f"corte {anios} → {a}: ajuste {len(tr):,} casos, {int(tr['y'].sum()):,} positivos; prueba {len(te):,}")
        P = ajustar(tr, te, c116, (el["arboles"],), el["categoricas"])[el["arboles"]]
        d = te[K].astype({"ruc_entidad": str, "ruc": str, "anio": int}).copy()
        d["s_D1_fijo"] = P.mean(axis=0)
        for i in range(P.shape[0]):
            d[f"s_sem{i}"] = P[i]
        d.to_parquet(OUT / f"puntajes_m1_{a}.parquet", index=False)
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        n0 = len(b)
        b = b.merge(d, on=K, how="inner")
        assert len(b) == n0, f"la prueba no son los mismos casos: {len(b)} frente a {n0}"
        for c in ("y",):
            b[c] = b[c].astype(bool)
        b["y_P"], b["y_S"], b["y_ABC"] = b["y"] & (b["nivel"] == 1), b["y"] & (b["nivel"] == 2), b["y"] & (b["nivel"] == 3)
        cortes[a] = b
    M = {"D1_fijo": "s_D1_fijo", "D1_linea_base": "s_D1_binario", "tamano": "s_tamano_proveedor"}
    res = {"elegido": el}
    L = [f"# Prueba de modelos 1: D1 con {el['arboles']} árboles fijos y 10 semillas; categóricas declaradas: {el['categoricas']}", ""]
    for nom, y in (("Señal adversa", "y"), ("Solo penalidades", "y_P"), ("Sanciones del Tribunal", "y_S"), ("A, B o C en la ventana", "y_ABC")):
        r = medir(cortes, y, M, ["tamano", "D1_linea_base"])
        res[y] = r
        L += [f"## {nom}", "", "| Método | Precisión en 5 | Frente al tamaño | Por año | Frente a D1 de la línea base | Por año |", "|---|---|---|---|---|---|"]
        for m, d in r["metodos"].items():
            ta, ba = d["frente_a"].get("tamano"), d["frente_a"].get("D1_linea_base")
            pa = lambda q: ", ".join("nd" if x["razon"] is None else f"{x['razon']:.2f}" for x in q["por_anio"].values()) if q else "—"  # noqa: E731
            L.append(f"| {m} | {d['precision']:.4f} | {fmt(ta) if ta else '—'} | {pa(ta)} | {fmt(ba) if ba else '—'} | {pa(ba)} |")
        L.append("")
    sem = []
    for i in SEMILLAS:
        r = medir(cortes, "y", {"u": f"s_sem{i}", "tamano": "s_tamano_proveedor"}, ["tamano"], n_boot=1)
        sem.append(r["metodos"]["u"]["frente_a"]["tamano"]["razon"])
    res["por_semilla"] = sem
    L += [f"Una sola semilla, frente al tamaño, señal adversa: de {min(sem):.3f} a {max(sem):.3f}.", ""]
    g = res["y"]["metodos"]["D1_fijo"]["frente_a"]["D1_linea_base"]
    sube_anios = sum(1 for x in g["por_anio"].values() if x["razon"] and x["razon"] > 1)
    s = res["y_S"]["metodos"]["D1_fijo"]["frente_a"]["D1_linea_base"]["razon"]
    adopta = bool(g["razon"] > 1 and sube_anios >= 2 and s >= 0.95)
    res["adopta"] = {"razon": g["razon"], "cortes_que_suben": sube_anios, "S": s, "se_adopta": adopta}
    L += [f"**Regla de adopción:** frente a la línea base {g['razon']:.3f}; sube en {sube_anios} de 3 cortes; en sanciones {s:.3f}. **Se adopta: {adopta}.**"]
    (OUT / "m1_confirmar.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "m1_confirmar.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    {"elegir": elegir, "confirmar": confirmar}[sys.argv[1]]()
    paso("listo")
