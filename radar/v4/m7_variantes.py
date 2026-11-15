"""Prueba de modelos 7 (HALLAZGOS 160, sección 27): lambdarank por entidad-año, margen con el tamaño e historia como indicador. Fijado antes de correr.

    python -I -X utf8 m7_variantes.py elegir
    python -I -X utf8 m7_variantes.py confirmar
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
from m1_d1_configuracion import ajustar, preparar_te  # noqa: E402
from m2_objetivo import etiqueta_ajuste, mitad  # noqa: E402
from m6_proveedor import regla  # noqa: E402
from medidas import fmt, medir  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
ARBOLES = (50, 100, 200)
CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
GANANCIA = {0: 0, 1: 1, 2: 3, 3: 7}
LIMITE_GRUPO = 10_000
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def tamano(d):
    return np.log1p(d[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1)).to_numpy()


def exp_con_tamano(d):
    return np.column_stack([exposicion(d), tamano(d)])


def lgb_caso(tr, te, cols, arboles, fx=exposicion, semillas=range(10)):
    """Igual que m1.ajustar, con otra función de exposición para el margen (V3) y con la matriz dada."""
    w = np.where(tr["y"], tr["peso"].clip(lower=0.25), 1.0)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(fx(tr), tr["y"], sample_weight=w)
    m_tr, m_te = lr.decision_function(fx(tr)), lr.decision_function(fx(te))
    T = Transformador(tr, cols)
    Xtr, Xte = T(tr), T(te)
    cat = [c for c in CATEGORICAS if c in Xtr.columns]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                  monotone_constraints=monotonia(Xtr.columns), monotone_constraints_method="advanced", verbose=-1, num_threads=12)
    ds = lgb.Dataset(Xtr, tr["y"].to_numpy(), weight=w, init_score=m_tr, categorical_feature=cat, free_raw_data=False)
    out = {a: [] for a in arboles}
    for s in semillas:
        b = lgb.train({**params, "seed": s}, ds, num_boost_round=max(arboles))
        for a in arboles:
            out[a].append(m_te + b.predict(Xte, raw_score=True, num_iteration=a))
    return {a: np.vstack(v).mean(axis=0) for a, v in out.items()}


def lambdarank(tr, rel, te, cols, arboles, semillas=range(10)):
    """Ordena dentro de cada entidad-año: solo aprende de los grupos con algún positivo y con más de un caso."""
    lr = LogisticRegression(C=1.0, max_iter=500).fit(exposicion(tr), rel > 0)
    m_te = lr.decision_function(exposicion(te))
    g = tr["ruc_entidad"].astype(str) + "|" + tr["anio"].astype(str)
    # LightGBM no admite más de 10 000 filas por grupo: las entidades-año mayores se parten al azar (160, sección 32)
    al_azar = pd.Series(np.random.default_rng(0).random(len(g)), index=g.index).groupby(g.to_numpy()).rank(method="first").astype(int) - 1
    parte = al_azar // LIMITE_GRUPO
    paso(f"      entidades-año partidas por tamaño: {int(g[parte > 0].nunique())}")
    g = (g + "#" + parte.astype(str)).to_numpy()
    s = pd.DataFrame({"g": g, "r": rel})
    util = s.groupby("g")["r"].transform("max").gt(0).to_numpy() & s.groupby("g")["r"].transform("size").gt(1).to_numpy()
    idx = np.flatnonzero(util)
    idx = idx[np.argsort(g[idx], kind="stable")]
    t = tr.iloc[idx]
    gr = pd.Series(g[idx]).value_counts(sort=False).to_numpy()
    m_tr = lr.decision_function(exposicion(t))
    T = Transformador(tr, cols)
    Xtr, Xte = T(t), T(te)
    cat = [c for c in CATEGORICAS if c in Xtr.columns]
    params = dict(objective="lambdarank", lambdarank_truncation_level=10, label_gain=[0, 1, 3, 7], learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, feature_fraction=0.8, lambda_l2=1.0,
                  monotone_constraints=monotonia(Xtr.columns), monotone_constraints_method="advanced", verbose=-1, num_threads=12)
    ds = lgb.Dataset(Xtr, rel[idx].astype(int), group=gr, init_score=m_tr, categorical_feature=cat, free_raw_data=False)
    out = {a: [] for a in arboles}
    for sem in semillas:
        b = lgb.train({**params, "seed": sem}, ds, num_boost_round=max(arboles))
        for a in arboles:
            out[a].append(m_te + b.predict(Xte, raw_score=True, num_iteration=a))
    paso(f"      lambdarank: {len(t):,} casos en {len(gr):,} grupos")
    return {a: np.vstack(v).mean(axis=0) for a, v in out.items()}


def indicadores(d):
    h = [c for c in d.columns if c.startswith(("h_prov_", "h_ent_", "h_rel_"))]
    return d.assign(**{c: (d[c].astype(float) > 0).astype(float) for c in h})


def relevancia(H, tr, corte, y):
    et = H.etiqueta_amplia(tr, corte, ventana_anios=None)
    return np.where(y, nivel(et["fuentes"]), 0).astype(int).clip(0, 3)


def variantes(H, tr, te, cols, corte, arboles=ARBOLES, cuales=("V1", "V2", "V3", "V4")):
    """{nombre_árboles: puntaje}."""
    r = {}
    y = tr["y"].to_numpy(dtype=bool)
    if "V1" in cuales:
        r |= {f"V1_{a}": s for a, s in lambdarank(tr, y.astype(int), te, cols, arboles).items()}
        paso("   V1 lista")
    if "V2" in cuales:
        r |= {f"V2_{a}": s for a, s in lambdarank(tr, relevancia(H, tr, corte, y), te, cols, arboles).items()}
        paso("   V2 lista")
    if "V3" in cuales:
        r |= {f"V3_{a}": s for a, s in lgb_caso(tr, te, cols, arboles, exp_con_tamano).items()}
        paso("   V3 lista")
    if "V4" in cuales:
        r |= {f"V4_{a}": s for a, s in lgb_caso(indicadores(tr), indicadores(te), cols, arboles).items()}
        paso("   V4 lista")
    return r


def elegir():
    H = Hechos(conocimiento="publicacion")
    c116 = [c for c in lista_121() if c not in NUEVAS]
    x = familia_a_la_decision(cargar_x([2018, 2019]), 2019)
    hist = H.historia(x)
    x = pd.concat([x, hist], axis=1)
    x = x[universo(x, hist)].copy()
    tr, te = x[x["anio"] == 2018].copy(), x[x["anio"] == 2019].copy()
    tr = tr[~H.etiqueta_amplia(tr, "2020-12-31")["conocido_al_cierre"].to_numpy()].reset_index(drop=True)
    te = te.join(H.etiqueta_amplia(te, "2021-12-31"))
    te = preparar_te(te[~te["conocido_al_cierre"]].reset_index(drop=True))
    y, peso = etiqueta_ajuste(H, tr, "T2", "2021-12-31")
    tr["y"], tr["peso"] = y, peso
    m_tr, m_te = mitad(tr["ruc"]), mitad(te["ruc"])
    partes = {}
    for f in (0, 1):
        t, s = tr[m_tr == f].reset_index(drop=True), te[m_te != f].copy()
        paso(f"mitad {f}: ajuste {len(t):,}, {int(t['y'].sum()):,} positivos; prueba {len(s):,}")
        s["s_D1"] = ajustar(t, s, c116, (50,), True)[50].mean(axis=0)
        for n, v in variantes(H, t, s, c116, "2021-12-31").items():
            s[f"s_{n}"] = v
        partes[f + 1] = s
    nombres = [c[2:] for c in partes[1].columns if c.startswith("s_") and c[2:] == "D1" or c[:3] == "s_V"]
    r = {yy: medir(partes, yy, {n: f"s_{n}" for n in nombres} | {"tamano": "s_tamano"}, ["tamano"], n_boot=100)["metodos"] for yy in ("y", "y_S")}
    filas = {}
    for n in nombres:
        py, ps = r["y"][n]["precision"], r["y_S"][n]["precision"]
        filas[n] = {"precision_y": py, "precision_S": ps, "media_geometrica": float(np.sqrt(py * ps))}
        paso(f"   {n}: {filas[n]}")
    mejor = max((n for n in nombres if n != "D1"), key=lambda n: filas[n]["media_geometrica"])
    pasa = filas[mejor]["media_geometrica"] >= 1.01 * filas["D1"]["media_geometrica"]
    (OUT / "m7_elegir.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "m7_elegido.json").write_text(json.dumps({"candidata": mejor if pasa else None, "mejor": mejor}), encoding="utf-8")
    paso(f"ELEGIDO: {mejor if pasa else 'ninguno'} (mejor {mejor})")


def confirmar():
    el = json.loads((OUT / "m7_elegido.json").read_text(encoding="utf-8"))["candidata"]
    if not el:
        paso("no pasa a confirmar")
        return
    assert not (OUT / "m7_confirmar.json").exists(), "la confirmación se corre una sola vez"
    vig_f = OUT / "vigente.json"
    vig = json.loads(vig_f.read_text(encoding="utf-8")) if vig_f.exists() else {"archivo": "puntajes_m2", "col": "s_m2", "nombre": "T2"}
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cl, ar = el.split("_")
    cortes = {}
    for a, anios in CORTES.items():
        tr, te = armar_corte(H, anios, a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        y, peso = etiqueta_ajuste(H, tr, "T2", f"{a}-12-31")
        tr["y"], tr["peso"] = y, peso
        v = variantes(H, tr, te, c116, f"{a}-12-31", (int(ar),), (cl,))
        d = te[K].astype({"ruc_entidad": str, "ruc": str, "anio": int}).copy()
        d["s_m7"] = v[el]
        d.to_parquet(OUT / f"puntajes_m7_{a}.parquet", index=False)
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        n0 = len(b)
        b = b.merge(d, on=K, how="inner").merge(pd.read_parquet(OUT / f"{vig['archivo']}_{a}.parquet", columns=K + [vig["col"]]).astype({"ruc_entidad": str, "ruc": str, "anio": int}), on=K, how="left")
        assert len(b) == n0, "la prueba no son los mismos casos"
        b["y"] = b["y"].astype(bool)
        b["y_S"] = b["y"] & (b["nivel"] == 2)
        cortes[a] = b
        paso(f"corte {a} listo")
    M = {"m7": "s_m7", "vig": vig["col"], "tamano": "s_tamano_proveedor"}
    r = {yy: medir(cortes, yy, M, ["vig", "tamano"]) for yy in ("y", "y_S")}
    g = regla(r["y"]["metodos"]["m7"]["frente_a"]["vig"], r["y_S"]["metodos"]["m7"]["frente_a"]["vig"])
    res = {"candidata": el, "frente_a": vig["nombre"], "adopta": g, "precision": {yy: r[yy]["metodos"]["m7"]["precision"] for yy in r}, "frente_tamano": {yy: fmt(r[yy]["metodos"]["m7"]["frente_a"]["tamano"]) for yy in r},
           "por_anio": {yy: {k: v["razon"] for k, v in r[yy]["metodos"]["m7"]["frente_a"]["vig"]["por_anio"].items()} for yy in r}}
    (OUT / "m7_confirmar.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if g["se_adopta"]:
        (OUT / "vigente.json").write_text(json.dumps({"archivo": "puntajes_m7", "col": "s_m7", "nombre": el}), encoding="utf-8")
    paso(f"{res}")


if __name__ == "__main__":
    {"elegir": elegir, "confirmar": confirmar}[sys.argv[1]]()
    paso("listo")
