"""Sonda de solo lectura para revisar bibliografia/prompt-mejora-final.md (HALLAZGOS 160, sección 71). Sin API. Solo años de exploración: pesos con 2020 y 2021, medida en 2023.

    python -I -X utf8 sonda_prompt_final.py
Salida: resultados/por_caso/sonda_prompt_final.json. Solo agregados.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import u2_ronda_final as U2  # noqa: E402
from casos import PRUEBA, SALIDA, TIPOS, top  # noqa: E402
from etiquetas import BASE, K  # noqa: E402
sys.argv = _a

assert tuple(PRUEBA) == (2020, 2021, 2023), "solo años de exploración"
AJUSTE, MEDIDA = (2020, 2021), 2023
VERDADES = {"amplia": "y", "impedimento": "y_imp_real", "sancion": "y_S", "penalidad": "y_P", "hallazgo": "y_rev", "del_caso": "y_local"}


def paso(t):
    print(t, flush=True)


def banderas(a):
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "monto", "tope"], filters=[("anio", "==", a)]).astype({"ruc_entidad": str, "ruc": str})
    q = pl.from_pandas(o).sort(["ruc_entidad", "ruc", "fecha", "id"]).with_columns(pl.col("fecha").cast(pl.Datetime("ms")), pl.col("monto").cast(pl.Float64), pl.col("tope").cast(pl.Float64), pl.lit(1).cast(pl.Int64).alias("uno"))
    q = q.with_columns(pl.col("monto").rolling_sum_by("fecha", "30d", closed="right").over(["ruc_entidad", "ruc"]).alias("s"), pl.col("uno").rolling_sum_by("fecha", "30d", closed="right").over(["ruc_entidad", "ruc"]).alias("n"))
    q = q.with_columns(((pl.col("n") >= 2) & (pl.col("s") > pl.col("tope"))).fill_null(False).alias("b30"), (pl.col("monto") >= 0.9 * pl.col("tope")).fill_null(False).alias("cerca"))
    r = q.group_by(["ruc_entidad", "ruc"]).agg(pl.col("b30").any().alias("f30"), pl.col("cerca").any().alias("f_cerca")).to_pandas()
    r["anio"] = a
    paso(f"banderas {a}: {len(o):,} órdenes; casos con 30 días {r['f30'].mean():.3f}; con orden cerca del tope {r['f_cerca'].mean():.3f}")
    return r.astype(TIPOS)


def preparar():
    E = U2.eventos()
    d = U2.casos(PRUEBA, E).merge(pd.concat([pd.read_parquet(SALIDA / f"u2_c_puntajes_{a}.parquet") for a in PRUEBA]).astype(TIPOS), on=K)
    d = d.merge(pd.concat([banderas(a) for a in PRUEBA]), on=K, how="left")
    d[["f30", "f_cerca"]] = d[["f30", "f_cerca"]].fillna(False).astype(float)
    d["y_local"] = d["y_estricta"] | d["y_S_ent"] | d["y_P_ent"]
    sig = lambda s: 1 / (1 + np.exp(-np.clip(s, -30, 30)))  # noqa: E731
    d["pS"], d["pP"] = sig(d["s_S_prov"].to_numpy()), sig(d["s_P_ampl"].to_numpy())
    d["R"] = 1 - (1 - d["pS"]) * (1 - d["pP"])
    d["logit_R"] = np.log(d["R"].clip(1e-12, 1 - 1e-12) / (1 - d["R"].clip(1e-12, 1 - 1e-12)))
    d["logM"] = np.log1p(d["c_monto"].astype(float))
    d["V"] = d["alerta"].astype(float)
    d["compra"] = (d["f30"] + d["f_cerca"]) / 2
    M_p = d.groupby(["ruc", "anio"])["c_monto"].transform("sum")
    d["c_local"] = d["c_monto"] / M_p.clip(lower=1e-9)
    assert set(d["anio"].unique()) == set(PRUEBA)
    return d


def logistica(q, cols, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=500))
    m.fit(q[cols].to_numpy(dtype=float), q[y].to_numpy(dtype=bool))
    esc, lr = m.steps[0][1], m.steps[1][1]
    return m, {c: float(w) for c, w in zip(cols, lr.coef_[0] / esc.scale_)}, {c: float(w) for c, w in zip(cols, lr.coef_[0])}


def metrica(q, col, semilla=0):
    L = top(q, col, semilla=semilla)
    r = {"mostrados": len(L), "entidades_anio": int(L["grupo"].nunique()), "proveedores_anio_mostrados": int(L.groupby(["ruc", "anio"]).ngroups)}
    for n, y in VERDADES.items():
        r[f"aciertos_{n}"] = int(L[y].sum())
    r["precision"] = r["aciertos_amplia"] / max(len(L), 1)
    r["precision_del_caso"] = r["aciertos_del_caso"] / max(len(L), 1)
    r["proveedores_anio_acierto"] = int(L[L["y"]].groupby(["ruc", "anio"]).ngroups)
    r["entidades_anio_con_acierto"] = int(L[L["y"]]["grupo"].nunique())
    return r


def ic_razon(q, a, b, y="y", n=500, semilla=0):
    """Razón de precisión en 5 por caso, a frente a b, con pesos de Poisson de entidades y proveedores."""
    A, B = top(q, a), top(q, b)
    e, ne = pd.factorize(pd.concat([A["ruc_entidad"], B["ruc_entidad"]]))
    p, np_ = pd.factorize(pd.concat([A["ruc"], B["ruc"]]))
    hit = np.r_[A[y].to_numpy(), B[y].to_numpy()].astype(float)
    c = len(A)
    rng = np.random.default_rng(semilla)

    def razon(w):
        pa = (w[:c] * hit[:c]).sum() / max(w[:c].sum(), 1e-12)
        pb = (w[c:] * hit[c:]).sum() / max(w[c:].sum(), 1e-12)
        return pa / pb if pb > 0 else np.nan
    uno = razon(np.ones(len(hit)))
    bs = [razon(rng.poisson(1.0, len(ne))[e] * rng.poisson(1.0, len(np_))[p]) for _ in range(n)]
    return {"razon": float(uno), "ic90": [float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))]}


def main():
    d = preparar()
    paso(f"casos {len(d):,}; por año {d.groupby('anio').size().to_dict()}; entidades-año {d.groupby('anio')['grupo'].nunique().to_dict()}")
    fit, med = d[d["anio"].isin(AJUSTE)], d[d["anio"] == MEDIDA].copy()
    res = {"casos_por_anio": {int(k): int(v) for k, v in d.groupby("anio").size().items()}, "entidades_anio": {int(k): int(v) for k, v in d.groupby("anio")["grupo"].nunique().items()}, "pesos": {}}
    res["composicion"] = {}
    for a, g in d.groupby("anio"):
        pv = g[g["y_P"]].groupby("ruc").size().sort_values(ascending=False)
        res["composicion"][int(a)] = {"casos": len(g), "base_amplia": float(g["y"].mean()), **{v: int(g[v].sum()) for v in ("y", "y_estricta", "y_rev", "y_imp_real", "y_S", "y_P", "y_S_ent", "y_P_ent", "y_local")}, "amplia_sin_verdad_del_caso": int((g["y"] & ~g["y_local"]).sum()), "penalidad_13_mayores": int(pv.head(13).sum()), "alertas": int(g["alerta"].sum())}
    paso(json.dumps(res["composicion"], ensure_ascii=False))

    # listas: la fórmula del prompt y sus variantes. Los pesos solo se ajustan con 2020 y 2021
    extra = {}
    for nom, x in (("x1", lambda q: q["logM"] * q["V"]), ("x2", lambda q: q["logM"] * q["R"]), ("x3", lambda q: q["logM"] * q["compra"]), ("x2c", lambda q: q["logM"] * q["R"] * q["c_local"])):
        extra[nom] = x
    for q in (fit, med, d):
        for nom, x in extra.items():
            q.loc[:, nom] = x(q) if q is not d else x(q)
    fit = d[d["anio"].isin(AJUSTE)]
    configs = {
        "prompt_multiplicativa": ["x1", "x2", "x3"],
        "prompt_con_c_local": ["x1", "x2c", "x3"],
        "suma": ["V", "logit_R", "logM", "f30", "f_cerca"],
    }
    for nom, cols in configs.items():
        m, w, std = logistica(fit, cols, "y")
        d[f"s_{nom}"] = m.decision_function(d[cols].to_numpy(dtype=float))
        res["pesos"][nom] = {"pesos": w, "pesos_estandarizados": std}
        paso(f"{nom}: {w}")
    # dentro de cada entidad-año: solo cuenta el contraste entre casos de la misma entidad
    cols = ["V", "logit_R", "logM", "f30", "f_cerca"]
    dm = fit[cols] - fit.groupby("grupo")[cols].transform("mean")
    lr = LogisticRegression(C=1e6, max_iter=500, fit_intercept=False)
    sd = dm.std().to_numpy()
    lr.fit(dm.to_numpy() / sd, fit["y"].to_numpy(dtype=bool))
    wd = lr.coef_[0] / sd
    d["s_suma_dentro"] = d[cols].to_numpy() @ wd
    res["pesos"]["suma_dentro"] = {"pesos": {c: float(w) for c, w in zip(cols, wd)}, "pesos_estandarizados": {c: float(w) for c, w in zip(cols, lr.coef_[0])}}
    paso(f"suma_dentro: {res['pesos']['suma_dentro']['pesos']}")
    # la alerta primero: regla, sin estimar su peso; el resto con la suma sin V
    resto = ["logit_R", "logM", "f30", "f_cerca"]
    sin_alerta = fit[fit["V"] == 0]
    dm2 = sin_alerta[resto] - sin_alerta.groupby("grupo")[resto].transform("mean")
    sd2 = dm2.std().to_numpy()
    lr2 = LogisticRegression(C=1e6, max_iter=500, fit_intercept=False)
    lr2.fit(dm2.to_numpy() / sd2, sin_alerta["y"].to_numpy(dtype=bool))
    wr = lr2.coef_[0] / sd2
    d["s_resto"] = d[resto].to_numpy() @ wr
    d["s_alerta_primero"] = d["s_resto"] + 1e4 * d["V"]
    res["pesos"]["alerta_primero"] = {"pesos": {c: float(w) for c, w in zip(resto, wr)}, "pesos_estandarizados": {c: float(w) for c, w in zip(resto, lr2.coef_[0])}}
    paso(f"alerta_primero: {res['pesos']['alerta_primero']['pesos']}")
    # el modelo y la alerta, sin regresión: alerta primero y luego el riesgo del proveedor; y el mismo con la suma de los dos riesgos y monto a partes iguales
    d["s_alerta_y_R"] = d["logit_R"] + 1e4 * d["V"]
    rk = lambda c: d.groupby("grupo")[c].rank(pct=True)  # noqa: E731
    d["s_alerta_iguales"] = 1e4 * d["V"] + rk("logit_R") + rk("logM")
    rng = np.random.default_rng(0)
    d["s_azar"] = rng.random(len(d))

    listas = {"azar": "s_azar", "monto": "s_monto", "tamano": "s_tamano_proveedor", "alerta_y_monto": "s_alerta_monto", "modelo_S": "s_S_prov", "modelo_P": "s_P_ampl", "prompt_multiplicativa": "s_prompt_multiplicativa", "prompt_con_c_local": "s_prompt_con_c_local", "suma": "s_suma", "suma_dentro": "s_suma_dentro", "alerta_primero": "s_alerta_primero", "alerta_y_R": "s_alerta_y_R", "alerta_iguales": "s_alerta_iguales"}
    res["medida_2023"] = {}
    res["ajuste_2020_2021"] = {}
    med = d[d["anio"] == MEDIDA]
    fit = d[d["anio"].isin(AJUSTE)]
    for n, c in listas.items():
        res["medida_2023"][n] = metrica(med, c)
        res["ajuste_2020_2021"][n] = metrica(fit, c)
        r = res["medida_2023"][n]
        paso(f"2023 {n}: P@5 {r['precision']:.4f} ({r['aciertos_amplia']} de {r['mostrados']}; {r['proveedores_anio_acierto']} prov-año de {r['proveedores_anio_mostrados']}; {r['entidades_anio_con_acierto']} ent-año); imp {r['aciertos_impedimento']}, S {r['aciertos_sancion']}, P {r['aciertos_penalidad']}, hall {r['aciertos_hallazgo']}, caso {r['aciertos_del_caso']}")
    res["condicional_2023"] = {}
    t = med[med.groupby("grupo")["y"].transform("any")]
    for n in ("monto", "tamano", "suma_dentro", "alerta_primero"):
        res["condicional_2023"][n] = metrica(t, listas[n])
    res["razones_2023"] = {}
    for n in ("prompt_multiplicativa", "suma", "suma_dentro", "alerta_primero", "alerta_y_R", "alerta_iguales"):
        res["razones_2023"][n] = {"frente_monto": ic_razon(med, listas[n], "s_monto"), "frente_tamano": ic_razon(med, listas[n], "s_tamano_proveedor")}
        paso(f"{n}: {res['razones_2023'][n]}")
    # ¿las banderas agregan al monto? En hallazgos, solo entidades-año con alguno (esa verdad solo existe donde hubo control)
    b = {}
    fh = fit[fit.groupby("grupo")["y_rev"].transform("any")]
    mh = med[med.groupby("grupo")["y_rev"].transform("any")]
    bcols = ["logM", "f30", "f_cerca"]
    dm3 = fh[bcols] - fh.groupby("grupo")[bcols].transform("mean")
    sd3 = dm3.std().to_numpy()
    lr3 = LogisticRegression(C=1e6, max_iter=500, fit_intercept=False)
    lr3.fit(dm3.to_numpy() / sd3, fh["y_rev"].to_numpy(dtype=bool))
    w3 = lr3.coef_[0] / sd3
    d.loc[:, "s_monto_banderas"] = d[bcols].to_numpy() @ w3
    mh = d[(d["anio"] == MEDIDA)]
    mh = mh[mh.groupby("grupo")["y_rev"].transform("any")]
    fh = d[d["anio"].isin(AJUSTE)]
    fh = fh[fh.groupby("grupo")["y_rev"].transform("any")]
    b["pesos"] = {c: float(w) for c, w in zip(bcols, w3)}
    for nom, q in (("2023", mh), ("2020_2021", fh)):
        fila = {}
        for n, c in (("monto", "s_monto"), ("monto_banderas", "s_monto_banderas"), ("tamano", "s_tamano_proveedor")):
            L = top(q, c)
            fila[n] = {"aciertos": int(L["y_rev"].sum()), "mostrados": len(L)}
        b[nom] = fila
    b["ic_2023"] = ic_razon(mh, "s_monto_banderas", "s_monto", y="y_rev")
    res["banderas_en_hallazgos"] = b
    paso(f"banderas en hallazgos: {json.dumps(b, ensure_ascii=False)}")
    (SALIDA / "sonda_prompt_final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    paso("LISTA")


if __name__ == "__main__":
    main()
