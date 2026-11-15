"""Vistas y medidas por caso (HALLAZGOS 160, secciones 38 y 39): etiquetas desde la tabla de Eventos y las dos columnas fijas, por caso y por proveedor-año."""
from pathlib import Path

import numpy as np
import pandas as pd

from etiquetas import BASE, K, cierre

DIR = BASE.parent / "casos"
OUT = BASE.parent / "resultados"
SALIDA = OUT / "por_caso"
TIPOS = {"ruc_entidad": str, "ruc": str, "anio": int}
PRUEBA = (2020, 2021, 2023)


def eventos():
    return pd.read_parquet(DIR / "eventos.parquet")


def etiquetas_caso(E, casos, corte, ventana=2):
    """Etiquetas de cada caso a la fecha `corte`. La ventana solo toca a y, y_S, y_P, y_P_ent e y_S_ent; lo estricto usa todo lo conocido."""
    corte = pd.Timestamp(corte)
    i3 = pd.MultiIndex.from_arrays([casos["ruc_entidad"].astype(str), casos["ruc"].astype(str), casos["anio"].astype(int)])
    i2 = pd.MultiIndex.from_arrays([casos["ruc"].astype(str), casos["anio"].astype(int)])
    sabido = (E["fecha_conocido"] <= corte).to_numpy()
    if ventana is None:
        en_v = sabido
    else:
        en_v = sabido & (E["fecha_conocido"].to_numpy() <= cierre(E["anio_hechos"].to_numpy() + ventana))
    f = E["fuente"].to_numpy()

    def por3(m):
        e = E[m & E["ruc_entidad"].notna().to_numpy()]
        return i3.isin(pd.MultiIndex.from_arrays([e["ruc_entidad"].astype(str), e["ruc"].astype(str), e["anio_hechos"].astype(int)]))

    def por2(m):
        e = E[m]
        return i2.isin(pd.MultiIndex.from_arrays([e["ruc"].astype(str), e["anio_hechos"].astype(int)]))

    abc = np.isin(f, ["A", "B", "C"])
    es_s, es_p, es_c = f == "S", f == "P", f == "C"
    d = pd.DataFrame(index=casos.index)
    d["y_estricta"] = por3(abc & sabido)
    d["y_rev"] = por3(np.isin(f, ["A", "B"]) & sabido)
    d["y_imp"] = por3(es_c & sabido)
    d["y_imp_real"] = d["y_imp"].to_numpy() & por3(es_c & E["impedimento_real"].to_numpy())
    d["y_C_otra"] = d["y_imp"].to_numpy() & ~d["y_imp_real"].to_numpy()
    d["y_S"], d["y_P"] = por2(es_s & en_v), por2(es_p & en_v)
    d["y_S_ent"], d["y_P_ent"] = por3(es_s & en_v), por3(es_p & en_v)
    d["y"] = por3(abc & en_v) | d["y_S"].to_numpy() | d["y_P"].to_numpy()
    return d


def top(t, col, k=5, semilla=0):
    """Las k filas de mayor puntaje de cada grupo. t ya viene solo con los grupos que importan."""
    s = t[col].astype(float).to_numpy()
    s = np.where(np.isnan(s), np.nanmin(s) - 1, s)
    r = np.random.default_rng(semilla)
    q = t.assign(_s=s + 1e-9 * r.random(len(s)) * (np.abs(s) + 1)).sort_values(["grupo", "_s"], ascending=[True, False])
    return q[q.groupby("grupo").cumcount() < k]


def conteo(top_, y):
    rev, hit = top_.groupby(["ruc", "anio"]).ngroups, top_[top_[y]].groupby(["ruc", "anio"]).ngroups
    return {"revisiones": len(top_), "aciertos": int(top_[y].sum()), "precision_caso": float(top_[y].mean()), "prov_anio_revisados": rev, "prov_anio_acierto": hit, "precision_prov_anio": hit / max(rev, 1)}


def dos_columnas(d, y, metodos):
    """Por método, en total y por año. d: casos de los años a medir. Solo entidades-año con algún positivo (la regla de siempre)."""
    t = d[d.groupby("grupo")[y].transform("any")]
    out = {}
    for n, c in metodos.items():
        L = top(t, c)
        out[n] = {**conteo(L, y), "por_anio": {int(a): conteo(g, y) for a, g in L.groupby("anio")}}
    return out


def columnas_adicionales(d, y, metodos):
    """Sección 46, sin reemplazar nada: proveedores-año hallados sobre los positivos que existen en esas entidades-año, y revisiones por proveedor-año distinto."""
    t = d[d.groupby("grupo")[y].transform("any")]
    total = t[t[y]].groupby(["ruc", "anio"]).ngroups
    out = {}
    for n, c in metodos.items():
        L = top(t, c)
        hit, rev = L[L[y]].groupby(["ruc", "anio"]).ngroups, L.groupby(["ruc", "anio"]).ngroups
        out[n] = {"prov_anio_hallados": hit, "prov_anio_positivos": total, "recall_prov_anio": hit / max(total, 1), "repeticion": len(L) / max(rev, 1)}
    return out


def ic_prov_anio(d, y, m, v, n=2000, semilla=0):
    """Razón de precisiones por proveedor-año, m frente a v, con IC al 90 % en dos vías (pesos de Poisson de entidades y proveedores)."""
    t = d[d.groupby("grupo")[y].transform("any")]
    a, b = top(t, m), top(t, v)
    e, ne = pd.factorize(pd.concat([a["ruc_entidad"], b["ruc_entidad"]]))
    p, np_ = pd.factorize(pd.concat([a["ruc"], b["ruc"]]))
    py, npy = pd.factorize(pd.concat([a["ruc"] + "|" + a["anio"].astype(str), b["ruc"] + "|" + b["anio"].astype(str)]))
    corte = len(a)
    prov_de = np.zeros(len(npy), dtype=int)
    prov_de[py] = p
    hit = np.r_[a[y].to_numpy(), b[y].to_numpy()].astype(bool)
    rng = np.random.default_rng(semilla)

    def prec(lo, hi, we, wp):
        act = we[e[lo:hi]] > 0
        rev, ac = np.zeros(len(npy), bool), np.zeros(len(npy), bool)
        rev[py[lo:hi][act]] = True
        ac[py[lo:hi][act & hit[lo:hi]]] = True
        w = wp[prov_de]
        return (w * ac).sum() / max((w * rev).sum(), 1e-12)

    uno_e, uno_p = np.ones(len(ne), dtype=int), np.ones(len(np_), dtype=int)
    p0 = prec(corte, len(py), uno_e, uno_p)
    raz = prec(0, corte, uno_e, uno_p) / p0 if p0 > 0 else None
    if raz is None:
        return {"razon": None, "ic90": None}
    bs = []
    for _ in range(n):
        we, wp = rng.poisson(1.0, len(ne)), rng.poisson(1.0, len(np_))
        d0 = prec(corte, len(py), we, wp)
        bs.append(prec(0, corte, we, wp) / d0 if d0 > 0 else np.nan)
    return {"razon": float(raz), "ic90": [float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))]}
