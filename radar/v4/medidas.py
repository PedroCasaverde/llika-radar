"""Medidas de la línea base (HALLAZGOS 160, C12 a C14). Biblioteca sin efectos: no lee ni escribe tablas.

- Precisión en k por entidad-año con algún positivo, como razón de sumas, igual que evaluacion.por_grupo_k.
- IC al 90 % por remuestreo de pesos de Poisson: por entidad, por proveedor y en dos vías (entidad y proveedor). La etiqueta
  amplia es del proveedor y se repite en muchas entidades, así que sus razones se juzgan con el IC en dos vías (M1).
- Si el comparador no acierta nada, la razón es «no definida» y se reportan los aciertos de cada lado (M10).
- Horas por acierto con su nombre exacto: en las entidades-año con algún positivo y en todas (M5).
- Comparadores locales para la alerta: el mismo número de casos dentro de las mismas entidades-año (M9).
"""
import numpy as np
import pandas as pd
from scipy.stats import binom


def aciertos_top(s, grupo, k=5, desempates=10, semilla=0):
    """Probabilidad de cada fila de estar en el top k de su grupo, con desempates al azar (la misma regla de por_grupo_k)."""
    s = np.asarray(s, dtype=float)
    g = pd.factorize(np.asarray(grupo))[0]
    rng = np.random.default_rng(semilla)
    n = len(s)
    prob = np.zeros(n)
    for _ in range(desempates):
        ruido = s + 1e-9 * rng.random(n) * (np.abs(s) + 1)
        orden = np.lexsort((-ruido, g))
        gs = g[orden]
        ini = np.r_[True, gs[1:] != gs[:-1]]
        pos = np.arange(n) - np.repeat(np.flatnonzero(ini), np.diff(np.r_[np.flatnonzero(ini), n]))
        prob[orden[pos < k]] += 1
    return prob / desempates


def contribuciones(y, s, grupo, k=5, desempates=10):
    """Lo que aporta cada fila a la suma de P@k (aciertos / min(k, tamaño del grupo)), solo en grupos con algún positivo."""
    y = np.asarray(y, dtype=bool)
    g = pd.Series(np.asarray(grupo))
    tiene = pd.Series(y).groupby(g.to_numpy()).transform("any").to_numpy()
    tam = g.map(g.value_counts()).to_numpy()
    p = aciertos_top(s, grupo, k, desempates)
    hits = p * y * tiene
    return hits / np.minimum(k, tam), hits, tiene


def razon_ic(c1, c0, entidad, proveedor, n=2000, semilla=0):
    """Razón de sumas c1 / c0 con IC al 90 % por entidad, por proveedor y en dos vías (pesos de Poisson que se multiplican)."""
    c1, c0 = np.asarray(c1, float), np.asarray(c0, float)
    m = (c1 > 0) | (c0 > 0)
    c1, c0 = c1[m], c0[m]
    e, ne = pd.factorize(np.asarray(entidad)[m])
    p, np_ = pd.factorize(np.asarray(proveedor)[m])
    s1, s0 = float(c1.sum()), float(c0.sum())
    r = {"suma_metodo": s1, "suma_comparador": s0, "razon": s1 / s0 if s0 > 0 else None}
    if s0 <= 0:
        r.update(ic90_entidad=None, ic90_proveedor=None, ic90_dos_vias=None, nota="no definida: el comparador no acierta")
        return r
    rng = np.random.default_rng(semilla)
    be, bp, bd = np.empty(n), np.empty(n), np.empty(n)
    for b in range(n):
        we, wp = rng.poisson(1.0, len(ne))[e], rng.poisson(1.0, len(np_))[p]
        for arr, w in ((be, we), (bp, wp), (bd, we * wp)):
            d = (w * c0).sum()
            arr[b] = (w * c1).sum() / d if d > 0 else np.nan
    ic = lambda a: [float(np.nanpercentile(a, 5)), float(np.nanpercentile(a, 95))]  # noqa: E731
    r.update(ic90_entidad=ic(be), ic90_proveedor=ic(bp), ic90_dos_vias=ic(bd))
    return r


def medir(cortes, y, metodos, bases, k=5, desempates=10, n_boot=2000):
    """cortes: {año: DataFrame con grupo, ruc_entidad, ruc, y las columnas de metodos}. metodos: {nombre: columna}.
    Devuelve, por método: P@k, aciertos, horas en los dos sentidos y razón frente a cada base, conjunta y por año."""
    filas = {m: [] for m in metodos}
    info = []
    for a, te in cortes.items():
        yy = te[y].to_numpy(dtype=bool)
        g = te["grupo"].to_numpy()
        tam = pd.Series(g).value_counts()
        tiene = pd.Series(yy).groupby(g).any()
        info.append({"anio": a, "grupos_con_positivo": int(tiene.sum()), "grupos": int(len(tam)), "positivos": int(yy.sum()),
                     "revisiones_con_positivo": float(np.minimum(k, tam.reindex(tiene[tiene].index)).sum()),
                     "revisiones_todas": float(np.minimum(k, tam).sum())})
        for m, col in metodos.items():
            c, h, _ = contribuciones(yy, te[col].to_numpy(dtype=float), g, k, desempates)
            filas[m].append(pd.DataFrame({"anio": a, "c": c, "h": h, "e": te["ruc_entidad"].to_numpy(), "p": te["ruc"].to_numpy()}))
    filas = {m: pd.concat(v, ignore_index=True) for m, v in filas.items()}
    ngr = sum(i["grupos_con_positivo"] for i in info)
    r1, r2 = sum(i["revisiones_con_positivo"] for i in info), sum(i["revisiones_todas"] for i in info)
    out = {"por_anio": info, "metodos": {}}
    for m, d in filas.items():
        h = float(d["h"].sum())
        out["metodos"][m] = {"precision": float(d["c"].sum() / max(ngr, 1)), "aciertos": h,
                             "aciertos_por_anio": {int(a): float(x) for a, x in d.groupby("anio")["h"].sum().items()},
                             "proveedores_con_acierto": int(d.loc[d["h"] > 0, "p"].nunique()),
                             "revisiones_por_acierto_con_positivo": r1 / h if h > 0 else None,
                             "revisiones_por_acierto_en_todas": r2 / h if h > 0 else None, "frente_a": {}}
    for b in bases:
        for m, d in filas.items():
            if m == b:
                continue
            d0 = filas[b]
            conj = razon_ic(d["c"], d0["c"], d["e"], d["p"], n=n_boot)
            conj["por_anio"] = {}
            for a in cortes:
                i = (d["anio"] == a).to_numpy()
                conj["por_anio"][int(a)] = razon_ic(d["c"].to_numpy()[i], d0["c"].to_numpy()[i], d["e"].to_numpy()[i], d["p"].to_numpy()[i], n=n_boot)
            out["metodos"][m]["frente_a"][b] = conj
    return out


def alerta_local(te, alerta, y, s_monto, s_otro=None, desempates=10, semilla=0):
    """La alerta frente al mismo número de casos DENTRO de cada entidad-año con alerta (M9): por monto, por otro puntaje y por
    azar (valor esperado). Prueba binomial de una cola sobre los aciertos de un solo método frente al monto local."""
    alerta, y = np.asarray(alerta, bool), np.asarray(y, bool)
    g = te["grupo"].to_numpy()
    n_al = pd.Series(alerta).groupby(g).transform("sum").to_numpy()
    tam = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    pos = pd.Series(y).groupby(g).transform("sum").to_numpy()
    rng = np.random.default_rng(semilla)

    def top(s):
        s = np.asarray(s, float)
        p = np.zeros(len(s))
        for _ in range(desempates):
            d = pd.DataFrame({"g": g, "s": s + 1e-9 * rng.random(len(s)) * (np.abs(s) + 1)}).sort_values(["g", "s"], ascending=[True, False])
            d["r"] = d.groupby("g").cumcount() + 1
            sel = d.index[d["r"].to_numpy() <= n_al[d.index]]
            p[sel] += 1
        return p / desempates
    t_m = top(s_monto)
    r = {"alertas": int(alerta.sum()), "entidades_anio_con_alerta": int(pd.Series(g[alerta]).nunique()), "aciertos_alerta": int(y[alerta].sum()),
         "aciertos_monto_local": float((t_m * y).sum())}
    if s_otro is not None:
        r["aciertos_otro_local"] = float((top(s_otro) * y).sum())
    primera = ~pd.Series(g).duplicated().to_numpy()
    r["aciertos_azar_local"] = float((n_al[primera] * pos[primera] / tam[primera]).sum())
    b = int((y & alerta & (t_m < 0.5)).sum())
    c = int((y & ~alerta & (t_m >= 0.5)).sum())
    r["solo_alerta"], r["solo_monto"] = b, c
    r["p_binomial_una_cola"] = float(binom.sf(b - 1, b + c, 0.5)) if b + c else None
    return r


def fmt(r):
    """Razón con su IC en dos vías, o «no definida»."""
    if r["razon"] is None:
        return f"no definida ({r['suma_metodo']:.2f} frente a 0)"
    i = r["ic90_dos_vias"]
    return f"{r['razon']:.2f} [{i[0]:.2f} a {i[1]:.2f}]"
