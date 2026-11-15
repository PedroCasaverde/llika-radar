"""Data v4, capa 4: armado de cortes y métricas (HALLAZGOS 116 a 118). Los conjuntos de evaluación se arman al vuelo y
nunca se guardan con etiquetas.

Regla de evaluación vigente (HALLAZGOS 129, 133 y 160; `medidas.py`):
- principal: razón de la precisión en 5 por entidad-año con algún positivo frente al comparador de cada vista; la 116
  usaba la diferencia de MAP frente al monto, que queda como sensibilidad;
- se reporta por año y por tipo de positivo, con la precisión en 1, 10 y 20 y el MAP como sensibilidad;
- empates: promedio de varios desempates al azar;
- intervalos: por entidad en las etiquetas de la relación; en dos vías (entidad y proveedor) en la etiqueta amplia.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from etiquetas import HOY_EXPLORACION, K, cierre  # noqa: E402

X_PATH = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "x_caso_anio.parquet"
DICC = X_PATH.parent / "diccionario_x.csv"
CATEGORICAS = ["e_tipo", "e_depto", "rubro_dom"]


def cargar_x(anios):
    x = pd.read_parquet(X_PATH, filters=[("anio", "in", list(anios))])
    for nombre, llave in (("x_territorial", ["ruc_entidad", "ruc", "anio"]), ("x_territorial_proveedor", ["ruc", "anio"]),
                          ("x_familia", ["ruc_entidad", "ruc", "anio"]), ("x_padre_hijo", ["ruc_entidad", "ruc", "anio"]),
                          ("x_padre_hijo_jne", ["ruc_entidad", "ruc", "anio"]), ("x_impedidos", ["ruc_entidad", "ruc", "anio"])):
        f = X_PATH.parent / f"{nombre}.parquet"           # tarjeta territorial (HALLAZGOS 128) y familia (132), a la fecha
        if f.exists():
            t = pd.read_parquet(f, filters=[("anio", "in", list(anios))])
            x = x.merge(t.astype({c: x[c].dtype for c in llave}), on=llave, how="left")
            tc = [c for c in t.columns if c.startswith(("t_", "tp_", "f_fam_", "f_ph_", "f_empfunc_", "f_funcprop_"))]
            x[tc] = x[tc].fillna(0).astype("float32")
    return x


def familia_a_la_decision(x, anio_decision):
    """HALLAZGOS 132: el posible familiar de las declaraciones cuenta si su declaración se presentó hasta el 31/12 del año
    de decisión, igual para entrenamiento y prueba. Antes de 2022 no había declaraciones: queda en cero."""
    y = min(int(anio_decision), 2025)
    for pref, nombre in (("f_fam", "f_familiar"), ("f_ph", "f_padre_hijo")):      # hermanos (132) y padre o madre e hijo (136)
        for v in ("prob", "ordenes"):
            c = f"{pref}_{v}_hasta_{y}"
            x[f"{nombre}_{v}"] = x[c] if c in x.columns else np.float32(0)
    for pref, nombre in (("f_empfunc", "f_empresa_funcionario"), ("f_funcprop", "f_funcionario_propio")):   # 148
        c = f"{pref}_ordenes_hasta_{y}"
        x[f"{nombre}_ordenes"] = x[c] if c in x.columns else np.float32(0)
    return x


def columnas(clases=("contemporánea", "estática fechable (foto de 2026)"), quitar=()):
    d = pd.read_csv(DICC)
    d = d[d["clase_temporal"].isin(clases) & (d["grupo"] != "universo") & ~d["variable"].isin(quitar)]
    return d["variable"].tolist()


def universo(x, hist=None, excluir_basicos=True, excluir_publicos=True):
    """Banderas de la capa 2 → máscara del universo. Los básicos y públicos se quedan si el proveedor tenía historia
    conocida (hechos o inhabilitaciones) al cierre del año."""
    fuera = pd.Series(False, index=x.index)
    if excluir_basicos:
        fuera |= x["u_servicio_basico"].astype(bool)
    if excluir_publicos:
        fuera |= x["u_proveedor_publico"].astype(bool)
    if hist is not None:
        con_historia = (hist.get("h_prov_hechos", 0) > 0) | (hist.get("h_prov_inhabilitaciones", 0) > 0)
        fuera &= ~con_historia
    return ~fuera


def armar_corte(H, anios_train, anio_test, horizonte=0, deteccion=True, fecha_prueba=HOY_EXPLORACION):
    """Entrenamiento con etiquetas conocidas al cierre del año de prueba; prueba con lo conocido a `fecha_prueba`.
    En detección (horizonte 0) se excluyen, en ambos, los casos cuyo hecho ya se conocía al cierre de su propio año."""
    corte = pd.Timestamp(f"{anio_test}-12-31")
    x = familia_a_la_decision(cargar_x(list(anios_train) + [anio_test]), anio_test)
    hist = H.historia(x)
    x = pd.concat([x, hist], axis=1)
    x = x[universo(x, hist)].copy()
    tr, te = x[x["anio"].isin(anios_train)].copy(), x[x["anio"] == anio_test].copy()
    et_tr = H.etiqueta(tr, fecha_corte=corte, horizonte=horizonte)
    H.comprobar_sin_futuro(et_tr, corte)
    if horizonte > 0:                       # la prevención solo puede usar hechos de años anteriores al de prueba
        et_tr.loc[tr["anio"] + horizonte > anio_test, ["y", "peso"]] = [False, 0.0]
    et_te = H.etiqueta(te, fecha_corte=fecha_prueba, horizonte=horizonte)
    tr, te = tr.join(et_tr), te.join(et_te)
    if deteccion and horizonte == 0:          # lo ya conocido al cierre de su año no se aprende ni se evalúa
        tr = tr[~tr["conocido_al_cierre"]]
        te = te[~te["conocido_al_cierre"]]
    tr["grupo"] = tr["ruc_entidad"] + "|" + tr["anio"].astype(str)
    te["grupo"] = te["ruc_entidad"] + "|" + te["anio"].astype(str)
    return tr, te


class Transformador:
    """Decide con el entrenamiento qué columnas van en logaritmo y los códigos de las categóricas; aplica lo mismo a la
    prueba (corrige el defecto 6 de HALLAZGOS 113)."""

    def __init__(self, tr, cols):
        self.cols = [c for c in cols if c in tr.columns]
        num = [c for c in self.cols if c not in CATEGORICAS]
        z = tr[num].astype(float)
        self.log = [c for c in num if z[c].max() > 1 and z[c].min() >= 0 and not c.startswith(("c_rub_", "c_kw", "c_cv"))
                    and "parte" not in c and "pct" not in c and "crec" not in c]
        self.codigos = {c: {v: i for i, v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CATEGORICAS if c in self.cols}

    def __call__(self, d):
        z = d[[c for c in self.cols if c not in CATEGORICAS]].astype("float32").copy()
        z[self.log] = np.log1p(z[self.log].clip(lower=0))
        for c, m in self.codigos.items():
            z[c] = d[c].astype(str).map(m).fillna(-1).astype("float32")
        return z


# ------------------------------------------------------------------------------------------------ métricas
def _ap_por_grupo(y, s, g, rng):
    d = pd.DataFrame({"g": g, "y": y, "s": s + 1e-9 * rng.random(len(s)) * (np.abs(s) + 1)})
    d = d.sort_values(["g", "s"], ascending=[True, False])
    d["r"] = d.groupby("g").cumcount() + 1
    d["acum"] = d.groupby("g")["y"].cumsum()
    ap = d[d["y"]].assign(p=lambda x: x["acum"] / x["r"]).groupby("g")["p"].mean()
    top5 = d[d["r"] <= 5].groupby("g")["y"].sum()
    return ap, top5


def por_entidad(y, s, grupo, desempates=10, semilla=0):
    """Métricas por entidad-año con algún positivo; AP y P@5 promediados sobre varios desempates al azar."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    g = pd.Series(np.asarray(grupo))
    con = g.map(pd.Series(y).groupby(g.to_numpy()).any())
    m = con.to_numpy()
    y, s, g = y[m], s[m], g[m].to_numpy()
    rng = np.random.default_rng(semilla)
    aps, tops = [], []
    for _ in range(desempates):
        ap, top5 = _ap_por_grupo(y, s, g, rng)
        aps.append(ap)
        tops.append(top5)
    ap = pd.concat(aps, axis=1).mean(axis=1)
    top5 = pd.concat(tops, axis=1).fillna(0).mean(axis=1)
    n = pd.Series(g).value_counts()
    p5 = top5 / np.minimum(5, n.reindex(top5.index))
    return pd.DataFrame({"AP": ap, "P@5": p5, "acierto@5": (top5 > 0).astype(float)})


def metricas(te, s, y="y", desempates=10):
    yy = te[y].to_numpy(dtype=bool)
    pe = por_entidad(yy, s, te["grupo"], desempates)
    q = te.groupby("grupo")["c_monto"].rank(pct=True).mul(5).clip(upper=4.999).astype(int).astype(str)
    pq_ = por_entidad(yy, s, (te["grupo"] + "|" + q).to_numpy(), desempates)
    orden = np.argsort(-np.asarray(s))
    todas = te.assign(_s=s).sort_values(["grupo", "_s"], ascending=[True, False])
    todas["r"] = todas.groupby("grupo").cumcount() + 1
    aciertos_revision = float(todas[todas["r"] <= 5].groupby("grupo")[y].mean().mean())
    return {"MAP_ent": float(pe["AP"].mean()), "P@5_ent": float(pe["P@5"].mean()), "acierto@5_ent": float(pe["acierto@5"].mean()),
            "MAP_quintil_monto": float(pq_["AP"].mean()), "aciertos_por_revision_todas": aciertos_revision,
            "P@1000_nacional": float(yy[orden[:1000]].mean()), "entidades_con_positivo": int(len(pe)), "positivos": int(yy.sum())}


def por_grupo_k(y, s, grupo, ks=(1, 3, 5, 10), desempates=10, semilla=0):
    """AP y precisión en k por grupo con algún positivo, promediados sobre desempates al azar (Pedro, 2/11: precisión en k
    también dentro de los rangos de monto)."""
    y, s, g = np.asarray(y, dtype=bool), np.asarray(s, dtype=float), np.asarray(grupo)
    con = pd.Series(y).groupby(g).transform("any").to_numpy()
    y, s, g = y[con], s[con], g[con]
    n = pd.Series(g).value_counts()
    rng = np.random.default_rng(semilla)
    aps, tops = [], {k: [] for k in ks}
    for _ in range(desempates):
        d = pd.DataFrame({"g": g, "y": y, "s": s + 1e-9 * rng.random(len(s)) * (np.abs(s) + 1)})
        d = d.sort_values(["g", "s"], ascending=[True, False])
        d["r"] = d.groupby("g").cumcount() + 1
        d["acum"] = d.groupby("g")["y"].cumsum()
        aps.append(d[d["y"]].assign(p=lambda x: x["acum"] / x["r"]).groupby("g")["p"].mean())
        for k in ks:
            tops[k].append(d[d["r"] <= k].groupby("g")["y"].sum())
    out = pd.DataFrame({"AP": pd.concat(aps, axis=1).mean(axis=1)})
    for k in ks:
        t = pd.concat(tops[k], axis=1).fillna(0).mean(axis=1)
        out[f"P@{k}"] = t / np.minimum(k, n.reindex(t.index))
    return out


def quintil_monto(te):
    q = te.groupby("grupo")["c_monto"].rank(pct=True).mul(5).clip(upper=4.999).astype(int).astype(str)
    return (te["grupo"] + "|" + q).to_numpy()


def metricas_k(te, s, y="y", ks=(1, 3, 5, 10)):
    """Precisión en k por entidad y dentro de los quintiles de monto de cada entidad, más el AP de cada caso."""
    a = por_grupo_k(te[y], s, te["grupo"].to_numpy(), ks).mean()
    b = por_grupo_k(te[y], s, quintil_monto(te), ks).mean()
    return {**{f"{m}_ent": float(v) for m, v in a.items()}, **{f"{m}_quintil": float(v) for m, v in b.items()}}


_CACHE_REFERENCIA = {}


def razon_conjunta(cortes, medida="AP", quintil=False, y="y", n=1000, semilla=0):
    """cortes: lista de (te, s_modelo, s_monto). Razón modelo / monto de una medida por grupo (AP o P@k), en todos los
    cortes juntos, con IC al 90 % remuestreando entidades. Para la vara de Pedro: el doble del monto."""
    k = int(medida.split("@")[1]) if "@" in medida else 5
    filas = []
    for te, s1, s0 in cortes:
        g = quintil_monto(te) if quintil else te["grupo"].to_numpy()
        a1 = por_grupo_k(te[y], s1, g, (k,), desempates=3)[medida]     # 3 desempates: la razón se calcula decenas de veces
        clave = (id(te), id(s0), medida, quintil, y)     # el monto se repite para cada candidata: se calcula una vez
        if clave not in _CACHE_REFERENCIA:              # (te y s0 viven en el diccionario de puntajes durante la corrida)
            _CACHE_REFERENCIA[clave] = por_grupo_k(te[y], s0, g, (k,), desempates=3)[medida]
        a0 = _CACHE_REFERENCIA[clave].reindex(a1.index)
        filas.append(pd.DataFrame({"m1": a1.to_numpy(), "m0": a0.to_numpy(), "entidad": pd.Series(a1.index).str.split("|").str[0].to_numpy()}))
    d = pd.concat(filas)
    pe = d.groupby("entidad")[["m1", "m0"]].sum()
    rng = np.random.default_rng(semilla)
    boot = []
    for _ in range(n):
        m = pe.iloc[rng.integers(0, len(pe), len(pe))]
        boot.append(m["m1"].sum() / max(m["m0"].sum(), 1e-12))
    return {"modelo": float(d["m1"].mean()), "monto": float(d["m0"].mean()), "razon": float(d["m1"].sum() / max(d["m0"].sum(), 1e-12)),
            "ic90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))]}


def diferencia_conjunta(cortes, desempates=10, n=1000, semilla=0):
    """cortes: lista de (te, s_modelo, s_monto). Diferencia de MAP por entidad (modelo − monto) en todos los cortes juntos,
    con IC al 90 % remuestreando ENTIDADES (cada entidad entra con todas sus entidades-año de todos los cortes)."""
    filas = []
    for i, (te, s1, s0) in enumerate(cortes):
        yy = te["y"].to_numpy(dtype=bool)
        a1 = por_entidad(yy, s1, te["grupo"], desempates)["AP"]
        a0 = por_entidad(yy, s0, te["grupo"], desempates)["AP"]
        d = (a1 - a0.reindex(a1.index)).rename("dif").reset_index().rename(columns={"index": "grupo", "g": "grupo"})
        d["entidad"] = d["grupo"].str.split("|").str[0]
        d["corte"] = i
        filas.append(d)
    d = pd.concat(filas)
    por_ent = d.groupby("entidad")["dif"].agg(["sum", "size"])
    rng = np.random.default_rng(semilla)
    ents = por_ent.index.to_numpy()
    boot = []
    for _ in range(n):
        m = por_ent.loc[rng.choice(ents, len(ents))]
        boot.append(m["sum"].sum() / m["size"].sum())
    por_corte = d.groupby("corte")["dif"].mean().round(4).to_dict()
    return {"dif_MAP": float(d["dif"].mean()), "ic90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
            "por_corte": por_corte, "entidades": int(len(ents))}
