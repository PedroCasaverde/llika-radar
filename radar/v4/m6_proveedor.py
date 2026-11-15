"""Prueba de modelos 6 (HALLAZGOS 160, sección 26): modelo del proveedor y la data que faltaba estructurar. Fijado antes de correr.

    python -I -X utf8 m6_proveedor.py elegir
    python -I -X utf8 m6_proveedor.py confirmar
"""
import glob
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lector_penalidades as LP  # noqa: E402
from comun import CRUDOS, PROC  # noqa: E402
from d1_linea_base import exposicion, monotonia  # noqa: E402
from d15_variables import NUEVAS, lista_121  # noqa: E402
from etiquetas import BASE, K, SELLADO_ANIO, SELLADO_DESDE, Hechos  # noqa: E402
from evaluacion import CATEGORICAS, Transformador, armar_corte, cargar_x, familia_a_la_decision, universo  # noqa: E402
from m1_d1_configuracion import ajustar, preparar_te  # noqa: E402
from m2_objetivo import etiqueta_ajuste, mitad  # noqa: E402
from m4_simples import rrf2  # noqa: E402
from medidas import fmt, medir  # noqa: E402

OUT = PROC / "v4" / "resultados"
ARBOLES = (50, 100, 200)
CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
K_CRED = 100.0
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


class Fuentes:
    """Las tablas crudas por proveedor, una sola vez. Nada de 2022 ni lo conocido desde 2026 salvo con abrir."""

    def __init__(self, abrir=False):
        p, _ = LP.fechar(LP.penalidades(CRUDOS / "oece_penalidades" / "_historia"))
        p = p[~p["es_consorcio"]].copy()
        p["mora"] = p["tipo_penalidad"].astype(str).str.upper().str.contains("MORA")
        p["fecha_conocido"] = pd.to_datetime(p["fecha_conocido"])
        s = pd.read_parquet(BASE / "sanciones.parquet").dropna(subset=["inicio"])
        mot = s["motivo"].astype(str).str.upper()
        s["falsa"], s["resolver"], s["impedido"] = mot.str.contains("FALS|INEXACT|ADULTER"), mot.str.contains("RESUELVA|RESOLUCI"), mot.str.contains("IMPEDIDO|SUPUESTOS")
        j = self.judiciales()
        if not abrir:
            p = p[(p["fecha"].dt.year != SELLADO_ANIO) & (p["fecha_conocido"] < SELLADO_DESDE)]
            s = s[s["inicio"] < SELLADO_DESDE]
            j = j[j["conocido"] < SELLADO_DESDE]
        self.p, self.s, self.j = p, s, j
        ad = pd.read_parquet(PROC / "ocds" / "adjudicaciones.parquet", columns=["ocid", "adjudicacion_id", "ruc"]).dropna(subset=["ruc"]).drop_duplicates()
        co = pd.read_parquet(PROC / "ocds" / "contratos.parquet", columns=["ocid", "adjudicacion_id", "fecha_firma", "fin", "monto", "moneda"])
        co = co.merge(ad, on=["ocid", "adjudicacion_id"])
        co["firma"] = pd.to_datetime(co["fecha_firma"].str[:10], errors="coerce")
        co["vence"] = pd.to_datetime(co["fin"].str[:10], errors="coerce")
        co["monto"] = pd.to_numeric(co["monto"], errors="coerce").where(co["moneda"].fillna("PEN").eq("PEN"))
        self.c = co.dropna(subset=["firma"]).astype({"ruc": str})
        o = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc", "anio"]).astype({"ruc": str, "anio": int})
        self.o = o.groupby(["ruc", "anio"]).size().rename("n").reset_index()
        paso(f"fuentes: penalidades {len(p):,}; sanciones {len(s):,}; judiciales {len(j):,}; contratos {len(self.c):,}")

    @staticmethod
    def judiciales():
        filas = []
        for f in sorted(glob.glob(str(CRUDOS / "oece_inhabilitaciones_judiciales" / "_historia" / "*.csv"))):
            ver = pd.Timestamp(Path(f).stem.split("_")[-1])
            d = pd.read_csv(f, sep="|", dtype=str, encoding="latin1", on_bad_lines="skip")
            d.columns = [c.upper() for c in d.columns]
            cr = next((c for c in d.columns if c.startswith("RUC")), None)
            ci = next((c for c in d.columns if "INICIO" in c), None)
            cf = next((c for c in d.columns if "FIN" in c and "CORTE" not in c), None)
            if cr is None or ci is None:
                continue
            filas.append(pd.DataFrame({"doc": d[cr].astype(str).str.strip(), "inicio": pd.to_datetime(d[ci].str[:10], errors="coerce", dayfirst=True),
                                       "fin": pd.to_datetime(d[cf].str[:10], errors="coerce", dayfirst=True) if cf else pd.NaT, "version": ver}))
        j = pd.concat(filas, ignore_index=True).dropna(subset=["inicio"])
        g = j.sort_values("version").groupby(["doc", "inicio"], sort=False)
        j = g.last().reset_index().assign(primera=g["version"].min().to_numpy())
        primeras = j["primera"].min()
        j["conocido"] = j["primera"].where(j["primera"] > primeras, j["inicio"])       # antes de la primera versión, su propia fecha
        j["dni"] = j["doc"].where(j["doc"].str.len() == 8, j["doc"].str[2:10].where(j["doc"].str.startswith("10")))
        return j

    def al_cierre(self, a, rucs):
        """Variables por proveedor conocidas al 31/12 del año a."""
        fin = pd.Timestamp(f"{a}-12-31")
        idx = pd.Index(pd.unique(rucs), name="ruc")
        d = pd.DataFrame(index=idx)
        p = self.p[self.p["fecha_conocido"] <= fin]
        g = p.groupby("ruc")
        d["h_prov_pen_mora"], d["h_prov_pen_otra"] = g["mora"].sum(), g["mora"].size() - g["mora"].sum()
        d["h_prov_pen_monto"], d["h_prov_pen_entidades"] = g["monto"].sum(), g["entidad"].nunique()
        d["h_prov_pen_reciente"] = p[p["fecha_conocido"] > fin - pd.DateOffset(years=2)].groupby("ruc").size()
        s = self.s[self.s["inicio"] <= fin]
        gs = s.groupby("ruc")
        d["h_prov_san_falsa"], d["h_prov_san_resolver"], d["h_prov_san_impedido"] = gs["falsa"].sum(), gs["resolver"].sum(), gs["impedido"].sum()
        d["h_prov_san_definitiva"] = gs["definitiva"].sum()
        vence = s["fin"].fillna(fin).clip(upper=fin)
        d["h_prov_san_meses"] = ((vence - s["inicio"]).dt.days.clip(lower=0) / 30.4).groupby(s["ruc"]).sum()
        j = self.j[self.j["conocido"] <= fin]
        dni = pd.Series(idx.str[2:10].where(idx.str.startswith("10")), index=idx)
        por_ruc = j.groupby("doc").size()
        por_dni = j.dropna(subset=["dni"]).groupby("dni").size()
        d["h_prov_jud"] = pd.Series(idx, index=idx).map(por_ruc).fillna(0) + dni.map(por_dni).fillna(0)
        act = j[j["fin"].isna() | (j["fin"] >= fin)]
        d["h_prov_jud_activa"] = (pd.Series(idx, index=idx).map(act.groupby("doc").size()).fillna(0) + dni.map(act.dropna(subset=["dni"]).groupby("dni").size()).fillna(0)).gt(0).astype(float)
        c = self.c[self.c["firma"] <= fin]
        ej = c[c["vence"] >= fin]
        d["p_ejec_contratos"], d["p_ejec_monto"] = ej.groupby("ruc").size(), ej.groupby("ruc")["monto"].sum()
        d["p_contratos_hasta_t"] = c.groupby("ruc").size()
        d["p_ordenes_hasta_t"] = self.o[self.o["anio"] <= a].groupby("ruc")["n"].sum()
        d = d.fillna(0.0).astype(float)
        expo = d["p_contratos_hasta_t"] + d["p_ordenes_hasta_t"]
        n_pen, n_san = d["h_prov_pen_mora"] + d["h_prov_pen_otra"], d[["h_prov_san_falsa", "h_prov_san_resolver", "h_prov_san_impedido"]].sum(axis=1)
        t_pen, t_san = n_pen.sum() / max(expo.sum(), 1), n_san.sum() / max(expo.sum(), 1)
        d["h_prov_cred_pen"] = (n_pen + K_CRED * t_pen) / (expo + K_CRED)
        d["h_prov_cred_san"] = (n_san + K_CRED * t_san) / (expo + K_CRED)
        return d


NUEVAS_P = ["h_prov_pen_mora", "h_prov_pen_otra", "h_prov_pen_monto", "h_prov_pen_entidades", "h_prov_pen_reciente", "h_prov_san_falsa", "h_prov_san_resolver", "h_prov_san_impedido",
            "h_prov_san_definitiva", "h_prov_san_meses", "h_prov_jud", "h_prov_jud_activa", "p_ejec_contratos", "p_ejec_monto", "p_contratos_hasta_t", "p_ordenes_hasta_t", "h_prov_cred_pen", "h_prov_cred_san"]


def con_nuevas(F, x):
    x = x.assign(ruc=x["ruc"].astype(str))
    partes = []
    for a, g in x.groupby("anio"):
        partes.append(g[["ruc"]].join(F.al_cierre(int(a), g["ruc"]), on="ruc").drop(columns="ruc"))
    return pd.concat([x, pd.concat(partes).reindex(x.index)], axis=1)


def por_proveedor(d, cols):
    agg = {c: "first" for c in cols if c not in ("c_monto", "c_ordenes")}
    r = d.groupby(["ruc", "anio"], sort=False).agg({**agg, "c_monto": "sum", "c_ordenes": "sum"}).reset_index()
    return r


def modelo_proveedor(tr, y, te, cols, arboles, semillas=range(10)):
    """Por proveedor-año: margen de exposición y exceso de riesgo. Devuelve {árboles: (puntaje del proveedor con margen, exceso)} por fila de te."""
    t = tr.assign(_y=y)
    ptr = por_proveedor(t, cols).merge(t.groupby(["ruc", "anio"], sort=False)["_y"].max().reset_index(), on=["ruc", "anio"])
    pte = por_proveedor(te, cols)
    yy = ptr["_y"].to_numpy(dtype=bool)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(exposicion(ptr), yy)
    m_tr, m_te = lr.decision_function(exposicion(ptr)), lr.decision_function(exposicion(pte))
    T = Transformador(ptr, cols)
    Xtr, Xte = T(ptr), T(pte)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                  monotone_constraints=monotonia(Xtr.columns), monotone_constraints_method="advanced", verbose=-1, num_threads=12)
    ds = lgb.Dataset(Xtr, yy.astype(int), init_score=m_tr, free_raw_data=False)
    exc = {a: [] for a in arboles}
    for s in semillas:
        b = lgb.train({**params, "seed": s}, ds, num_boost_round=max(arboles))
        for a in arboles:
            exc[a].append(b.predict(Xte, raw_score=True, num_iteration=a))
    llave = pd.MultiIndex.from_frame(pte[["ruc", "anio"]])
    pos = llave.get_indexer(pd.MultiIndex.from_frame(te[["ruc", "anio"]]))
    return {a: ((m_te + np.mean(v, axis=0))[pos], np.mean(v, axis=0)[pos]) for a, v in exc.items()}


def candidatas(tr, y, te, cprov, m_case, s_d1, grupo, arboles=ARBOLES):
    r = {}
    M = modelo_proveedor(tr, y, te, cprov, arboles)
    for a, (p, e) in M.items():
        r[f"P1_{a}"], r[f"P2_{a}"], r[f"P4_{a}"] = p, m_case + e, rrf2(p, s_d1, grupo)
    return r


def margen_caso(tr, y, te):
    w = np.where(y, tr["peso"].clip(lower=0.25), 1.0)
    return LogisticRegression(C=1.0, max_iter=500).fit(exposicion(tr), y, sample_weight=w).decision_function(exposicion(te))


def elegir():
    H = Hechos(conocimiento="publicacion")
    F = Fuentes()
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    x = familia_a_la_decision(cargar_x([2018, 2019]), 2019)
    hist = H.historia(x)
    x = pd.concat([x, hist], axis=1)
    x = con_nuevas(F, x[universo(x, hist)].copy())
    tr, te = x[x["anio"] == 2018].copy(), x[x["anio"] == 2019].copy()
    tr = tr[~H.etiqueta_amplia(tr, "2020-12-31")["conocido_al_cierre"].to_numpy()].reset_index(drop=True)
    te = te.join(H.etiqueta_amplia(te, "2021-12-31"))
    te = preparar_te(te[~te["conocido_al_cierre"]].reset_index(drop=True))
    te["anio"] = te["anio"].astype(int)
    tr["anio"] = tr["anio"].astype(int)
    y, peso = etiqueta_ajuste(H, tr, "T2", "2021-12-31")
    tr["y"], tr["peso"] = y, peso
    m_tr, m_te = mitad(tr["ruc"]), mitad(te["ruc"])
    partes = {}
    for f in (0, 1):
        t, s = tr[m_tr == f], te[m_te != f].copy()
        paso(f"mitad {f}: ajuste {len(t):,}, {int(t['y'].sum()):,} positivos; prueba {len(s):,}")
        s["s_D1"] = ajustar(t, s, c116, (50,), True)[50].mean(axis=0)
        s["s_P3"] = ajustar(t, s, c116 + NUEVAS_P, (50,), True)[50].mean(axis=0)
        for n, v in candidatas(t, t["y"].to_numpy(), s, cprov, margen_caso(t, t["y"].to_numpy(), s), s["s_D1"].to_numpy(), s["grupo"].to_numpy()).items():
            s[f"s_{n}"] = v
        partes[f + 1] = s
    nombres = [c[2:] for c in partes[1].columns if c.startswith("s_") and c[2:] in {"D1", "P3"} | {f"{p}_{a}" for p in ("P1", "P2", "P4") for a in ARBOLES}]
    filas = {}
    r = {yy: medir(partes, yy, {n: f"s_{n}" for n in nombres} | {"tamano": "s_tamano"}, ["tamano"], n_boot=100)["metodos"] for yy in ("y", "y_S")}
    for n in nombres:
        py, ps = r["y"][n]["precision"], r["y_S"][n]["precision"]
        filas[n] = {"precision_y": py, "precision_S": ps, "media_geometrica": float(np.sqrt(py * ps))}
        paso(f"   {n}: {filas[n]}")
    mejor = max((n for n in nombres if n != "D1"), key=lambda n: filas[n]["media_geometrica"])
    pasa = filas[mejor]["media_geometrica"] >= 1.01 * filas["D1"]["media_geometrica"]
    (OUT / "m6_elegir.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "m6_elegido.json").write_text(json.dumps({"candidata": mejor if pasa else None, "mejor": mejor}), encoding="utf-8")
    paso(f"ELEGIDO: {mejor if pasa else 'ninguno'} (mejor {mejor})")


def regla(res_y, res_s):
    gy, gs = res_y, res_s
    sube = lambda g, u: sum(1 for x in g["por_anio"].values() if x["razon"] and x["razon"] >= u)  # noqa: E731
    via_s = gs["razon"] >= 1.05 and sube(gs, 1.05) >= 2 and gy["razon"] >= 0.95
    via_y = gy["razon"] >= 1.01 and sube(gy, 1.01) >= 2 and gs["razon"] >= 0.95
    return {"y": gy["razon"], "S": gs["razon"], "ic_y": gy["ic90_dos_vias"], "ic_S": gs["ic90_dos_vias"], "cortes_y": sube(gy, 1.01), "cortes_S": sube(gs, 1.05), "via_S": bool(via_s), "via_y": bool(via_y), "se_adopta": bool(via_s or via_y)}


def confirmar():
    el = json.loads((OUT / "m6_elegido.json").read_text(encoding="utf-8"))["candidata"]
    if not el:
        paso("no pasa a confirmar")
        return
    assert not (OUT / "m6_confirmar.json").exists(), "la confirmación se corre una sola vez"
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    F = Fuentes()
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    cortes = {}
    for a, anios in CORTES.items():
        tr, te = armar_corte(H, anios, a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        tr, te = con_nuevas(F, tr), con_nuevas(F, te)
        tr["anio"], te["anio"] = tr["anio"].astype(int), te["anio"].astype(int)
        y, peso = etiqueta_ajuste(H, tr, "T2", f"{a}-12-31")
        tr["y"], tr["peso"] = y, peso
        d = te[K].astype({"ruc_entidad": str, "ruc": str, "anio": int}).copy()
        m2 = pd.read_parquet(OUT / f"puntajes_m2_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        d = d.merge(m2, on=K, how="left")
        if el == "P3":
            d["s_m6"] = ajustar(tr, te, c116 + NUEVAS_P, (50,), True)[50].mean(axis=0)
        else:
            cl, ar = el.split("_")
            v = candidatas(tr, y, te, cprov, margen_caso(tr, y, te), d["s_m2"].to_numpy(), te["ruc_entidad"].astype(str).to_numpy(), (int(ar),))
            d["s_m6"] = v[el]
        d[K + ["s_m6"]].to_parquet(OUT / f"puntajes_m6_{a}.parquet", index=False)
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        n0 = len(b)
        b = b.merge(d, on=K, how="inner")
        assert len(b) == n0, "la prueba no son los mismos casos"
        b["y"] = b["y"].astype(bool)
        b["y_S"] = b["y"] & (b["nivel"] == 2)
        cortes[a] = b
        paso(f"corte {a} listo")
    M = {"m6": "s_m6", "m2": "s_m2", "tamano": "s_tamano_proveedor"}
    r = {yy: medir(cortes, yy, M, ["m2", "tamano"]) for yy in ("y", "y_S")}
    g = regla(r["y"]["metodos"]["m6"]["frente_a"]["m2"], r["y_S"]["metodos"]["m6"]["frente_a"]["m2"])
    res = {"candidata": el, "adopta": g, "precision": {yy: r[yy]["metodos"]["m6"]["precision"] for yy in r}, "frente_tamano": {yy: fmt(r[yy]["metodos"]["m6"]["frente_a"]["tamano"]) for yy in r},
           "por_anio": {yy: {k: v["razon"] for k, v in r[yy]["metodos"]["m6"]["frente_a"]["m2"]["por_anio"].items()} for yy in r}}
    (OUT / "m6_confirmar.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if g["se_adopta"]:
        (OUT / "vigente.json").write_text(json.dumps({"archivo": "puntajes_m6", "col": "s_m6", "nombre": el}), encoding="utf-8")
    paso(f"{res}")


if __name__ == "__main__":
    {"elegir": elegir, "confirmar": confirmar}[sys.argv[1]]()
    paso("listo")
