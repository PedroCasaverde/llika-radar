"""La lista única de LLIKA con el universo limpio (HALLAZGOS 160, secciones 88 y 89; Pedro, 11/11: «aplícalo»).

Usa lista_unica.py tal cual, sin modificarlo, y cambia dos cosas antes de armar la lista. Los modelos son los mismos.
    1. Salen de la lista y de los comparadores los casos con la mitad o más de su monto en compras por catálogo electrónico (acuerdo marco):
       la orden tiene una gemela en la tabla de Perú Compras (misma entidad, proveedor, año y monto, a 45 días o menos) o su texto lo dice.
    2. Las órdenes de pasajes no cuentan para la bandera del quinto puesto, ni se muestran como la orden con bandera.

    python -I -X utf8 lista_unica_limpia.py exploracion   # 2020, 2021 y 2023: repite 646 sin los cambios y 639 con ellos; rehace las comprobaciones
    python -I -X utf8 lista_unica_limpia.py segunda       # 2022 y 2024, una sola vez: repite 415 sin los cambios y después mide con ellos
    python -I -X utf8 lista_unica_limpia.py congelar      # 2025, una sola vez y sin etiquetas: repite la lista congelada anterior y congela la nueva
    python -I -X utf8 lista_unica_limpia.py expediente    # las diez señales sobre la lista nueva de 2025
Salida, fuera de git: resultados/lista_unica_limpia/. La carpeta resultados/lista_unica/ no se toca. Las listas traen RUC y nombres: no salen de datos/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import lista_unica as LU  # noqa: E402
import p1_tablas as P1  # noqa: E402
import sonda_nash as SN  # noqa: E402  (solo sus funciones de sorteo exacto)
from casos import DIR, SALIDA, TIPOS, eventos, top  # noqa: E402
from comun import PROC  # noqa: E402
from etiquetas import BASE, HOY_EXPLORACION, K, Hechos  # noqa: E402
from m6_proveedor import Fuentes  # noqa: E402

sys.argv = _a

SAL = LU.OUT / "lista_unica_limpia"
VIEJA = LU.SALIDA_L
K2 = ["ruc_entidad", "ruc"]
DICE_CATALOGO = r"ACUERDOS?\s+MARCO|CONVENIO\s+MARCO|CAT[AÁ]LOGO\s+ELECTR|PER[UÚ]\s*COMPRAS"
PASAJE = r"PASAJE|BOLETOS?\s+A[EÉ]RE|BOLETOS?\s+ELECTR"
AEROLINEA = r"AIRLINES|AEROL[IÍ]NEA|SKY AIRLINE|JETSMART|STAR PERU|AVIACION"
DIAS_GEMELA = 45
HECHOS = ["h_prov_penalidades", "h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones"]
TAUS = (0.05, 0.10, 0.20, 0.50)
ESTADO = {}
_CAT = None
_compras_original = LU.compras


def catalogo():
    global _CAT
    if _CAT is None:
        c = pd.read_parquet(PROC / "perucompras" / "ordenes_catalogo.parquet", columns=["ruc_entidad", "ruc", "fecha", "monto", "anio", "estado"]).astype({"ruc_entidad": str, "ruc": str})
        c = c[~c["estado"].eq("ORDEN DE COMPRA NULA")]
        _CAT = c.assign(fecha_c=pd.to_datetime(c["fecha"], errors="coerce"), mm=c["monto"].round(2))
    return _CAT


def _banderas(o):
    """La regla de lista_unica.compras sobre las órdenes que se le pasen: la bandera y la orden con bandera más grande de cada caso."""
    q = pl.from_pandas(o[["id", "ruc_entidad", "ruc", "fecha", "monto", "tope"]]).sort(["ruc_entidad", "ruc", "fecha", "id"]).with_columns(
        pl.col("fecha").cast(pl.Datetime("ms")), pl.col("monto").cast(pl.Float64), pl.col("tope").cast(pl.Float64), pl.lit(1).cast(pl.Int64).alias("uno"))
    q = q.with_columns(pl.col("monto").rolling_sum_by("fecha", "30d", closed="right").over(K2).alias("s"), pl.col("uno").rolling_sum_by("fecha", "30d", closed="right").over(K2).alias("n"))
    q = q.with_columns(((pl.col("n") >= 2) & (pl.col("s") > pl.col("tope"))).fill_null(False).alias("b30"), (pl.col("monto") >= 0.9 * pl.col("tope")).fill_null(False).alias("cerca"))
    q = q.with_columns((pl.col("b30") | pl.col("cerca")).alias("bandera"))
    p = q.sort("monto", descending=True)
    g = p.group_by(K2, maintain_order=True).agg(pl.col("b30").any().alias("f30"), pl.col("cerca").any().alias("f_cerca"))
    b = p.filter(pl.col("bandera")).group_by(K2, maintain_order=True).agg(pl.col("id").first().alias("orden_con_bandera"), pl.col("monto").first().alias("monto_orden_con_bandera"))
    return g.join(b, on=K2, how="left").to_pandas()


def compras_limpias(a):
    """Reemplaza a lista_unica.compras. Devuelve lo mismo, con la bandera sin pasajes, y agrega cat50 y las columnas *_todo con la regla anterior, para comprobar."""
    g0 = _compras_original(a).rename(columns={"f30": "f30_todo", "f_cerca": "f_cerca_todo", "orden_con_bandera": "orden_con_bandera_todo", "monto_orden_con_bandera": "monto_orden_con_bandera_todo"})
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "monto", "tope"], filters=[("anio", "==", a)]).astype({"ruc_entidad": str, "ruc": str})
    o["fecha"] = pd.to_datetime(o["fecha"])
    t = pd.read_parquet(BASE / "ordenes_texto.parquet", columns=["id", "descripcion"], filters=[("anio", "==", a)])
    d = t["descripcion"].fillna("").str.upper()
    t["dice"], t["pasaje"] = d.str.contains(DICE_CATALOGO, regex=True), d.str.contains(PASAJE, regex=True)
    o = o.merge(t[["id", "dice", "pasaje"]], on="id", how="left")
    o["dice"], o["pasaje"] = o["dice"].eq(True), o["pasaje"].eq(True)
    o["mm"] = o["monto"].round(2)
    c = catalogo()
    c = c[c["anio"] == a]
    x = o[["id"] + K2 + ["mm", "fecha"]].merge(c[K2 + ["mm", "fecha_c"]], on=K2 + ["mm"], how="inner")
    x = x[(x["fecha"] - x["fecha_c"]).dt.days.abs() <= DIAS_GEMELA]
    o["gemela"] = o["id"].isin(x["id"])
    o["cat"] = o["gemela"] | o["dice"]
    m = o.assign(mc=o["monto"] * o["cat"]).groupby(K2).agg(m=("monto", "sum"), mc=("mc", "sum")).reset_index()
    m["cat50"] = m["mc"] / m["m"].clip(lower=1e-9) >= 0.5
    g = g0.merge(m[K2 + ["cat50"]], on=K2, how="left").merge(_banderas(o[~o["pasaje"]]), on=K2, how="left")
    for col in ("f30", "f_cerca", "cat50", "f30_todo", "f_cerca_todo"):
        g[col] = g[col].eq(True)
    ESTADO[a] = {"ordenes": len(o), "con_gemela_pct": round(100 * float(o["gemela"].mean()), 2), "que_lo_dicen_pct": round(100 * float(o["dice"].mean()), 2), "de_catalogo_pct": round(100 * float(o["cat"].mean()), 2),
                 "de_pasajes_pct": round(100 * float(o["pasaje"].mean()), 2), "casos_de_catalogo": int(m["cat50"].sum()), "casos_de_catalogo_pct": round(100 * float(m["cat50"].mean()), 2)}
    return g


LU.compras = compras_limpias      # preparar_lista llama a compras: así la lista sale con la bandera sin pasajes y con cat50


def sin_cambios(q):
    """La lista de antes, para comprobar: todos los casos y la bandera con todas las órdenes."""
    return q.assign(f30=q["f30_todo"].eq(True), f_cerca=q["f_cerca_todo"].eq(True), orden_con_bandera=q["orden_con_bandera_todo"], monto_orden_con_bandera=q["monto_orden_con_bandera_todo"])


def limpio(q):
    return q[~q["cat50"].eq(True)].reset_index(drop=True)


def prov(L, y="y"):
    return int(L[L[y].astype(bool)].groupby(["ruc", "anio"]).ngroups)


def techo(q, y="y", k=5):
    """Lo máximo que vería una lista perfecta con k por entidad: flujo máximo entre proveedores-año con señal y entidades-año con k puestos."""
    e = q[q[y].astype(bool)][["ruc", "anio", "grupo"]].drop_duplicates()
    if e.empty:
        return 0
    pi, pu = pd.factorize(e["ruc"] + "|" + e["anio"].astype(str))
    gi, gu = pd.factorize(e["grupo"])
    n_p, n_g = len(pu), len(gu)
    fin = 1 + n_p + n_g
    filas = np.r_[np.zeros(n_p, dtype=int), 1 + pi, 1 + n_p + np.arange(n_g)]
    cols = np.r_[1 + np.arange(n_p), 1 + n_p + gi, np.full(n_g, fin)]
    cap = np.r_[np.ones(n_p, dtype=int), np.ones(len(e), dtype=int), np.full(n_g, k)]
    return int(maximum_flow(csr_matrix((cap.astype(np.int32), (filas, cols)), shape=(fin + 1, fin + 1)), 0, fin).flow_value)


def lista_k(q, k):
    """Como lista_unica.lista, con k por entidad: k − 1 por el puntaje y el último para la compra más grande con bandera."""
    filas = []
    for _, g in q.groupby("grupo", sort=False):
        if len(g) <= k:
            filas.extend(g.index.tolist())
            continue
        el = g.nlargest(k - 1, "s_lista").index.tolist()
        r = g.drop(index=el)
        b = r[r["f30"] | r["f_cerca"]]
        filas.extend(el + [(b if len(b) else r)["c_monto"].idxmax()])
    return q.loc[filas]


def motivos(F):
    """Por motivo, y el historial partido en con y sin hechos previos en la ficha (sección 79)."""
    con = F[HECHOS].astype(float).fillna(0).sum(axis=1) > 0
    m = F["motivo"].where(F["motivo"] != "historial del proveedor", pd.Series(np.where(con, "historial del proveedor, con antecedentes", "historial del proveedor, sin antecedentes"), index=F.index))
    return {n: {"filas": len(g), "proveedores_con_senal": prov(g), "por_100_filas": round(100 * prov(g) / len(g), 2)} for n, g in F.groupby(m)}


def universo(q, L):
    """Lo que hay por ver y lo que ve la lista, por verdad. Solo agregados."""
    r = {"casos": len(q), "entidades_anio": int(q["grupo"].nunique()), "techo_con_5": techo(q), "por_verdad": {}}
    for v, y in LU.VERDADES.items():
        u, l = prov(q, y), prov(L, y)
        r["por_verdad"][v] = {"con_senal": u, "lista": l, "no_vio": u - l, "alcance_pct": round(100 * l / max(u, 1), 1)}
    return r


def disuasion(q):
    """Probabilidad de revisión de cada proveedor-año con señal, con cada regla (como sonda_stackelberg), y el margen para salir de la lista (como sonda_nash). q: un año, con índice 0..N−1."""
    assert isinstance(q.index, pd.RangeIndex) and q.index.start == 0 and q["anio"].nunique() == 1
    base = LU.lista(q)
    N = len(q)
    en = np.zeros(N, bool)
    en[base.index.to_numpy()] = True
    quinto = np.zeros(N, bool)
    quinto[base.index[base["es_quinto"].to_numpy()].to_numpy()] = True
    g = pd.factorize(q["grupo"])[0]
    G = int(g.max()) + 1
    tam = np.bincount(g, minlength=G)
    grande = (tam > 5)[g]
    A4 = en & grande & ~quinto
    R4 = grande & ~A4
    s, punt = q["s_lista"].to_numpy(float), q["puntaje"].to_numpy(float)
    vinc = (q["grado"] >= 2).to_numpy()
    rc = pd.factorize(q["ruc"])[0]
    nr = int(rc.max()) + 1

    def marca(t):
        m = np.zeros(N)
        m[t.index.to_numpy()] = 1.0
        return m

    pi = {"lista": en.astype(float), "quinto sorteado por riesgo": np.where(~grande | A4, 1.0, SN.prob_uno(SN.preparar_sorteo(g, R4, punt), N)),
          "monto": marca(top(q, "s_monto")), "tamaño": marca(top(q, "s_tamano")), "cinco al azar": np.where(grande, 5 / tam[g], 1.0)}
    for r_, p_ in pi.items():
        assert np.allclose(np.bincount(g, weights=p_, minlength=G), np.minimum(tam, 5)), f"{r_}: no revisa cinco por entidad"
    d = {}
    for v, col in (("señal", "y"), ("hallazgo", "y_rev")):
        y = q[col].astype(bool).to_numpy()
        viol = np.bincount(rc[y], minlength=nr) > 0
        for r_, p_ in pi.items():
            lq = np.log(np.clip(1 - p_, 1e-15, 1.0))
            d[(v, r_)] = (1 - np.exp(np.bincount(rc[y], weights=lq[y], minlength=nr)))[viol]
    y = q["y"].astype(bool).to_numpy()
    o = np.lexsort((np.arange(N), -s, g))
    ini, lon = SN.tramos(g[o])
    s5 = np.full(G, np.nan)
    m5 = lon > 4
    s5[g[o][ini[m5]]] = s[o[ini[m5] + 4]]
    mm = A4 & ~vinc & y & (s > 0)
    return d, (1 - s5[g[mm]] / s[mm])


def comprobaciones(U):
    """Con el universo limpio de 2020, 2021 y 2023: cuántos por entidad, disuasión, techo y alcance con el orden de la web."""
    q = pd.concat(U.values())
    con_senal = prov(q)
    r = {"proveedores_con_senal": con_senal, "techo_con_5": sum(techo(u) for u in U.values()), "por_entidad": {}, "alcance_por_orden_de_la_web": {}}
    for k in (3, 5, 7, 10):
        L = pd.concat([lista_k(u, k) for u in U.values()])
        M = pd.concat([top(u, "s_monto", k=k) for u in U.values()])
        Z = pd.concat([top(u, "s_tamano", k=k) for u in U.values()])
        x = LU.ic_proveedores(L, M)
        r["por_entidad"][k] = {"filas": len(L), "lista": prov(L), "monto": prov(M), "tamaño": prov(Z), "alcance_pct": round(100 * prov(L) / con_senal, 1), "hallazgos_lista": prov(L, "y_rev"), "hallazgos_monto": prov(M, "y_rev"),
                               "frente_al_monto": round(x["razon"], 3), "ic90": [round(v, 3) for v in x["ic90"]]}
        LU.paso(f"con {k} por entidad: {r['por_entidad'][k]}")
    assert r["por_entidad"][5]["lista"] == prov(pd.concat([LU.lista(u) for u in U.values()])), "lista_k con 5 no es la lista"
    orden = q.sort_values(["grupo", "s_lista"], ascending=[True, False])
    puesto = orden.groupby("grupo").cumcount()
    for k in (5, 10, 20, 50, 100):
        t = orden[puesto < k]
        r["alcance_por_orden_de_la_web"][k] = {"filas": len(t), "proveedores": prov(t), "pct": round(100 * prov(t) / con_senal, 1)}
    D, margen = {}, []
    for a, u in U.items():
        d, m = disuasion(u)
        margen.extend(m.tolist())
        for k_, v in d.items():
            D.setdefault(k_, []).append(v)
    r["disuasion"] = {}
    for (v, regla), partes in D.items():
        dv = np.concatenate(partes)
        f = {"proveedores": int(len(dv)), "aciertos_esperados": round(float(dv.sum()), 1), "probabilidad_media_pct": round(100 * float(dv.mean()), 2)}
        if v == "señal":
            f["disuadidos_segun_tau"] = {f"{t:g}": int((dv >= t - 1e-12).sum()) for t in TAUS}
        r["disuasion"].setdefault(v, {})[regla] = f
    fija = round(r["disuasion"]["señal"]["lista"]["aciertos_esperados"])
    o = np.sort(np.concatenate(D[("señal", "quinto sorteado por riesgo")]))[::-1]
    r["disuasion"]["sortear_gana_hasta_tau"] = round(float(o[fija - 1]), 4)
    x = np.array(margen)
    r["margen_para_salir"] = {"filas": int(len(x)), "mediana_pct": round(100 * float(np.median(x)), 1), "p25_pct": round(100 * float(np.percentile(x, 25)), 1), "p75_pct": round(100 * float(np.percentile(x, 75)), 1)}
    return r


def huellas():
    return {**LU.scripts(), "lista_unica_limpia.py": LU.huella(__file__)[:16], "sonda_nash.py": LU.huella(Path(__file__).parent / "sonda_nash.py")[:16]}


def guardar(r, nombre, titulo, notas):
    (SAL / f"{nombre}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (SAL / f"{nombre}.md").write_text(LU.md(r, titulo) + "\n" + "\n".join(notas) + "\n", encoding="utf-8")
    print(LU.md(r, titulo), flush=True)


def igual_a_la_guardada(F0, ruta):
    """La lista rehecha sin los cambios debe tener las mismas filas, motivos y órdenes visibles que la guardada."""
    V = pd.read_parquet(ruta, columns=K + ["motivo", "orden_visible"]).astype(TIPOS)
    m = F0[K + ["motivo", "orden_visible"]].astype(TIPOS).merge(V, on=K, how="outer", suffixes=("", "_v"), indicator=True)
    return {"filas": len(F0), "filas_guardadas": len(V), "llaves_distintas": int((m["_merge"] != "both").sum()), "motivos_distintos": int((m["motivo"] != m["motivo_v"]).sum()),
            "ordenes_distintas": int((m["orden_visible"].astype(str) != m["orden_visible_v"].astype(str)).sum())}


def exploracion():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    E = eventos()
    Pa = pd.read_parquet(DIR / "proveedor_anio.parquet").astype({"ruc": str, "anio": int})
    cprov, n = LU.comunes()
    partes, U, fichas, antes, comprobado = {}, {}, [], {}, {}
    for a, anios in LU.EXPLORACION.items():
        d = LU.armar_anio(a, anios, H, E, Pa, cprov, n, HOY_EXPLORACION)
        g = pd.read_parquet(SALIDA / f"sonda_pasado_{a}.parquet", columns=K + ["s_S_limpio", "s_P_limpio"]).astype(TIPOS)
        m = d[K + ["s_S", "s_P"]].merge(g, on=K, how="left")
        comprobado[a] = {"dif_S": float((m["s_S"] - m["s_S_limpio"]).abs().max()), "dif_P": float((m["s_P"] - m["s_P_limpio"]).abs().max())}
        assert max(comprobado[a].values()) < 1e-6, "los modelos no reproducen la sonda"
        q0 = LU.preparar_lista(d, a)
        del d
        antes[a] = prov(LU.lista(sin_cambios(q0)))
        q = limpio(q0)
        L = LU.lista(q)
        F = LU.ficha(L.copy(), a, n)
        F.to_parquet(SAL / f"lista_{a}.parquet", index=False)
        fichas.append(F)
        partes[a] = LU.partes_de(q, L)
        U[a] = q
        LU.paso(f"ETAPA {a}: {ESTADO[a]}; sin los cambios {antes[a]}; con ellos {prov(L)} en {len(L):,} filas")
        del q0
    r = LU.tabla(partes)
    total = r["por_verdad"]["señal adversa"]["lista única"]["proveedores"]
    assert sum(antes.values()) == 646, f"sin los cambios debía repetir 646 y dio {sum(antes.values())}"
    assert total == 639, f"con los cambios debía repetir 639 y dio {total}"
    LU.paso("reproduce 646 sin los cambios y 639 con ellos")
    F = pd.concat(fichas)
    r["comprobado"], r["motivos"], r["ordenes"], r["antes_por_anio"] = comprobado, motivos(F), ESTADO.copy(), antes
    r["universo"] = universo(pd.concat(U.values()), pd.concat([p["lista única"] for p in partes.values()]))
    r["comprobaciones"] = comprobaciones(U)
    r["huellas"] = huellas()
    guardar(r, "tabla", "Lista única de LLIKA, universo limpio: 2020, 2021 y 2023 (exploración; cifras optimistas)", ["Sin compras por catálogo y con la bandera sin pasajes (HALLAZGOS 160, sección 89). Antes de la limpieza: 646."])
    print(json.dumps({k: r[k] for k in ("universo", "motivos", "comprobaciones")}, ensure_ascii=False, indent=1, default=float), flush=True)


def segunda():
    assert not (SAL / "segunda_medicion.json").exists(), "la segunda medición con el universo limpio se corre una sola vez"
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F_ = Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    cprov, n = LU.comunes()
    vieja = json.loads((VIEJA / "segunda_medicion.json").read_text(encoding="utf-8"))
    partes, U, fichas, comprobado = {}, [], [], {}
    for a, anios in LU.SEGUNDA.items():
        Pa = P1.tabla_proveedor(H, F_, anios + [a]).astype({"ruc": str, "anio": int})
        LU.paso(f"ETAPA {a}: tabla de proveedores lista")
        d = LU.armar_anio(a, anios, H, E, Pa, cprov, n, LU.FECHA_SEGUNDA)
        ex = pd.read_parquet(SALIDA / f"examen_puntajes_{a}.parquet", columns=K + LU.ETIQUETAS + ["alerta", "s_monto"]).astype(TIPOS)
        m = d[K + LU.ETIQUETAS + ["alerta", "c_monto"]].merge(ex, on=K, how="outer", suffixes=("", "_ex"), indicator=True)
        c = {"llaves_distintas": int((m["_merge"] != "both").sum()), **{x: int((m[x].astype(bool) != m[f"{x}_ex"].astype(bool)).sum()) for x in LU.ETIQUETAS + ["alerta"]},
             "dif_monto": float((m["c_monto"].astype(float) - m["s_monto"].astype(float)).abs().max())}
        assert c["llaves_distintas"] == 0 and sum(c[x] for x in LU.ETIQUETAS + ["alerta"]) == 0 and c["dif_monto"] < 1, "no coincide con el examen: se detiene antes de medir"
        q0 = LU.preparar_lista(d, a)
        del d, Pa, m, ex
        L0 = LU.lista(sin_cambios(q0))
        c["sin_los_cambios"] = {**igual_a_la_guardada(LU.ficha(L0.copy(), a, n), VIEJA / f"lista_{a}.parquet"), "proveedores": prov(L0), "proveedores_guardados": vieja["por_anio"][str(a)]["lista única"]["proveedores"]}
        s = c["sin_los_cambios"]
        LU.paso(f"sin los cambios, {a}: {s}")
        assert s["llaves_distintas"] == 0 and s["motivos_distintos"] == 0 and s["ordenes_distintas"] == 0 and s["proveedores"] == s["proveedores_guardados"], "no reproduce la lista guardada: se detiene antes de medir"
        q = limpio(q0)
        L = LU.lista(q)
        F = LU.ficha(L.copy(), a, n)
        F.to_parquet(SAL / f"lista_{a}.parquet", index=False)
        fichas.append(F)
        partes[a] = LU.partes_de(q, L)
        U.append(q[K + ["grupo"] + LU.ETIQUETAS])
        c["casos_de_catalogo_fuera"] = int(q0["cat50"].sum())
        c["filas_de_la_anterior_que_eran_de_catalogo"] = int(L0["cat50"].sum())
        comprobado[a] = c
        LU.paso(f"ETAPA {a}: {ESTADO[a]}; lista limpia {len(L):,} filas; proveedores con señal {prov(L)}")
        del q0
    r = LU.tabla(partes)
    F = pd.concat(fichas)
    r["comprobado"], r["motivos"], r["ordenes"] = comprobado, motivos(F), ESTADO.copy()
    r["universo"] = universo(pd.concat(U), pd.concat([p["lista única"] for p in partes.values()]))
    n_f = r["por_verdad"]["señal adversa"]["lista única"]["filas"]
    r["en_vano_por_acierto"] = {x: round((n_f - r["por_verdad"]["señal adversa"][x]["proveedores"]) / max(r["por_verdad"]["señal adversa"][x]["proveedores"], 1), 1) for x in ("lista única", "monto", "tamaño", "azar")}
    r["antes_de_la_limpieza"] = {x: vieja["por_verdad"]["señal adversa"][x] for x in vieja["por_verdad"]["señal adversa"]}
    r["rotulo"] = "segunda medición, no ciega, rehecha por la limpieza de catálogo y pasajes (sección 89)"
    r["huellas"] = huellas()
    guardar(r, "segunda_medicion", "Lista única de LLIKA, universo limpio: 2022 y 2024 (segunda medición, no ciega)", ["Rehecha una vez por la limpieza de catálogo y pasajes (HALLAZGOS 160, sección 89). Antes: 415 frente a 153."])
    print(json.dumps({k: r[k] for k in ("universo", "motivos", "en_vano_por_acierto", "razones", "razones_por_anio", "comprobado")}, ensure_ascii=False, indent=1, default=float), flush=True)


def congelar():
    man, sal = SAL / "lista_2025_manifiesto.json", SAL / "lista_2025_unica.parquet"
    assert not man.exists(), "la lista limpia de 2025 se congela una sola vez"
    vman = json.loads((VIEJA / "lista_2025_manifiesto.json").read_text(encoding="utf-8"))
    anterior = VIEJA / "lista_2025_unica.parquet"
    assert LU.huella(anterior) == vman["sha256_lista"], "la lista congelada anterior no tiene su huella"
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F_ = Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    cprov, n = LU.comunes()
    Pa = P1.tabla_proveedor(H, F_, LU.AJUSTE_LISTA + [LU.ANIO_LISTA]).astype({"ruc": str, "anio": int})
    LU.paso("ETAPA 2025: tabla de proveedores lista")
    d = LU.armar_anio(LU.ANIO_LISTA, LU.AJUSTE_LISTA, H, E, Pa, cprov, n, LU.CORTE_LISTA, con_etiquetas=False)
    assert not any(c.startswith(("y", "fuentes", "conocido", "peso")) for c in d.columns), "la lista no guarda etiquetas"
    q0 = LU.preparar_lista(d, LU.ANIO_LISTA)
    del d, Pa
    L0 = LU.lista(sin_cambios(q0))
    s = igual_a_la_guardada(LU.ficha(L0.copy(), LU.ANIO_LISTA, n, con_etiquetas=False), anterior)
    LU.paso(f"sin los cambios, 2025: {s}")
    assert s["llaves_distintas"] == 0 and s["motivos_distintos"] == 0 and s["ordenes_distintas"] == 0, "no reproduce la lista congelada anterior: se detiene antes de congelar"
    q = limpio(q0)
    L = LU.lista(q)
    F = LU.ficha(L.copy(), LU.ANIO_LISTA, n, con_etiquetas=False)
    assert not any(c.startswith(("y", "fuentes", "conocido", "peso")) for c in F.columns), "la lista no guarda etiquetas"
    F.to_parquet(sal, index=False)
    assert LU.huella(anterior) == vman["sha256_lista"], "la lista congelada anterior cambió"
    aer = lambda t: int(t["ruc"].map(n).fillna("").str.upper().str.contains(AEROLINEA, regex=True).sum())  # noqa: E731
    k3 = ["ruc_entidad", "ruc"]
    nuevas = L[k3].merge(L0[k3], on=k3, how="left", indicator=True)
    m = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "anio": LU.ANIO_LISTA, "ajuste": LU.AJUSTE_LISTA, "corte": LU.CORTE_LISTA, "filas": len(F), "entidades": int(F["ruc_entidad"].nunique()), "proveedores": int(F["ruc"].nunique()),
         "motivos": F["motivo"].value_counts().to_dict(), "sha256_lista": LU.huella(sal), "sha256_lista_anterior_sin_tocar": vman["sha256_lista"], "sha256_eventos": LU.huella(DIR / "eventos_examen.parquet"),
         "ordenes": ESTADO[LU.ANIO_LISTA], "reproduce_la_anterior": s,
         "cambios": {"casos_en_el_universo": len(q0), "casos_de_catalogo_fuera": int(q0["cat50"].sum()), "filas_de_la_anterior_que_eran_de_catalogo": int(L0["cat50"].sum()), "filas_que_no_estaban_en_la_anterior": int((nuevas["_merge"] == "left_only").sum()),
                     "filas_de_aerolineas_antes": aer(L0), "filas_de_aerolineas_despues": aer(L), "filas_con_bandera_antes": int((L0["f30"] | L0["f_cerca"]).sum()), "filas_con_bandera_despues": int((L["f30"] | L["f_cerca"]).sum()),
                     "personas_naturales_pct": round(100 * float(F["ruc"].astype(str).str.startswith("10").mean()), 1)},
         "huellas": huellas()}
    man.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=1), flush=True)
    LU.paso(f"CONGELADA: {m['filas']:,} filas; {m['entidades']:,} entidades; huella {m['sha256_lista'][:16]}; la anterior sigue en {vman['sha256_lista'][:16]}")


def expediente():
    import expediente as X
    X.SALIDA_L = SAL
    sys.argv = [sys.argv[0], str(LU.ANIO_LISTA)]
    X.main()


def main():
    SAL.mkdir(parents=True, exist_ok=True)
    modo = sys.argv[1] if len(sys.argv) > 1 else "exploracion"
    {"exploracion": exploracion, "segunda": segunda, "congelar": congelar, "expediente": expediente}[modo]()
    LU.paso(f"LISTA LIMPIA {modo}")


if __name__ == "__main__":
    main()
