"""Data v4, capa 2: X por caso (entidad × proveedor × año), 2018-2025, SIN etiquetas ni filtros (HALLAZGOS 118).

Lo que depende de los hechos (antecedentes, informes de la entidad, riesgo histórico) NO está aquí: lo da la capa 3
(etiquetas.Hechos.historia) a la fecha y con el sellado. El universo va como banderas (u_*), no como filtro.

Correcciones frente a la v3 (HALLAZGOS 113): tope de 9 UIT en 2022; «sin cruce de texto» separado de «texto vacío»;
historia de la relación con «no se sabe» en 2018 y ventanas fijas (t−1, t−2) en vez de acumulados; sin k-medias;
servicios básicos con el número de entidades del proveedor contado hasta t; el paso a logaritmo lo decide la capa 4
con los datos de entrenamiento.

Sale en datos/procesados/v4/: x_caso_anio.parquet (float32) y diccionario_x.csv / .md (grupo, tipo, clase temporal,
fuente y nota de cada variable). Al final corre las pruebas de la capa.

    python -I -X utf8 capa2_x.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from comun import PROC, SALIDA, funcionarios  # noqa: E402

BASE = PROC / "v4" / "base"
OUT = PROC / "v4"
K = ["ruc_entidad", "ruc", "anio"]
T0 = time.time()
COMO_ANTES = "--como-antes" in sys.argv       # c_desc_repetida vacía sin repetidas: solo para la prueba de igualdad (160)
TRABAJO = (r"APOYO|ASISTENTE|LOCACION|LOCADOR|ESPECIALISTA|TECNICO|PROFESIONAL|AUXILIAR|PROMOTOR|DIGITADOR|SECRETARI|"
           r"CHOFER|CONDUCTOR|VIGILANTE|COORDINADOR|ANALISTA|ABOGAD|CONTADOR|INGENIERO|OPERADOR|RESPONSABLE")
KW_BASICO = (r"CONSUMO DE AGUA|SERVICIO DE AGUA|SERV\.? DE AGUA|AGUA POTABLE|ENERG.A EL.CTRICA|SUMINISTRO EL.CTRICO|"
             r"SUMINISTRO DE ENERG|TELEFON.A|SERVICIO DE INTERNET|GAS NATURAL|RECIBO DE (LUZ|AGUA)|PAGO DE (LUZ|AGUA|TEL)")
GENERICA = r"^(SERVICIO|SERVICIOS|ADQUISICION|ADQUISICIÓN|COMPRA|BIENES|SERVICIO DE APOYO|SERVICIO DIVERSO|SERVICIOS DIVERSOS)\.?$"
RUBROS = {
    "obras": r"OBRA|CONSTRUC|EDIFIC|CARRETER|PAVIMENT|ASFALT|MANTENIMIENTO DE (LA |EL |LOS |LAS )?(VIA|LOCAL|INFRAEST|CAMINO|TROCHA)",
    "consultoria": r"CONSULTOR|ESTUDIO|EXPEDIENTE TECNICO|ASESOR|SUPERVISION|AUDITOR",
    "alimentos": r"ALIMENT|V.VERES|RACION|DESAYUNO|ALMUERZO|REFRIGERIO|COMIDA|PAN |LECHE",
    "combustible": r"COMBUSTIBLE|PETR.LEO|GASOLIN|DIESEL|D.ESEL|GLP|LUBRICANTE",
    "salud": r"MEDICAMENT|MEDIC|LABORATORIO|SALUD|HOSPITAL|FARMAC|ODONTOL|REACTIVO",
    "limpieza_seguridad": r"LIMPIEZA|VIGILANCIA|SEGURIDAD|FUMIGA|DESINFEC",
    "alquiler_transporte": r"ALQUILER|ARRENDAMIENTO|TRANSPORTE|MOVILIDAD|FLETE|TRASLADO|CAMIONETA|VEHICULO",
    "servicios_personales": r"SERVICIO DE APOYO|ASISTENTE|LOCACION|ESPECIALISTA|TECNICO EN|PROFESIONAL|AUXILIAR|PROMOTOR|DIGITADOR",
    "tecnologia": r"COMPUTADORA|SOFTWARE|IMPRESORA|TONER|INFORMATIC|LAPTOP|SERVIDOR|REDES|CABLEADO",
    "utiles": r"UTILES|PAPEL|OFICINA|ESCRITORIO|MATERIAL DE",
    "publicidad_eventos": r"PUBLICIDAD|DIFUSION|SPOT|EVENTO|IMPRESION|BANNER|GIGANTOGRAFIA",
}
# columnas de la matriz proveedor-año (paso 29) que NO dependen de etiquetas; los ant_* pasan a la capa 3
P_CONTEMPORANEAS = ["ordenes", "monto", "entidades", "exp_ocds_procesos", "exp_ocds_entidades", "exp_ocds_ganados", "exp_ocds_monto",
                    "cat_ordenes", "cat_monto", "cat_entidades", "anios_previos_activo", "nuevo", "historia_desconocida",
                    "crecimiento_monto", "sin_anio_previo", "parte_servicios", "ocds_n_pares_persistentes", "ocds_n_pares_rotan",
                    "ocds_postor_compania", "ocds_postula_inhabilitado", "ocds_gana_inhabilitado", "ocds_monto_gana_inhabilitado",
                    "ocds_entidades_perdio_y_orden", "ocds_ordenes_tras_perder", "ocds_monto_tras_perder", "ocds_ganados_postor_unico"]
P_ANACRONICAS = ["v0_puntaje_max", "v0_puntaje_suma", "v0_casos", "alerta_doble_percepcion", "alerta_empresa_espejo", "alerta_familiar",
                 "alerta_funcionario_propio", "alerta_mismo_dueno", "alerta_sancionado", "territorial_ordenes", "red_vecinos",
                 "red_vecinos_inhabilitados", "red_corte_posterior", "red_parte_inhabilitados", "tiene_socios_registrados"]


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def relacion(o):
    paso("relación: montos, órdenes, meses, tope (9 UIT en 2022), rachas")
    o = o.copy()
    o["cerca"] = (o["monto"] >= 0.8 * o["tope"]) & (o["monto"] <= o["tope"])
    o["mes"] = o["fecha"].dt.month
    o["menor_10"] = o["monto"] < 10
    c = o.groupby(K).agg(c_ordenes=("id", "size"), c_monto=("monto", "sum"), c_monto_medio=("monto", "mean"), c_monto_max=("monto", "max"),
                         c_parte_servicios=("servicio", "mean"), c_cerca_tope=("cerca", "sum"), c_meses=("mes", "nunique"),
                         c_ordenes_menores_10=("menor_10", "sum")).reset_index()
    d = o.groupby(K + ["fecha"]).size().rename("n").reset_index()
    dd = d[d["n"] >= 2].groupby(K).agg(c_dias_varias=("n", "size"), c_max_mismo_dia=("n", "max")).reset_index()
    m = o.groupby(K + ["mes"]).agg(s=("monto", "sum"), n=("id", "size"), tope=("tope", "first")).reset_index()
    sup = m[(m["n"] >= 2) & (m["s"] > m["tope"])].groupby(K).size().rename("c_meses_superan_tope").reset_index()
    g = m.groupby(K)["s"].agg(["mean", "std"]).reset_index()
    g["c_cv_mensual"] = (g["std"].fillna(0) / g["mean"]).fillna(0)
    m = m.sort_values(K + ["mes"])
    m["bloque"] = m.groupby(K)["mes"].diff().ne(1).groupby([m[k] for k in K]).cumsum()
    racha = m.groupby(K + ["bloque"]).size().groupby(level=[0, 1, 2]).max().rename("c_racha").reset_index()
    dic = m.assign(dd=m["mes"].eq(12) * m["s"]).groupby(K).agg(dd=("dd", "sum"), t=("s", "sum"), c_primer_mes=("mes", "min"),
                                                                 c_ultimo_mes=("mes", "max")).reset_index()
    dic["c_parte_diciembre"] = dic["dd"] / dic["t"]
    for x in (dd, sup, g[K + ["c_cv_mensual"]], racha, dic[K + ["c_parte_diciembre", "c_primer_mes", "c_ultimo_mes"]]):
        c = c.merge(x, on=K, how="left")
    c[["c_dias_varias", "c_max_mismo_dia", "c_meses_superan_tope"]] = c[["c_dias_varias", "c_max_mismo_dia", "c_meses_superan_tope"]].fillna(0)
    ge, gp = c.groupby(["ruc_entidad", "anio"]), c.groupby(["ruc", "anio"])
    c["c_parte_gasto_entidad"] = c["c_monto"] / ge["c_monto"].transform("sum")
    c["c_parte_ventas_proveedor"] = c["c_monto"] / gp["c_monto"].transform("sum")
    c["c_puesto_en_entidad"] = ge["c_monto"].rank(ascending=False, method="min")
    c["c_proveedores_entidad"] = ge["ruc"].transform("size")
    c["c_entidades_del_proveedor"] = gp["ruc_entidad"].transform("size")
    return c


def _suma_kahan_pandas(m, st, g_ini):
    """Réplica vectorizada (entre grupos) de la suma móvil de pandas (roll_sum): Kahan con compensaciones de suma y de resta
    separadas, reinicio cuando la ventana no se solapa con la anterior y el arreglo de valores iguales consecutivos."""
    n = len(m)
    pos = np.arange(n) - g_ini                     # posición dentro del grupo
    orden = np.lexsort((np.arange(n), pos))        # por posición; dentro, por fila
    cortes = np.searchsorted(pos[orden], np.arange(pos.max() + 2))
    sx, ca, cr = np.zeros(n), np.zeros(n), np.zeros(n)   # estado por grupo, guardado en la fila de inicio del grupo
    nobs, prev, ncons = np.zeros(n, np.int64), np.zeros(n), np.zeros(n, np.int64)
    out = np.empty(n)
    for p in range(pos.max() + 1):
        i = orden[cortes[p]:cortes[p + 1]]         # filas en la posición p de su grupo
        gi, s = g_ini[i], st[i]
        fresca = (p == 0) | (s >= i)
        f, fg = i[fresca], gi[fresca]              # ventanas nuevas: se reinicia con la fila i
        if len(f):
            v = m[f]
            y = v - 0.0
            tt = 0.0 + y
            ca[fg], sx[fg], cr[fg], nobs[fg], prev[fg], ncons[fg] = tt - 0.0 - y, tt, 0.0, 1, v, 1
        d, dg = i[~fresca], gi[~fresca]            # ventanas que se deslizan: salen filas y entra la fila i
        if len(d):
            s_prev, s_new = st[d - 1], st[d]
            nrem = s_new - s_prev
            for r in range(int(nrem.max())):
                a = nrem > r
                j, q = s_prev[a] + r, dg[a]
                y = -m[j] - cr[q]
                tt = sx[q] + y
                cr[q], sx[q] = tt - sx[q] - y, tt
                nobs[q] -= 1
            v = m[d]
            y = v - ca[dg]
            tt = sx[dg] + y
            ca[dg], sx[dg] = tt - sx[dg] - y, tt
            nobs[dg] += 1
            ncons[dg] = np.where(v == prev[dg], ncons[dg] + 1, 1)
            prev[dg] = v
        out[i] = np.where(ncons[gi] >= nobs[gi], prev[gi] * nobs[gi], sx[gi])
    return out


def _ventanas_anio(o):
    d = pl.from_pandas(o[K + ["id", "fecha", "monto", "tope"]]).sort(K + ["fecha", "id"])
    d = d.with_columns(pl.struct(K).rle_id().alias("g"))
    g = d["g"].to_numpy().astype(np.int64)
    t = d["fecha"].dt.epoch("s").to_numpy().astype(np.int64)
    m, tp = d["monto"].to_numpy().astype(np.float64), d["tope"].to_numpy().astype(np.float64)
    n = len(g)
    ini = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    g_ini = np.repeat(ini, np.diff(np.r_[ini, n]))
    mismo = g[1:] == g[:-1]
    # el máximo móvil del tope es el tope de la fila si no baja dentro del grupo (2022: de 8 a 9 UIT)
    assert (tp[1:][mismo] >= tp[:-1][mismo]).all(), "tope no monótono dentro de un grupo"
    llave = g * (int(t.max() - t.min()) + 31 * 86400 + 1) + (t - t.min())
    cols = {}
    for dias in (7, 30):
        st = np.searchsorted(llave, llave - dias * 86400, side="right")   # primera fila con fecha > t − días, en el mismo grupo
        suma = _suma_kahan_pandas(m, st, g_ini)
        pasa = ((np.arange(n) - st + 1) >= 2) & (suma > tp)
        cols[f"c_max_suma_{dias}d"] = np.maximum.reduceat(suma, ini)
        cols[f"c_pasa_tope_{dias}d"] = np.add.reduceat(pasa.astype(np.int64), ini)
    res = d[ini].select(K).to_pandas()
    for c in ("c_max_suma_7d", "c_pasa_tope_7d", "c_max_suma_30d", "c_pasa_tope_30d"):
        res[c] = cols[c]
    return res


def ventanas(o):
    """Da lo mismo que groupby().rolling() de pandas, 26 veces más rápido (HALLAZGOS 160, sección 12)."""
    paso("fraccionamiento en ventanas móviles de 7 y 30 días")
    return pd.concat([_ventanas_anio(o[o["anio"] == a]) for a in sorted(o["anio"].unique())], ignore_index=True)


def _texto_anio(o, a):
    f = BASE / "ordenes_texto.parquet"
    lf = pl.scan_parquet(f)
    filtro = pl.col("anio") == a if "anio" in lf.collect_schema() else pl.col("id").is_between(int(o["id"].min()), int(o["id"].max()))
    t = lf.filter(filtro).select(["id", "descripcion", "cruce_texto", "texto_vacio"]).collect()
    x = pl.from_pandas(o[["id"] + K]).join(t, on="id", how="left")
    x = x.with_columns(u=pl.col("descripcion").fill_null("").str.to_uppercase().str.strip_chars(),
                       c_texto_sin_cruce=~pl.col("cruce_texto").fill_null(False), c_texto_vacio=pl.col("texto_vacio").fill_null(False))
    con = ~pl.col("c_texto_sin_cruce") & ~pl.col("c_texto_vacio")
    nw = pl.col("u").str.count_matches(r"[^\s\x1c-\x1f]+")      # igual que len(str.split()) de Python

    def si(e, nombre):
        return pl.when(con).then(e.cast(pl.Float64)).otherwise(None).alias(nombre)
    ex = [pl.col("c_texto_sin_cruce").cast(pl.Float64), pl.col("c_texto_vacio").cast(pl.Float64),
          si(pl.col("u").str.contains(TRABAJO), "c_kw_trabajo"), si(pl.col("u").str.contains(KW_BASICO), "c_kw_basico"),
          si(pl.col("u").str.contains(GENERICA) | (nw <= 2), "c_desc_generica"), si(nw, "c_desc_palabras")]
    ex += [si(pl.col("u").str.contains(rx), f"c_rub_{k}") for k, rx in RUBROS.items()]
    r = x.select(K + ["u", con.alias("_con")] + ex)
    cols = [c for c in r.columns if c.startswith("c_")]
    agg = r.group_by(K).agg([pl.col(c).mean() for c in cols])
    cnt = r.filter(pl.col("_con")).group_by(K + ["u"]).len()
    rep = cnt.group_by(K).agg(num=pl.col("len").filter(pl.col("len") >= 2).sum(), den=pl.col("len").sum(), hay=(pl.col("len") >= 2).any())
    # sin descripciones repetidas es 0; antes quedaba vacía por un defecto de la división en pandas (160, G)
    rep = rep.select(K + [pl.when(pl.col("hay")).then(pl.col("num") / pl.col("den"))
                          .otherwise(None if COMO_ANTES else 0.0).alias("c_desc_repetida")])
    return agg.join(rep, on=K, how="left").to_pandas()


def texto(o):
    """Una sola pasada con polars por año: 9 veces más rápido y sin cargar los 23 millones de textos juntos (160, sección 12)."""
    paso("texto de las órdenes: trabajo, básico, genérico, rubros, repetidos; sin cruce separado de vacío")
    return pd.concat([_texto_anio(o[o["anio"] == a], int(a)) for a in sorted(o["anio"].unique())], ignore_index=True)


def historia_relacion(c):
    paso("historia de la relación: año previo y dos años antes (no se sabe en 2018)")
    llave = set(zip(c["ruc_entidad"], c["ruc"], c["anio"]))
    monto = c.set_index(K)["c_monto"]
    prev = [(e, r, a - 1) in llave for e, r, a in zip(c["ruc_entidad"], c["ruc"], c["anio"])]
    prev2 = [(e, r, a - 2) in llave for e, r, a in zip(c["ruc_entidad"], c["ruc"], c["anio"])]
    c["c_rel_anio_previo"] = np.where(c["anio"] > 2018, prev, np.nan)
    c["c_rel_dos_anios_antes"] = np.where(c["anio"] > 2019, prev2, np.nan)
    mp = monto.reindex(pd.MultiIndex.from_arrays([c["ruc_entidad"], c["ruc"], c["anio"] - 1])).to_numpy()
    c["c_crec_relacion"] = np.where(c["anio"] > 2018, np.where(np.isnan(mp), np.nan, np.log1p(c["c_monto"]) - np.log1p(mp)), np.nan)
    return c


def pares_y_entidad(c, o, ent):
    paso("relativas a los pares y contexto de la entidad")
    ge = c.groupby(["ruc_entidad", "anio"])
    c["c_monto_pct_entidad"] = ge["c_monto"].rank(pct=True)
    c["c_ordenes_pct_entidad"] = ge["c_ordenes"].rank(pct=True)
    c["c_monto_medio_pct_entidad"] = ge["c_monto_medio"].rank(pct=True)
    c["c_monto_rel_mediana_entidad"] = c["c_monto"] / ge["c_monto"].transform("median")
    rub = [x for x in c.columns if x.startswith("c_rub_")]
    c["rubro_dom"] = np.where(c[rub].fillna(0).max(axis=1) > 0, c[rub].fillna(0).idxmax(axis=1).str[6:], "otro")
    e = ent.set_index("ruc_entidad")
    c["e_tipo"] = c["ruc_entidad"].map(e["tipo"]).fillna("otra")
    c["e_depto"] = c["ruc_entidad"].map(e["departamento"]).fillna("sin dato")
    c["c_monto_pct_tipo_rubro"] = c.groupby(["e_tipo", "rubro_dom", "anio"])["c_monto"].rank(pct=True)
    tot = c.groupby(["ruc_entidad", "anio"]).agg(e_proveedores=("ruc", "size"), e_monto_total=("c_monto", "sum"),
                                                e_parte_natural=("ruc", lambda s: s.str.startswith("10").mean())).reset_index()
    o2 = o.assign(cerca=(o["monto"] >= 0.8 * o["tope"]) & (o["monto"] <= o["tope"]))
    eb = o2.groupby(["ruc_entidad", "anio"]).agg(e_ordenes=("id", "size"), e_parte_cerca_tope=("cerca", "mean")).reset_index()
    c = c.merge(tot, on=["ruc_entidad", "anio"], how="left").merge(eb, on=["ruc_entidad", "anio"], how="left")
    c["c_cerca_tope_rel"] = c["c_cerca_tope"] / c["c_ordenes"] - c["e_parte_cerca_tope"]
    c["e_hhi_proveedores"] = c.groupby(["ruc_entidad", "anio"])["c_parte_gasto_entidad"].transform(lambda s: float((s ** 2).sum()))
    f = funcionarios()
    d = f[f["decide"]]
    dec = []
    for a in sorted(c["anio"].unique()):
        ini, fin = pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{a}-12-31")
        dec.append(d[(d["inicio"] <= fin) & (d["cese"] >= ini)].groupby("ruc_entidad")["persona"].nunique()
                   .rename("e_declarantes_decision").reset_index().assign(anio=a))
    c = c.merge(pd.concat(dec).astype({"ruc_entidad": str}), on=["ruc_entidad", "anio"], how="left")
    c["e_declarantes_decision"] = c["e_declarantes_decision"].fillna(0)
    return c


def alertas_y_proveedor(c):
    paso("alertas del caso y contexto del proveedor sin etiquetas")
    a = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["ruc_entidad", "ruc", "anio", "tarjeta", "prob"]).astype({"ruc_entidad": str, "ruc": str})
    n = a.groupby(K + ["tarjeta"]).size().unstack(fill_value=0).add_prefix("a_").reset_index()
    pf = a[a["tarjeta"] == "familiar"].groupby(K)["prob"].max().rename("a_familiar_prob").reset_index()
    v = pd.read_parquet(SALIDA / "radar_casos.parquet", columns=K + ["puntaje"]).rename(columns={"puntaje": "a_puntaje_radar"}).astype({"ruc_entidad": str, "ruc": str})
    t = pd.read_parquet(SALIDA / "tarjeta_territorial.parquet", columns=["ruc_entidad", "ruc", "fecha"]).astype({"ruc_entidad": str, "ruc": str})
    t = t.assign(anio=t["fecha"].dt.year).groupby(K).size().rename("a_territorial").reset_index()
    for x in (n, pf, v, t):
        c = c.merge(x, on=K, how="left")
    acols = [x for x in c.columns if x.startswith("a_")]
    c[acols] = c[acols].fillna(0)
    px = pd.read_parquet(PROC / "benchmark" / "x_proveedor_anio.parquet", columns=["ruc", "anio"] + P_CONTEMPORANEAS + P_ANACRONICAS)
    px = px.rename(columns={x: f"p_{x}" for x in P_CONTEMPORANEAS + P_ANACRONICAS})
    px["p_historia_desconocida"] = px["p_historia_desconocida"].astype(float).where(px["anio"] > 2018)   # en 2018 es el año, no una variable
    for col in ("p_anios_previos_activo", "p_nuevo", "p_sin_anio_previo", "p_crecimiento_monto"):
        px[col] = px[col].astype(float).where(px["anio"] > 2018)
    return c.merge(px.astype({"ruc": str}), on=["ruc", "anio"], how="left")


def ubicacion_y_universo(c, o, prov):
    paso("ubicación (padrón 2026), RUC reciente y banderas de universo")
    p = prov.set_index("ruc")
    e = pd.read_parquet(BASE / "entidades.parquet").set_index("ruc_entidad")
    ub_p, ub_e = c["ruc"].map(p["ubigeo_padron_2026"]), c["ruc_entidad"].map(e["ubigeo_padron_2026"])
    con = ub_p.notna() & ub_e.notna()
    for nombre, n in (("distrito", 6), ("provincia", 4), ("departamento", 2)):
        c[f"c_local_{nombre}"] = np.where(con, ub_p.str[:n] == ub_e.str[:n], np.nan)
    c["c_sin_ubicacion"] = ~con
    dire = (ub_p.fillna("") + "|" + c["ruc"].map(p["direccion_padron_2026"]).fillna("")).where(c["ruc"].map(p["direccion_padron_2026"]).notna())
    c["c_misma_direccion_en_entidad"] = (c.assign(_d=dire).dropna(subset=["_d"]).groupby(["ruc_entidad", "anio", "_d"])["ruc"]
                                         .transform("size") - 1).reindex(c.index)
    emp = c["ruc"].str.startswith("20")
    c["c_ruc_reciente_pct"] = np.nan
    c.loc[emp, "c_ruc_reciente_pct"] = pd.to_numeric(c.loc[emp, "ruc"], errors="coerce").groupby(c.loc[emp, "anio"]).rank(pct=True)
    c["p_n_rubros"] = c.groupby(["ruc", "anio"])["rubro_dom"].transform("nunique")
    # entidades del proveedor hasta el año t (no hasta 2025): corrige la fuga menor del filtro de la v3
    ol = pl.from_pandas(o[["ruc", "ruc_entidad", "anio"]])
    primera = ol.group_by(["ruc", "ruc_entidad"]).agg(pl.col("anio").min().alias("a0"))        # año en que aparece cada par
    nuevas = primera.group_by(["ruc", "a0"]).len().rename({"a0": "anio", "len": "nuevas"})
    ne = (ol.select(["ruc", "anio"]).unique().join(nuevas, on=["ruc", "anio"], how="left").with_columns(pl.col("nuevas").fill_null(0))
          .sort(["ruc", "anio"]).with_columns(pl.col("nuevas").cum_sum().over("ruc").alias("p_entidades_hasta_t"))
          .select(["ruc", "anio", "p_entidades_hasta_t"]).to_pandas())
    c = c.merge(ne, on=["ruc", "anio"], how="left")
    c["u_servicio_basico"] = (c["ruc"].map(p["servicio_basico_lista"]).fillna(False).astype(bool)
                              | (emp & (c["c_kw_basico"].fillna(0) >= 0.8) & (c["p_entidades_hasta_t"] >= 30)))
    c["u_proveedor_publico"] = c["ruc"].map(p["es_entidad_publica"]).fillna(False).astype(bool)
    c["u_persona_natural"] = c["ruc"].str.startswith("10")
    c["u_recurrente_empresa"] = emp & (c["c_meses"] >= 6) & ((c["c_cv_mensual"] < 0.5) | (c["c_racha"] >= 4))
    c["c_pts_trabajador"] = (((c["c_meses"] >= 6).astype(int) + (c["c_racha"] >= 4).astype(int) + (c["c_cv_mensual"] < 0.35).astype(int)
                              + (c["c_parte_servicios"] >= 0.95).astype(int) + (c["c_kw_trabajo"].fillna(0) >= 0.5).astype(int))
                             * c["u_persona_natural"])
    return c


def banderas_lote2(c):
    """Banderas de las «otras», del registro y de las anuladas (HALLAZGOS 160, C3); fuera de las variables por empezar con u_."""
    f = BASE / "ordenes.parquet"
    if "otra_categoria" not in pl.scan_parquet(f).collect_schema():
        return c
    paso("banderas del lote 2: otras, fecha de registro y anuladas")
    o = pl.scan_parquet(f).select(K + ["monto", "otra_categoria", "fecha_registro_primera", "anulada_mismo_dia_o_despues", "ultima_version_otras"])
    o = o.with_columns(v=pl.col("fecha_registro_primera") <= pl.datetime(pl.col("anio").cast(pl.Int32) + 1, 3, 31, 23, 59, 59))
    g = o.group_by(K).agg(u_parte_otras=pl.col("otra_categoria").mean(), u_solo_otras=pl.col("otra_categoria").all(),
                          u_visible_0331=pl.col("v").any(), u_parte_monto_visible_0331=(pl.col("monto") * pl.col("v")).sum() / pl.col("monto").sum(),
                          u_parte_anuladas_mismo_dia=pl.col("anulada_mismo_dia_o_despues").mean(),
                          u_parte_version_otras=pl.col("ultima_version_otras").mean()).collect(engine="streaming").to_pandas()
    e = o.group_by(["ruc_entidad", "anio"]).agg(u_entidad_parte_otras=pl.col("otra_categoria").mean()).collect(engine="streaming").to_pandas()
    return c.merge(g, on=K, how="left").merge(e, on=["ruc_entidad", "anio"], how="left")


def diccionario(c):
    filas = []
    for col in c.columns:
        if col in K:
            continue
        grupo = {"c": "relación", "a": "alertas", "p": "proveedor", "e": "entidad", "u": "universo"}.get(col.split("_")[0], "otra")
        if col.startswith("a_") and col not in ("a_sancionado",):
            clase, fuente = "anacrónica", "declaraciones juradas de 2022 y 2026 y cortes de socios desde 11/2021 (radar v0)"
        elif col in [f"p_{x}" for x in P_ANACRONICAS]:
            clase, fuente = "anacrónica", "radar v0, socios desde 11/2021 o foto territorial de 10/2026"
        elif col.startswith(("c_local_", "c_sin_ubicacion", "c_misma_direccion")):
            clase, fuente = "estática fechable (foto de 2026)", "padrón reducido de SUNAT"
        elif col == "e_declarantes_decision":
            clase, fuente = "anacrónica", "declaraciones juradas de 2022 y 2026"
        else:
            clase, fuente = "contemporánea", "órdenes de CONOSCE, OCDS o catálogo, del mismo año o anteriores"
        s = c[col]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            tipo = "categórica"
        else:
            v = s.dropna().astype(float)
            tipo = "binaria" if v.isin([0, 1]).all() else "proporción" if v.between(0, 1).all() else "conteo o monto"
        filas.append({"variable": col, "grupo": grupo, "tipo": tipo, "clase_temporal": clase, "fuente": fuente,
                      "lleno_pct": round(100 * float(s.notna().mean()), 1)})
    return pd.DataFrame(filas)


def pruebas(c, d):
    paso("pruebas de la capa 2")
    fallas = []
    if c.duplicated(K).any():
        fallas.append("casos repetidos")
    prohibidas = [x for x in c.columns if x.startswith(("y", "te_", "ant_", "h_", "ya_", "conocido"))]
    if prohibidas:
        fallas.append(f"columnas derivadas de etiquetas en la capa 2: {prohibidas}")
    num = c.select_dtypes("number")
    inf = [x for x in num.columns if np.isinf(num[x].to_numpy(dtype=float, na_value=np.nan)).any()]
    if inf:
        fallas.append(f"infinitos en {inf}")
    print("   pruebas:", "todas bien" if not fallas else fallas, flush=True)


def main():
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "monto", "servicio", "anio", "tope"])
    ent = pd.read_parquet(BASE / "entidades.parquet")
    prov = pd.read_parquet(BASE / "proveedores.parquet")
    c = relacion(o)
    c = c.merge(ventanas(o), on=K, how="left").merge(texto(o), on=K, how="left")
    c = historia_relacion(c)
    c = pares_y_entidad(c, o, ent)
    c = alertas_y_proveedor(c)
    c = ubicacion_y_universo(c, o, prov)
    c = banderas_lote2(c)
    num = [x for x in c.columns if x not in K and (pd.api.types.is_numeric_dtype(c[x]) or pd.api.types.is_bool_dtype(c[x]))]
    c[num] = c[num].astype("float32")
    c["anio"] = c["anio"].astype("int16")
    c.to_parquet(OUT / "x_caso_anio.parquet", index=False)          # se guarda antes del diccionario: si algo falla después, no se pierde
    paso("X guardada; diccionario y pruebas")
    d = diccionario(c)
    pruebas(c, d)
    d.to_csv(OUT / "diccionario_x.csv", index=False, encoding="utf-8")
    md = ["# Diccionario de la capa 2 (X por caso-año, data v4)", "", "| Variable | Grupo | Tipo | Clase temporal | Fuente | % lleno |",
          "|---|---|---|---|---|---|"] + [f"| `{r.variable}` | {r.grupo} | {r.tipo} | {r.clase_temporal} | {r.fuente} | {r.lleno_pct} |"
                                           for r in d.itertuples()]
    (OUT / "diccionario_x.md").write_text("\n".join(md), encoding="utf-8")
    paso(f"listo: {len(c):,} casos (2018-2025), {c.shape[1]} columnas; clases: {d['clase_temporal'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
