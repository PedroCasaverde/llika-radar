"""Paso 29: data estructurada para el benchmark (HALLAZGOS 95). Solo arma tablas; no evalúa nada.

1. eventos.parquet: hechos del Tribunal por tipo, informes de control, penalidades y multas, con fecha de los hechos y
   fecha en que se supo (y si alguna es imputada).
2. x_proveedor_anio.parquet: una fila por proveedor-año (2018-2025) con lo que se sabía al 31/12 de ese año.
3. pares_exploracion.parquet: X de 2018, 2019 y 2020 con las y de t+1 y de t+1 a t+2 por tipo, sin nada del periodo
   sellado (comun.sellado). Los pares sellados no se escriben aquí.
4. diccionario.md y la revisión de fuga.

    python -I -X utf8 29_estructura_benchmark.py
Salida en datos/procesados/benchmark/ (fuera de git). Solo imprime agregados.
"""
import glob
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import CRUDOS, PROC, SALIDA, entidades, sellado  # noqa: E402
import lector_penalidades as LP  # noqa: E402

OUT = PROC / "benchmark"
ANIOS = list(range(2018, 2026))
T0 = time.time()
DIC = {}            # diccionario: columna -> (fuente, regla de fecha)


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def fin(a):
    return pd.Timestamp(f"{a}-12-31")


def anotar(cols, fuente, regla):
    for c in cols:
        DIC[c] = (fuente, regla)


# ----------------------------------------------------------------------------------------------- 1. eventos
TIPO = {"impedimento": "impedimento", "documento_falso_o_inexacto": "falsedad",
        "resolucion_de_contrato": "resolucion_contrato", "otro": "tribunal_otro", "sin_registro": "tribunal_otro",
        "informe_de_control": "informe"}


def penalidades_crudas():
    """Una fila por penalidad (RUC, contrato, fecha), con la primera versión en que apareció (lector único, 160 C1).
    Los consorcios sin RUC quedan fuera: no hay fuente abierta de sus integrantes."""
    p = LP.penalidades(CRUDOS / "oece_penalidades" / "_historia")
    return p[~p["es_consorcio"]].reset_index(drop=True)


def eventos():
    ev = pd.read_parquet(PROC / "panel/eventos_etiqueta.parquet")
    ev = pd.DataFrame({"ruc": ev["ruc"], "tipo": ev["motivo"].map(TIPO).fillna("tribunal_otro"), "fuente": ev["fuente"],
                       "documento": ev["documento"], "fecha_hechos": ev["fecha_hechos"],
                       "fecha_conocido": ev["fecha_conocido"], "conocido_imputado": False, "hechos_imputado": False,
                       "monto": np.nan})
    p, r = LP.fechar(penalidades_crudas())          # fecha de conocimiento (160, C2)
    print(f"demora de publicación: mediana general {r['mediana_dias']:.0f} días, p90 {r['p90_dias']:.0f}; entidades con "
          f"mediana propia: {r['entidades_con_mediana']:,}; anteriores a 11/2021: {r['anteriores']:,}, de ellas "
          f"{r['tardias']:,} aparecen después de 12/2021", flush=True)
    pen = pd.DataFrame({"ruc": p["ruc"], "tipo": "penalidad", "fuente": "oece_penalidades", "documento": p["contrato"],
                        "fecha_hechos": p["fecha"], "fecha_conocido": p["fecha_conocido"],
                        "conocido_mediana": p["conocido_mediana"], "conocido_p90": p["conocido_p90"],
                        "conocido_imputado": p["conocido_imputado"], "hechos_imputado": False, "monto": p["monto"]})
    ant = pd.read_parquet(SALIDA / "antecedentes.parquet")
    m = ant[ant["tipo"] == "multa"]
    mul = pd.DataFrame({"ruc": m["ruc"], "tipo": "multa", "fuente": "oece_multas", "documento": None,
                        "fecha_hechos": m["fecha"], "fecha_conocido": m["fecha"], "conocido_imputado": False,
                        "hechos_imputado": True, "monto": m["monto"]})      # la multa no trae la fecha de los hechos
    e = pd.concat([ev, pen, mul], ignore_index=True).dropna(subset=["ruc", "fecha_hechos", "fecha_conocido"])
    e["anio_hechos"] = e["fecha_hechos"].dt.year
    e["canal"] = np.where(e["tipo"] == "penalidad", "ocds", "")
    return e


# ----------------------------------------------------------------------------------------------- 2. X
def acumulado(df, col_fecha, nombre, rucs):
    """Cuántos eventos de cada RUC se conocían al 31/12 de cada año (acumulado desde el inicio)."""
    d = df[df["ruc"].isin(rucs)][["ruc", col_fecha]].copy()
    d["a"] = d[col_fecha].dt.year.clip(lower=2017)
    d = d[d["a"] <= max(ANIOS)]
    t = d.groupby(["ruc", "a"]).size().unstack(fill_value=0)
    for a in range(2017, max(ANIOS) + 1):
        if a not in t:
            t[a] = 0
    t = t[sorted(t.columns)].cumsum(axis=1)[ANIOS]
    out = t.stack().rename(nombre).reset_index().rename(columns={"a": "anio"})
    return out[out[nombre] > 0]


def bloque_v0():
    c = pd.read_parquet(SALIDA / "radar_casos.parquet", columns=["ruc", "anio", "puntaje"])
    v = c.groupby(["ruc", "anio"]).agg(v0_puntaje_max=("puntaje", "max"), v0_puntaje_suma=("puntaje", "sum"),
                                        v0_casos=("puntaje", "size")).reset_index()
    a = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["ruc", "anio", "tarjeta", "prob"])
    n = a.groupby(["ruc", "anio", "tarjeta"]).size().unstack(fill_value=0).add_prefix("alerta_").reset_index()
    pf = a[a["tarjeta"] == "familiar"].groupby(["ruc", "anio"])["prob"].max().rename("familiar_prob_max").reset_index()
    out = v.merge(n, on=["ruc", "anio"], how="outer").merge(pf, on=["ruc", "anio"], how="left")
    anotar([x for x in out.columns if x not in ("ruc", "anio")], "radar v0 (radar_casos, radar_alertas)",
           "órdenes del año; las alertas de funcionario, doble percepción y familiar usan declaraciones de 2022 y 2026 "
           "(disponibles en la práctica desde 2022)")
    return out


def bloque_antecedentes(ev, rucs):
    paso("antecedentes conocidos al cierre de cada año")
    e = sellado(ev)                                     # nada de hechos de 2022 ni de lo conocido desde 2026
    partes = [acumulado(e[e["tipo"] == t], "fecha_conocido", f"ant_{t}_conocidos", rucs)
              for t in ("impedimento", "falsedad", "resolucion_contrato", "tribunal_otro", "informe", "penalidad", "multa")]
    pen = e[e["tipo"] == "penalidad"]                   # prueba de sensibilidad de la fecha imputada
    partes.append(acumulado(pen, "conocido_mediana", "ant_penalidad_conocidos_mediana", rucs))
    partes.append(acumulado(pen, "conocido_p90", "ant_penalidad_conocidos_p90", rucs))
    s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio"]).dropna()
    partes.append(acumulado(s, "inicio", "ant_inhabilitaciones", rucs))
    out = partes[0]
    for p in partes[1:]:
        out = out.merge(p, on=["ruc", "anio"], how="outer")
    anotar([c for c in out.columns if c.startswith("ant_")], "eventos.parquet y registro de sanciones",
           "acumulado de lo conocido (o iniciado) hasta el 31/12 del año; sin hechos de 2022 ni lo conocido desde 2026")
    return out


def bloque_red():
    paso("red de socios y representantes")
    cortes = sorted((PROC / "acc3_conformacion").glob("conformacion_corte*.parquet"))
    fechas = {f: pd.Timestamp(f.stem[-8:]) for f in cortes}
    s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio"]).dropna()
    primera = s.groupby("ruc")["inicio"].min()
    salidas, cache = [], {}
    for a in ANIOS:
        validos = [f for f in cortes if fechas[f] <= fin(a)]
        f = validos[-1] if validos else cortes[0]
        if f not in cache:
            c = pd.read_parquet(f, columns=["ruc_proveedor", "dni_integrante", "usable_red_empresas"])
            c = c[c["usable_red_empresas"].fillna(False).astype(bool)].dropna(subset=["dni_integrante"]).drop_duplicates()
            per = c.groupby("dni_integrante")["ruc_proveedor"].nunique()
            c = c[c["dni_integrante"].isin(per[(per >= 2) & (per <= 50)].index)]
            e = c.merge(c, on="dni_integrante")
            e = e[e["ruc_proveedor_x"] != e["ruc_proveedor_y"]][["ruc_proveedor_x", "ruc_proveedor_y"]].drop_duplicates()
            con_socios = set(pd.read_parquet(f, columns=["ruc_proveedor"])["ruc_proveedor"])
            cache[f] = (e, con_socios)
        e, con_socios = cache[f]
        sanc = primera[primera <= fin(a)].index
        e2 = e.assign(v_inhab=e["ruc_proveedor_y"].isin(sanc))
        r = e2.groupby("ruc_proveedor_x").agg(red_vecinos=("ruc_proveedor_y", "size"),
                                             red_vecinos_inhabilitados=("v_inhab", "sum")).reset_index()
        r = r.rename(columns={"ruc_proveedor_x": "ruc"})
        r["anio"] = a
        r["red_corte_posterior"] = fechas[f] > fin(a)
        salidas.append((r, a, con_socios, fechas[f] > fin(a)))
    red = pd.concat([x[0] for x in salidas], ignore_index=True)
    red["red_parte_inhabilitados"] = red["red_vecinos_inhabilitados"] / red["red_vecinos"]
    socios = pd.concat([pd.DataFrame({"ruc": list(cs), "anio": a, "tiene_socios_registrados": True})
                        for _, a, cs, _ in salidas], ignore_index=True)
    anotar(["red_vecinos", "red_vecinos_inhabilitados", "red_parte_inhabilitados", "red_corte_posterior"],
           "conformación de proveedores del OECE (cortes desde 11/2021) y registro de sanciones",
           "corte de socios más reciente hasta el 31/12 del año; antes de 11/2021 se usa el primero (red_corte_posterior=True); "
           "inhabilitaciones iniciadas hasta el cierre")
    anotar(["tiene_socios_registrados"], "conformación de proveedores", "mismo corte que la red; falso = faltante como información")
    return red, socios


def bloque_ocds():
    paso("señales del OCDS y postor único")
    s = pd.read_parquet(SALIDA / "senales_ocds_proveedor_anio.parquet").add_prefix("ocds_").rename(
        columns={"ocds_ruc": "ruc", "ocds_anio": "anio"})
    po = pd.read_parquet(PROC / "ocds/postores.parquet", columns=["ocid", "ruc"])
    n = po.groupby("ocid")["ruc"].transform("size")
    unico = po[n == 1]
    # el postor único ganado va al año en que se adjudicó (HALLAZGOS 160, C4b); si falta la fecha, al de la convocatoria
    ad = pd.read_parquet(PROC / "ocds/adjudicaciones.parquet", columns=["ocid", "ruc", "fecha"])
    ad["anio_adj"] = pd.to_numeric(ad["fecha"].str[:4], errors="coerce")
    ad = ad.groupby(["ocid", "ruc"], as_index=False)["anio_adj"].min()
    pr = pd.read_parquet(PROC / "ocds/procesos.parquet", columns=["ocid", "fecha_convocatoria"])
    pr["anio_conv"] = pd.to_numeric(pr["fecha_convocatoria"].str[:4], errors="coerce")
    u = unico.merge(ad, on=["ocid", "ruc"]).merge(pr[["ocid", "anio_conv"]], on="ocid")
    u["anio"] = u["anio_adj"].fillna(u["anio_conv"])
    pu = u.groupby(["ruc", "anio"]).size().rename("ocds_ganados_postor_unico").reset_index()
    pu["anio"] = pu["anio"].astype(int)
    out = s.merge(pu, on=["ruc", "anio"], how="outer")
    anotar([c for c in out.columns if c.startswith("ocds_")], "OCDS del OECE (radar/26 y procesos)",
           "procesos convocados hasta el 31/12 del año (ventana t-2 a t en los pares)")
    return out


def bloque_territorial():
    t = pd.read_parquet(SALIDA / "tarjeta_territorial.parquet", columns=["ruc", "fecha"])
    t["anio"] = t["fecha"].dt.year
    out = t.groupby(["ruc", "anio"]).size().rename("territorial_ordenes").reset_index()
    anotar(["territorial_ordenes"], "tarjeta territorial (radar/25)", "órdenes del año; solo dato, su uso lo decide Pedro")
    return out


def tipo_entidad(nombre):
    n = str(nombre).upper()
    for k, v in (("MUNICIPALIDAD DISTRITAL", "municipalidad distrital"), ("MUNICIPALIDAD PROVINCIAL", "municipalidad provincial"),
                 ("MUNICIPALIDAD", "municipalidad"), ("GOBIERNO REGIONAL", "gobierno regional"), ("REGION ", "gobierno regional"),
                 ("MINISTERIO", "ministerio"), ("UNIVERSIDAD", "universidad"), ("HOSPITAL", "salud"), ("SALUD", "salud"),
                 ("ESSALUD", "salud"), ("PROGRAMA", "programa o proyecto"), ("PROYECTO", "programa o proyecto"),
                 ("EJERCITO", "defensa e interior"), ("POLICIA", "defensa e interior"), ("MARINA", "defensa e interior"),
                 ("FUERZA AEREA", "defensa e interior")):
        if k in n:
            return v
    return "otra"


def bloque_ordenes():
    paso("entidad principal, servicios y departamento")
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["ruc_entidad", "ruc", "monto", "servicio", "anio"])
    o = o[o["anio"].isin(ANIOS)]
    sv = o.groupby(["ruc", "anio"])["servicio"].mean().rename("parte_servicios").reset_index()
    pe = o.groupby(["ruc", "anio", "ruc_entidad"])["monto"].sum().reset_index()
    pe = pe.sort_values("monto").drop_duplicates(["ruc", "anio"], keep="last")
    ent = entidades().drop_duplicates("RUC").set_index("RUC")
    pe["depto_entidad_principal"] = pe["ruc_entidad"].map(ent["DEPARTAMENTO"])
    pe["tipo_entidad_principal"] = pe["ruc_entidad"].map(ent["NOMBRE_DE_ENTIDAD"]).map(tipo_entidad)
    out = pe[["ruc", "anio", "depto_entidad_principal", "tipo_entidad_principal"]].merge(sv, on=["ruc", "anio"], how="outer")
    anotar(["parte_servicios", "depto_entidad_principal", "tipo_entidad_principal"], "órdenes de CONOSCE y catálogo de entidades",
           "órdenes del año; la entidad principal es la de mayor monto")
    return out


def x_matriz(ev):
    paso("panel base")
    p = pd.read_parquet(PROC / "panel/panel_proveedor_anio.parquet",
                        columns=["ruc", "anio", "ordenes", "monto", "entidades", "ocds_procesos", "ocds_entidades",
                                 "en_consorcio", "ocds_ganados", "ocds_monto", "cat_ordenes", "cat_monto", "cat_entidades",
                                 "canal_conosce", "canal_ocds", "canal_catalogo"])
    p = p.rename(columns={"ocds_procesos": "exp_ocds_procesos", "ocds_entidades": "exp_ocds_entidades",
                          "ocds_ganados": "exp_ocds_ganados", "ocds_monto": "exp_ocds_monto"})
    anotar([c for c in p.columns if c not in ("ruc", "anio")], "panel proveedor-año (radar/23)", "actividad del año")
    rucs = set(p["ruc"])
    paso("historia del proveedor")
    p = p.sort_values(["ruc", "anio"])
    p["primer_anio"] = p.groupby("ruc")["anio"].transform("min")
    p["anios_previos_activo"] = p.groupby("ruc").cumcount()
    p["nuevo"] = (p["anio"] == p["primer_anio"]) & (p["anio"] > 2018)
    p["historia_desconocida"] = p["anio"] == 2018                 # 2018 es el inicio de los datos
    prev = p.groupby("ruc")["monto"].shift(1)
    prev_anio = p.groupby("ruc")["anio"].shift(1)
    p["crecimiento_monto"] = np.where(prev_anio == p["anio"] - 1, np.log1p(p["monto"].fillna(0)) - np.log1p(prev.fillna(0)), np.nan)
    p["sin_anio_previo"] = np.isnan(p["crecimiento_monto"])        # faltante como información (propuesta 92)
    p["persona_natural"] = p["ruc"].str.startswith("10")
    anotar(["primer_anio", "anios_previos_activo", "nuevo", "historia_desconocida", "crecimiento_monto", "sin_anio_previo",
            "persona_natural"],
           "panel proveedor-año", "solo años hasta el del renglón; nuevo = primer año visto (segmento C del Minedu)")
    for nombre, b in (("v0", bloque_v0()), ("antecedentes", bloque_antecedentes(ev, rucs)), ("ocds", bloque_ocds()),
                      ("territorial", bloque_territorial()), ("ordenes", bloque_ordenes())):
        p = p.merge(b, on=["ruc", "anio"], how="left")
    red, socios = bloque_red()
    p = p.merge(red, on=["ruc", "anio"], how="left").merge(socios, on=["ruc", "anio"], how="left")
    for c in ("tiene_socios_registrados", "ocds_postor_compania", "red_corte_posterior"):   # marcas sí/no
        p[c] = p[c].fillna(False).astype(bool)
    cnt = [c for c in p.columns if c.startswith(("alerta_", "ant_", "ocds_", "red_vecinos", "v0_", "territorial_"))
           and p[c].dtype != bool]
    p[cnt] = p[cnt].fillna(0)
    return p


# ----------------------------------------------------------------------------------------------- 3. pares
TIPOS_Y = ["impedimento", "falsedad", "resolucion_contrato", "tribunal_otro", "informe", "penalidad", "multa"]


def pares(x, ev):
    paso("pares de exploración")
    e = sellado(ev)
    e = e[e["fecha_conocido"] <= pd.Timestamp("2025-12-31")]
    filas = []
    for t in (2018, 2019, 2020):
        b = x[x["anio"] == t].copy()
        for tp in TIPOS_Y:
            h1 = set(e.loc[(e["tipo"] == tp) & (e["anio_hechos"] == t + 1), "ruc"])
            b[f"y1_{tp}"] = b["ruc"].isin(h1)
            if t + 2 <= 2021:
                h2 = set(e.loc[(e["tipo"] == tp) & e["anio_hechos"].between(t + 1, t + 2), "ruc"])
                b[f"y2_{tp}"] = b["ruc"].isin(h2)
            else:
                b[f"y2_{tp}"] = pd.NA                    # t+2 = 2022 está sellado
        trib = ["impedimento", "falsedad", "resolucion_contrato", "tribunal_otro"]
        b["y1_tribunal"] = b[[f"y1_{k}" for k in trib]].any(axis=1)
        b["y1_principal"] = b["y1_tribunal"] | b["y1_informe"]          # la y actual: sanción o informe
        filas.append(b)
    return pd.concat(filas, ignore_index=True)


# ----------------------------------------------------------------------------------------------- 4. fuga y diccionario
def revisar_fuga(x, ev):
    paso("revisión de fuga")
    problemas = []
    e = sellado(ev)
    for t in (2018, 2019, 2020):
        # los antecedentes del año t solo pueden contar eventos conocidos hasta el 31/12 de t
        conocidos = e[e["fecha_conocido"] <= fin(t)].groupby("ruc").size()
        col = x.loc[x["anio"] == t].set_index("ruc")[[c for c in x.columns if c.startswith("ant_") and c.endswith("_conocidos")]].sum(axis=1)
        exceso = (col - conocidos.reindex(col.index).fillna(0)) > 0
        if exceso.any():
            problemas.append(f"{t}: {int(exceso.sum())} proveedores con antecedentes mayores que lo conocido al cierre")
    sell = ev[(ev["anio_hechos"] == 2022) | (ev["fecha_conocido"] >= "2026-01-01")]
    if len(sell) and (x[x["anio"] <= 2020]["ant_informe_conocidos"].sum() < 0):
        problemas.append("conteo negativo")
    return problemas


def diccionario(x, pr):
    L = ["# Diccionario de la data del benchmark", "", "Generado por `radar/29_estructura_benchmark.py` (HALLAZGOS 95). "
         "Unidad: proveedor-año. Toda X usa lo que se sabía al 31/12 del año del renglón.", "",
         "| Columna | Tipo | % lleno o con valor | Fuente | Regla de fecha |", "|---|---|---|---|---|"]
    for c in x.columns:
        if c in ("ruc", "anio"):
            continue
        s = x[c]
        lleno = (s.fillna(0) != 0).mean() if s.dtype != object else s.notna().mean()
        f, r = DIC.get(c, ("", ""))
        L.append(f"| `{c}` | {s.dtype} | {100 * lleno:.1f} % | {f} | {r} |")
    L += ["", "## Y de los pares de exploración", "",
          "`y1_<tipo>`: hechos de ese tipo en t+1; `y2_<tipo>`: en t+1 o t+2 (vacío cuando t+2 = 2022, sellado). "
          "Solo hechos conocidos hasta el 31/12/2025.", "", "| Columna | t=2018 | t=2019 | t=2020 |", "|---|---|---|---|"]
    for c in [c for c in pr.columns if c.startswith(("y1_", "y2_"))]:
        v = pr.groupby("anio")[c].apply(lambda s: int(s.fillna(False).astype(bool).sum()) if s.notna().any() else None)
        L.append(f"| `{c}` | " + " | ".join("—" if v.get(a) is None else f"{v.get(a):,}" for a in (2018, 2019, 2020)) + " |")
    (OUT / "diccionario.md").write_text("\n".join(L), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paso("eventos")
    ev = eventos()
    ev.to_parquet(OUT / "eventos.parquet", index=False)
    print(ev.groupby("tipo").agg(eventos=("ruc", "size"), proveedores=("ruc", "nunique"),
                                 conocido_imputado=("conocido_imputado", "mean")).round(3).to_string(), flush=True)
    x = x_matriz(ev)
    x.to_parquet(OUT / "x_proveedor_anio.parquet", index=False)
    print(f"X: {len(x):,} filas, {x.shape[1] - 2} columnas", flush=True)
    pr = pares(x, ev)
    pr.to_parquet(OUT / "pares_exploracion.parquet", index=False)
    print(f"Pares de exploración: {len(pr):,} filas; y1_principal por año: "
          f"{pr.groupby('anio')['y1_principal'].sum().astype(int).to_dict()}", flush=True)
    prob = revisar_fuga(x, ev)
    print("Revisión de fuga: " + ("sin problemas" if not prob else "; ".join(prob)), flush=True)
    diccionario(x, pr)
    paso(f"listo: {OUT}")


if __name__ == "__main__":
    main()
