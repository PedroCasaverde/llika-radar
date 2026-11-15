"""Paso 23: panel proveedor-año con la etiqueta fechada por los hechos (PENDIENTES, sección 8, pasos 2 y 3). Solo arma
las tablas: no evalúa nada, así que no adelanta resultados del diseño que se pre-registra (HALLAZGOS 70 y 71).

Eventos de la etiqueta, uno por proveedor y hecho:
- Tribunal: cada infracción sancionada (decisión «sanciona») de las 10 837 resoluciones leídas con Claude, con su fecha
  de los hechos, para el administrado y, si es un consorcio, para cada integrante. El motivo sale del registro de
  sanciones (impedimento, documento falso o inexacto, resolución del contrato, otro). Lleva también la fecha de la
  resolución: cuándo se supo.
- Informes de control: cada proveedor con RUC válido que un hecho nombra como beneficiado, o cuya orden del hecho se
  halló en CONOSCE; fecha del hecho (o de la orden) y si el hecho describe un vínculo.
Las multas quedan fuera (aprobado por Pedro el 31/10).

Panel: un renglón por proveedor y año (2018-2025) en cualquiera de los tres canales (órdenes de CONOSCE, postulaciones
del OCDS, órdenes de catálogo de Perú Compras), con una marca por canal, y por cada año si hubo un hecho que terminó en
sanción o en informe, con la fecha más temprana en que se supo.

    python -I -X utf8 23_panel_etiqueta.py
Salida en datos/procesados/panel/ (fuera de git). Solo imprime agregados.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "datos" / "_scripts"))
from comun import SALIDA  # noqa: E402
from revisar_pilotos import doc, ruc_ok  # noqa: E402
from tce_resoluciones import normalizar  # noqa: E402

DATOS = Path(__file__).parents[1] / "datos" / "procesados"
OUT = DATOS / "panel"


def sin_trib(c):
    return re.sub(r"-T[A-Z]+", "", c) if isinstance(c, str) else None


def motivo(m):
    m = str(m).upper()
    if "IMPEDID" in m:
        return "impedimento"
    if "FALS" in m or "INEXACT" in m:
        return "documento_falso_o_inexacto"
    if "CONTRATO" in m and ("RESUELVA" in m or "RESOLUCI" in m or "RESCISI" in m):
        return "resolucion_de_contrato"
    return "otro"


def eventos_tribunal():
    sanc = pd.read_parquet(SALIDA / "sanciones.parquet")
    sanc["n2"] = sanc["resolucion"].map(lambda r: sin_trib(normalizar(r)))
    mot = sanc.groupby(["n2", "ruc"])["motivo"].first().map(motivo)
    filas = []
    for linea in (DATOS / "claude" / "resultados_resoluciones.jsonl").open(encoding="utf-8"):
        r = json.loads(linea)
        if "salida" not in r:
            continue
        s = r["salida"]
        n2 = sin_trib(normalizar(s["numero"] or r["custom_id"].replace("_", "-")))
        for e in s["expedientes"]:
            rucs = [doc(e["administrado"]["ruc"], 11)] + [doc(i["ruc"], 11) for i in e["integrantes_consorcio"]]
            rucs = [x for x in dict.fromkeys(rucs) if x and ruc_ok(x)]
            for inf in e["infracciones"]:
                if inf["decision"] != "sanciona" or not re.match(r"\d{4}-\d\d-\d\d", inf["fecha_hechos"] or ""):
                    continue
                for ruc in rucs:
                    filas.append({"ruc": ruc, "fuente": "tribunal", "fecha_hechos": inf["fecha_hechos"],
                                  "fecha_conocido": s["fecha_resolucion"] or None, "literal": inf["literal"],
                                  "motivo": mot.get((n2, ruc), "sin_registro"), "documento": n2, "vinculo": None})
    return pd.DataFrame(filas)


def eventos_informes():
    h = pd.read_parquet(DATOS / "validado" / "informes_hechos.parquet")
    p = pd.read_parquet(DATOS / "validado" / "informes_proveedores.parquet")
    o = pd.read_parquet(DATOS / "validado" / "informes_ordenes.parquet")
    fichas = pd.read_csv(Path(__file__).parents[1] / "datos" / "crudos" / "acc8_informes_cgr" / "fichas" /
                         "fichas_informes.csv", dtype=str, usecols=["CresCodigoFormato", "fecha_emision"])
    emision = fichas.drop_duplicates("CresCodigoFormato").set_index("CresCodigoFormato")["fecha_emision"]
    hh = h.set_index(["informe", "hecho"])
    a = p[p["ruc_valido"] & p["rol"].eq("beneficiado")][["informe", "hecho", "ruc"]]
    b = o[o["id_orden_conosce"].notna()][["informe", "hecho", "ruc_proveedor_conosce", "fecha"]].rename(
        columns={"ruc_proveedor_conosce": "ruc", "fecha": "fecha_orden"})
    e = pd.concat([a, b], ignore_index=True)
    e["desde"] = [hh["desde"].get((i, n)) for i, n in zip(e["informe"], e["hecho"])]
    e["vinculo"] = [hh["vinculo"].get((i, n)) for i, n in zip(e["informe"], e["hecho"])]
    fecha = e["fecha_orden"].where(e["fecha_orden"].astype(str).str.match(r"\d{4}-\d\d-\d\d"), e["desde"])
    e = e.assign(fuente="informe", fecha_hechos=fecha, fecha_conocido=e["informe"].map(emision), literal=None,
                 motivo="informe_de_control", documento=e["informe"])
    e = e[e["fecha_hechos"].astype(str).str.match(r"\d{4}-\d\d-\d\d")]
    return e[["ruc", "fuente", "fecha_hechos", "fecha_conocido", "literal", "motivo", "documento", "vinculo"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.concat([eventos_tribunal(), eventos_informes()], ignore_index=True)
    for c in ("fecha_hechos", "fecha_conocido"):
        # las fechas extraídas vienen en AAAA-MM-DD y las de las fichas en DD/MM/AAAA: cada una con su formato (leer
        # todo con dayfirst daba vuelta el día y el mes de las fechas ISO con día de 12 o menos)
        s = ev[c].astype(str).str[:10]
        iso = pd.to_datetime(s.where(s.str.match(r"\d{4}-\d\d-\d\d")), format="%Y-%m-%d", errors="coerce")
        dmy = pd.to_datetime(s.where(s.str.match(r"\d\d/\d\d/\d{4}")), format="%d/%m/%Y", errors="coerce")
        ev[c] = iso.fillna(dmy)
    ev = ev.dropna(subset=["fecha_hechos"]).drop_duplicates(["ruc", "fuente", "fecha_hechos", "documento", "motivo"])
    ev["anio_hechos"] = ev["fecha_hechos"].dt.year
    ev.to_parquet(OUT / "eventos_etiqueta.parquet", index=False)
    print(f"eventos: {len(ev):,} ({ev['ruc'].nunique():,} proveedores)")
    sin_sellado = ev[ev["anio_hechos"] != 2022]             # 2022 está sellado: no se imprime por año (160, lote 2)
    print(pd.crosstab(sin_sellado["anio_hechos"].clip(2010, 2026), sin_sellado["fuente"]).to_string())
    print(ev.groupby("fuente")["motivo"].value_counts().to_string())

    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["ruc", "ruc_entidad", "anio", "monto"])
    pan = o.groupby(["ruc", "anio"]).agg(ordenes=("monto", "size"), monto=("monto", "sum"),
                                         entidades=("ruc_entidad", "nunique")).reset_index()
    ex = pd.read_parquet(DATOS / "ocds" / "exposicion_proveedor_anio.parquet").rename(
        columns={"procesos": "ocds_procesos", "entidades": "ocds_entidades", "ganados": "ocds_ganados",
                 "monto_adjudicado": "ocds_monto"})
    # universo de tres canales (aprobado por Pedro el 1/11, HALLAZGOS 73): órdenes de CONOSCE, postulaciones del OCDS y
    # órdenes de catálogo de Perú Compras; cada fila dice por qué canales vendió o postuló ese año
    pan = pan.merge(ex[ex["anio"].between(2018, 2025)], on=["ruc", "anio"], how="outer")
    cat = DATOS / "perucompras" / "ordenes_catalogo.parquet"
    if cat.exists():
        c = pd.read_parquet(cat, columns=["ruc", "anio", "monto", "ruc_entidad"]).dropna(subset=["ruc", "anio"])
        c = c[c["anio"].between(2018, 2025)].astype({"anio": "int64"})
        c = c.groupby(["ruc", "anio"]).agg(cat_ordenes=("monto", "size"), cat_monto=("monto", "sum"),
                                           cat_entidades=("ruc_entidad", "nunique")).reset_index()
        pan = pan.astype({"anio": "int64"}).merge(c, on=["ruc", "anio"], how="outer")
    for col in ("ordenes", "monto", "entidades", "ocds_procesos", "ocds_ganados", "cat_ordenes", "cat_monto"):
        if col in pan:
            pan[col] = pan[col].fillna(0)
    pan["canal_conosce"] = pan["ordenes"] > 0
    pan["canal_ocds"] = pan["ocds_procesos"] > 0
    pan["canal_catalogo"] = pan["cat_ordenes"] > 0 if "cat_ordenes" in pan else False
    for fuente in ("tribunal", "informe"):
        g = ev[ev["fuente"] == fuente].groupby(["ruc", "anio_hechos"]).agg(
            **{f"hechos_{fuente}": ("documento", "nunique"), f"conocido_{fuente}": ("fecha_conocido", "min")})
        pan = pan.merge(g, left_on=["ruc", "anio"], right_index=True, how="left")
        pan[f"hechos_{fuente}"] = pan[f"hechos_{fuente}"].fillna(0).astype(int)
    for m in ("impedimento", "documento_falso_o_inexacto", "resolucion_de_contrato"):
        k = set(map(tuple, ev.loc[ev["motivo"] == m, ["ruc", "anio_hechos"]].to_numpy()))
        pan[f"y_{m}"] = [(r, a) in k for r, a in zip(pan["ruc"], pan["anio"])]
    pan["y_tribunal"] = pan["hechos_tribunal"] > 0
    pan["y_informe"] = pan["hechos_informe"] > 0
    pan["y"] = pan["y_tribunal"] | pan["y_informe"]
    pan.to_parquet(OUT / "panel_proveedor_anio.parquet", index=False)
    print(f"\npanel: {len(pan):,} pares proveedor-año, {pan['ruc'].nunique():,} proveedores")
    print(pan[pan["anio"] != 2022].groupby("anio")[["y_tribunal", "y_informe", "y"]].sum().to_string())


if __name__ == "__main__":
    main()
