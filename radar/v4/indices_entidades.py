"""Los seis índices de cada entidad en 2025 y su lugar entre todas las entidades de la lista (etapa B del cierre, 17/11/2026).

La pantalla calculaba los índices con los 100 primeros proveedores de cada entidad y los comparaba solo con las entidades
que traía su JSON. Este script los calcula para las 2 647 entidades de la lista de 2025, con todos sus proveedores y con
las reglas de la ficha:

    vin    con vínculo: es una autoridad de la zona o su empresa, o tiene los dos apellidos de una (grados 3 y 2)
    ant    con antecedentes: penalidad, multa o inhabilitación publicada hasta el 31/12/2025
    fra    compras partidas: dos o más compras en 30 días que suman más que el tope, o una de 90 % del tope o más
    tope   cerca del tope: parte de las compras que llega al 90 % del tope
    conc   le compra a pocos: parte del monto que se llevan sus cinco proveedores más grandes
    dep    dependen de ella: le venden a la entidad el 80 % o más de lo que le venden al Estado

    python -I -X utf8 indices_entidades.py

Necesita puntajes_2025.py ya corrido. Solo lee: no toca la lista congelada ni los puntajes. No corre ningún modelo.
Salidas, en resultados/lista_unica_limpia/:
    indices_2025.parquet              una fila por entidad, con sus seis índices y el lugar de cada uno (fuera de git)
    distribucion_indices_2025.json    los valores de cada índice, ordenados y sin el RUC de la entidad: es lo que la
                                      pantalla lleva para comparar; su copia va en la fuente de la pantalla, en git
    indices_2025_manifiesto.json      conteos, huellas y tiempo
El «lugar» de una entidad en un índice es la parte de las entidades con un valor más bajo (su percentil). Los valores
se guardan en centésimas de punto porcentual, para que la pantalla y este script comparen los mismos enteros.
Solo imprime cifras agregadas.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
CRUDOS, PROC = RAIZ / "datos" / "crudos", RAIZ / "datos" / "procesados"
BASE = PROC / "v4" / "base"
SAL = PROC / "v4" / "resultados" / "lista_unica_limpia"
ANIO, CORTE, TOPE = 2025, pd.Timestamp("2025-12-31"), 42800.0
HUELLA_LISTA = "189af867041d1270bf132ab6ca25041e9a3e782713171758dadf22a761a4aff2"
# Versiones de las fuentes publicadas justo después del corte del 31/12/2025 (lo conocido al cierre del año)
PENALIDADES = ["oece_penalidades/_historia/penalidades_v260_2026-01-03.csv", "oece_penalidades/_historia/penalidades2018_v249_2026-01-03.csv"]
MULTAS = "oece_multas/_historia/sancionados_multa_v265_2026-01-03.csv"
IND = ["vin", "ant", "fra", "tope", "conc", "dep"]
K2 = ["ruc_entidad", "ruc"]
PUNTAJES, LISTA = SAL / "puntajes_2025.parquet", SAL / "lista_2025_unica.parquet"
TABLA, DISTRIBUCION, MANIFIESTO = SAL / "indices_2025.parquet", SAL / "distribucion_indices_2025.json", SAL / "indices_2025_manifiesto.json"
T0 = time.time()


def paso(t):
    print(f"[{time.time() - T0:5.1f} s] {t}", flush=True)


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def con_antecedentes():
    """Los RUC con una penalidad, una multa o una inhabilitación publicada hasta el corte: los filtros de la ficha
    (exportar_entidad.antecedentes), para todos los proveedores."""
    pen = pd.concat([pd.read_csv(CRUDOS / f, sep="|", dtype=str, encoding="cp1252", encoding_errors="replace") for f in PENALIDADES], ignore_index=True)
    pen.columns = [c.strip().upper() for c in pen.columns]
    f = pen["FECHA PENALIDAD"].str.strip()
    f = pd.to_datetime(f, errors="coerce", format="%d/%m/%y").fillna(pd.to_datetime(f, errors="coerce", format="%d/%m/%Y"))
    mul = pd.read_csv(CRUDOS / MULTAS, sep="|", dtype=str, encoding="cp1252", encoding_errors="replace")
    san = pd.read_parquet(BASE / "sanciones.parquet", columns=["ruc", "inicio"]).astype({"ruc": str})
    return (set(pen.loc[f <= CORTE, "RUC CONTRATISTA"]) | set(mul.loc[pd.to_datetime(mul["FECHA_INICIO"], errors="coerce", format="%Y%m%d") <= CORTE, "RUC"])
            | set(san.loc[pd.to_datetime(san["inicio"]) <= CORTE, "ruc"]))


def indices(casos, ordenes, ant):
    """Los seis índices, en porcentaje, de las entidades de «casos» (ruc_entidad, ruc, grado, c_local) con sus compras
    del año (id, ruc_entidad, ruc, fecha, monto). Es la regla de la página: las compras partidas se miran por día, con
    todas las compras del proveedor en la entidad."""
    q = pl.from_pandas(ordenes[["id"] + K2 + ["fecha", "monto"]]).sort(K2 + ["fecha", "id"]).with_columns(
        pl.col("fecha").cast(pl.Datetime("ms")).dt.truncate("1d"), pl.col("monto").cast(pl.Float64), pl.lit(1).cast(pl.Int64).alias("uno"))
    q = q.with_columns(pl.col("monto").rolling_sum_by("fecha", "30d", closed="right").over(K2).alias("s"), pl.col("uno").rolling_sum_by("fecha", "30d", closed="right").over(K2).alias("n"))
    q = q.with_columns(((pl.col("n") >= 2) & (pl.col("s") > TOPE)).alias("b30"), (pl.col("monto") >= 0.9 * TOPE).alias("cerca"))
    g = q.group_by(K2).agg(pl.col("b30").any().alias("b30"), pl.col("cerca").sum().alias("n_cerca"), pl.len().alias("n_ord"), pl.col("monto").sum().alias("total")).to_pandas()
    c = casos[K2 + ["grado", "c_local"]].merge(g, on=K2, how="left", validate="one_to_one")
    assert c["n_ord"].notna().all(), "hay casos sin compras en el año"
    c["vin"] = c["grado"] >= 2
    c["ant"] = c["ruc"].isin(ant)
    c["fra"] = c["b30"] | (c["n_cerca"] > 0)
    c["dep"] = np.round(100 * c["c_local"].astype(float)) >= 80
    c["t5"] = c["total"].where(c.groupby("ruc_entidad")["total"].rank(method="first", ascending=False) <= 5, 0.0)
    e = c.groupby("ruc_entidad").agg(casos=("ruc", "size"), vin=("vin", "sum"), ant=("ant", "sum"), fra=("fra", "sum"), dep=("dep", "sum"),
                                     n_cerca=("n_cerca", "sum"), compras=("n_ord", "sum"), t5=("t5", "sum"), total=("total", "sum"))
    return pd.DataFrame({"casos": e["casos"].astype(int), "compras": e["compras"].astype(int), "vin": 100 * e["vin"] / e["casos"], "ant": 100 * e["ant"] / e["casos"],
                         "fra": 100 * e["fra"] / e["casos"], "tope": 100 * e["n_cerca"] / e["compras"], "conc": 100 * e["t5"] / e["total"], "dep": 100 * e["dep"] / e["casos"]})


def indices_de_la_pagina(datos):
    """Los mismos índices con la regla escrita como en la página (plantilla.html), sobre los casos de un JSON de la
    pantalla. Sirve para comprobar que «indices» y la página dicen lo mismo con los mismos casos."""
    from datetime import date
    tope, out = float(datos["tope"]), {}
    for e in datos["entidades"]:
        cs, n = e["casos"], len(e["casos"])

        def partidas(c):
            os_ = sorted((date.fromisoformat(o[0]), float(o[3] or 0)) for o in c["ord"])
            dentro = lambda t: [m for t2, m in os_ if 0 <= (t - t2).days < 30]  # noqa: E731
            return any(m >= 0.9 * tope for _, m in os_) or any(len(dentro(t)) >= 2 and sum(dentro(t)) > tope for t, _ in os_)
        totales = sorted((sum(float(o[3] or 0) for o in c["ord"]) for c in cs), reverse=True)
        compras = [float(o[3] or 0) for c in cs for o in c["ord"]]
        out[e["ruc"]] = {"vin": 100 * sum(1 for c in cs if c.get("vin")) / n, "ant": 100 * sum(1 for c in cs if any((c.get("ant") or {}).get(k) for k in ("pen", "inh", "mul"))) / n,
                         "fra": 100 * sum(1 for c in cs if partidas(c)) / n, "tope": 100 * sum(1 for m in compras if m >= 0.9 * tope) / len(compras),
                         "conc": 100 * sum(totales[:5]) / sum(totales), "dep": 100 * sum(1 for c in cs if (c.get("clocal") or 0) >= 80) / n}
    return out


def centesimas(valores):
    return np.round(100 * np.asarray(valores, dtype=float)).astype(int)


def lugar(ordenados, c):
    """Parte de las entidades, en porcentaje, con un valor más bajo que «c» (en centésimas)."""
    return 100 * float(np.searchsorted(ordenados, c, side="left")) / len(ordenados)


def leer_casos(entidades=None):
    q = pd.read_parquet(PUNTAJES, columns=K2 + ["grado", "c_local"], filters=[("ruc_entidad", "in", list(entidades))] if entidades else None)
    return q.astype({"ruc_entidad": str, "ruc": str})


def leer_ordenes(casos, entidades=None):
    f = [("anio", "==", ANIO)] + ([("ruc_entidad", "in", list(entidades))] if entidades else [])
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id"] + K2 + ["fecha", "monto", "tope"], filters=f).astype({"ruc_entidad": str, "ruc": str})
    assert set(o["tope"].unique()) == {int(TOPE)}, "el tope de 2025 no es uno solo"
    return o.merge(casos[K2], on=K2, how="inner")


def main():
    faltan = [f"  - falta {r.relative_to(RAIZ)}: se arma con {p}" for r, p in (
        (PUNTAJES, "los puntajes de 2025 (puntajes_2025.py)"), (LISTA, "la lista congelada de 2025 (lista_unica_limpia.py congelar)"),
        (BASE / "ordenes.parquet", "la base de órdenes (capa1_base.py)"), (BASE / "sanciones.parquet", "la base de sanciones (capa1_base.py)"),
        *[(CRUDOS / f, "las descargas del OECE (datos/_scripts/bajar_historia_wiki.py)") for f in PENALIDADES + [MULTAS]]) if not r.exists()]
    if faltan:
        print("ERROR: no están los datos que necesita este paso:\n" + "\n".join(faltan), file=sys.stderr)
        sys.exit(2)
    if huella(LISTA) != HUELLA_LISTA:
        print("ERROR: la lista de 2025 no tiene la huella de la lista congelada (189af867…): no se calculan los índices con otra lista.", file=sys.stderr)
        sys.exit(2)
    casos = leer_casos()
    paso(f"ETAPA casos: {len(casos):,} de {casos['ruc_entidad'].nunique():,} entidades")
    ordenes = leer_ordenes(casos)
    paso(f"ETAPA compras de {ANIO} de esos casos: {len(ordenes):,}")
    ant = con_antecedentes()
    paso(f"ETAPA proveedores con antecedentes al {CORTE:%d/%m/%Y}: {len(ant):,}")
    r = indices(casos, ordenes, ant)
    assert len(r) == casos["ruc_entidad"].nunique() and r[IND].notna().all().all() and r[IND].ge(0).all().all() and r[IND].le(100 + 1e-9).all().all()
    cent = {k: centesimas(r[k]) for k in IND}
    orden = {k: np.sort(cent[k]) for k in IND}
    t = pd.DataFrame({"ruc_entidad": r.index, "casos": r["casos"].to_numpy(), "compras": r["compras"].to_numpy()})
    for k in IND:
        t[k] = cent[k] / 100
        t[f"lugar_{k}"] = [lugar(orden[k], c) for c in cent[k]]
    t = t.sort_values("ruc_entidad").reset_index(drop=True)
    t.to_parquet(TABLA, index=False)
    hp = huella(PUNTAJES)
    dist = {"anio": ANIO, "entidades": len(t), "unidad": "centésimas de punto porcentual", "lista": HUELLA_LISTA[:16], "puntajes": hp[:16],
            "indices": {k: [int(x) for x in orden[k]] for k in IND}}
    # Una sola línea y sin salto al final: así git no le cambia ningún byte y su huella es la misma en cualquier clon
    DISTRIBUCION.write_bytes(json.dumps(dist, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    paso(f"ETAPA índices de {len(t):,} entidades")
    cu = lambda k, p: round(float(np.quantile(orden[k], p)) / 100, 2)  # noqa: E731
    m = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "anio": ANIO, "corte": f"{CORTE:%Y-%m-%d}", "tope": TOPE, "entidades": len(t), "casos": len(casos), "compras": len(ordenes),
         "proveedores_con_antecedentes": len(ant), "entidades_con_20_casos_o_menos": int((t["casos"] <= 20).sum()), "entidades_con_mas_de_100_casos": int((t["casos"] > 100).sum()),
         "resumen": {k: {"p25": cu(k, .25), "mitad": cu(k, .5), "p75": cu(k, .75), "p90": cu(k, .9), "maximo": cu(k, 1), "entidades_en_cero": int((orden[k] == 0).sum())} for k in IND},
         "sha256_indices": huella(TABLA), "sha256_distribucion": huella(DISTRIBUCION), "sha256_puntajes": hp, "sha256_lista_congelada": HUELLA_LISTA,
         "huella_del_script": huella(__file__)[:16], "segundos": round(time.time() - T0, 1)}
    MANIFIESTO.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    paso(f"FIN: {DISTRIBUCION.name}, {DISTRIBUCION.stat().st_size / 1024:.0f} KB, huella {m['sha256_distribucion'][:16]} | la mitad de las entidades tiene, o menos: "
         + ", ".join(f"{k} {m['resumen'][k]['mitad']}" for k in IND))


if __name__ == "__main__":
    main()
