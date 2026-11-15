"""Paso 26: señales del OCDS por proveedor-año (HALLAZGOS 76). Solo construye las variables; no evalúa nada.

1. Postores que compiten siempre juntos: pares persistentes, pares que rotan y postor de compañía (ventana t-2 a t).
2. Sancionado que postula o gana en un proceso mientras está inhabilitado.
3. De proceso perdido a orden menor en la misma entidad (180 días).
Cada variable del año t usa solo lo fechado hasta el 31/12 de t.

    python -I -X utf8 26_senales_ocds.py
Salida: datos/procesados/radar/senales_ocds_proveedor_anio.parquet y ocds_pares.parquet. Solo imprime agregados.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import PROC, SALIDA  # noqa: E402

OCDS = PROC / "ocds"
ANIOS = range(2018, 2026)
MIN_JUNTOS, MIN_PARTE, MAX_POSTORES, DIAS = 3, 0.8, 15, 180


def fecha(s):
    return pd.to_datetime(s.str[:10], format="%Y-%m-%d", errors="coerce")


def cargar():
    pr = pd.read_parquet(OCDS / "procesos.parquet", columns=["ocid", "ruc_entidad", "fecha_convocatoria"])
    pr["fecha"] = fecha(pr["fecha_convocatoria"])
    pr = pr.dropna(subset=["fecha"]).drop_duplicates("ocid")
    po = pd.read_parquet(OCDS / "postores.parquet", columns=["ocid", "ruc", "es_consorcio"])
    po = po[~po["es_consorcio"] & po["ruc"].str.fullmatch(r"\d{11}", na=False)].drop_duplicates(["ocid", "ruc"])
    ad = pd.read_parquet(OCDS / "adjudicaciones.parquet", columns=["ocid", "ruc", "fecha", "monto"])
    ad = ad[ad["ruc"].str.fullmatch(r"\d{11}", na=False)].copy()
    ad["fecha"] = fecha(ad["fecha"])
    # «ganó» a la fecha (HALLAZGOS 160, C4b): adjudicado hasta el 31/12 del año de la convocatoria. Se guarda aparte si ganó
    # alguna vez, y cuándo se supo el resultado del proceso: la primera adjudicación, a cualquiera, consorcios incluidos.
    ganadores = ad[["ocid", "ruc", "fecha"]].merge(pr[["ocid", "fecha"]].rename(columns={"fecha": "f_conv"}), on="ocid")
    a_tiempo = ganadores["fecha"].isna() | (ganadores["fecha"].dt.year <= ganadores["f_conv"].dt.year)
    primera = pd.read_parquet(OCDS / "adjudicaciones.parquet", columns=["ocid", "fecha"]).assign(f=lambda x: fecha(x["fecha"]))
    primera = primera.groupby("ocid")["f"].min().rename("f_resultado").reset_index()
    po = (po.merge(pr[["ocid", "ruc_entidad", "fecha"]], on="ocid")
          .merge(ganadores.loc[a_tiempo, ["ocid", "ruc"]].drop_duplicates().assign(gano=True), on=["ocid", "ruc"], how="left")
          .merge(ganadores[["ocid", "ruc"]].drop_duplicates().assign(gano_alguna_vez=True), on=["ocid", "ruc"], how="left")
          .merge(primera, on="ocid", how="left"))
    po["gano"] = po["gano"].fillna(False).astype(bool)
    po["gano_alguna_vez"] = po["gano_alguna_vez"].fillna(False).astype(bool)
    po["anio"] = po["fecha"].dt.year
    return po, ad


def pares_por_anio(po):
    """Un renglón por par (a < b) y año: procesos compartidos y cuántos de ellos ganó cada uno."""
    n = po.groupby("ocid")["ruc"].transform("size")
    chico = po[(n >= 2) & (n <= MAX_POSTORES) & po["anio"].between(min(ANIOS) - 2, max(ANIOS))]
    salida = []
    for anio, g in chico.groupby("anio"):
        g = g[["ocid", "ruc", "gano"]]
        m = g.merge(g, on="ocid", suffixes=("_a", "_b"))
        m = m[m["ruc_a"] < m["ruc_b"]]
        agg = m.groupby(["ruc_a", "ruc_b"]).agg(juntos=("ocid", "size"), gana_a=("gano_a", "sum"),
                                                gana_b=("gano_b", "sum")).reset_index()
        agg["anio"] = anio
        salida.append(agg)
        print(f"  pares {anio}: {len(agg):,}")
    procs = chico.groupby(["ruc", "anio"]).agg(procesos=("ocid", "size"), ganados=("gano", "sum")).reset_index()
    return pd.concat(salida, ignore_index=True), procs


def senal_pares(pares, procs):
    filas, persistentes = [], []
    for t in ANIOS:
        v = pares[pares["anio"].between(t - 2, t)].groupby(["ruc_a", "ruc_b"])[["juntos", "gana_a", "gana_b"]].sum()
        v = v[v["juntos"] >= MIN_JUNTOS].reset_index()
        p = procs[procs["anio"].between(t - 2, t)].groupby("ruc")[["procesos", "ganados"]].sum()
        v["proc_a"] = v["ruc_a"].map(p["procesos"])
        v["proc_b"] = v["ruc_b"].map(p["procesos"])
        v["parte"] = v["juntos"] / v[["proc_a", "proc_b"]].min(axis=1)
        v = v[v["parte"] >= MIN_PARTE].copy()
        v["rota"] = (v["gana_a"] > 0) & (v["gana_b"] > 0)
        v["anio"] = t
        persistentes.append(v)
        # postor de compañía: 5+ procesos, ninguno ganado, 80 % o más con un mismo RUC que gana alguno de los compartidos
        lados = pd.concat([
            v.rename(columns={"ruc_a": "ruc", "gana_b": "gana_otro", "proc_a": "proc"})[["ruc", "juntos", "gana_otro", "proc"]],
            v.rename(columns={"ruc_b": "ruc", "gana_a": "gana_otro", "proc_b": "proc"})[["ruc", "juntos", "gana_otro", "proc"]]])
        lados["ganados"] = lados["ruc"].map(p["ganados"])
        comp = lados[(lados["proc"] >= 5) & (lados["ganados"] == 0) & (lados["juntos"] / lados["proc"] >= MIN_PARTE)
                     & (lados["gana_otro"] > 0)]["ruc"].unique()
        por = pd.concat([v[["ruc_a", "rota"]].rename(columns={"ruc_a": "ruc"}), v[["ruc_b", "rota"]].rename(columns={"ruc_b": "ruc"})])
        f = por.groupby("ruc").agg(n_pares_persistentes=("rota", "size"), n_pares_rotan=("rota", "sum")).reset_index()
        f = f.merge(pd.DataFrame({"ruc": comp, "postor_compania": True}), on="ruc", how="outer")
        f["anio"] = t
        filas.append(f)
    s = pd.concat(filas, ignore_index=True)
    s[["n_pares_persistentes", "n_pares_rotan"]] = s[["n_pares_persistentes", "n_pares_rotan"]].fillna(0).astype(int)
    s["postor_compania"] = s["postor_compania"].fillna(False).astype(bool)
    return s, pd.concat(persistentes, ignore_index=True)


def senal_inhabilitados(po, ad):
    s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio", "fin"]).dropna(subset=["inicio"])
    s["fin"] = s["fin"].fillna(pd.Timestamp("2100-01-01"))

    def dentro(d, col):
        m = d[d["ruc"].isin(s["ruc"])][["ruc", col] + (["monto"] if "monto" in d else [])].reset_index(names="i")
        m = m.merge(s, on="ruc")
        m = m[(m[col] >= m["inicio"]) & (m[col] <= m["fin"])].drop_duplicates("i")
        m["anio"] = m[col].dt.year
        return m

    p = dentro(po, "fecha").groupby(["ruc", "anio"]).size().rename("postula_inhabilitado")
    g = dentro(ad.dropna(subset=["fecha"]), "fecha").groupby(["ruc", "anio"]).agg(
        gana_inhabilitado=("i", "size"), monto_gana_inhabilitado=("monto", "sum"))
    return pd.concat([p, g], axis=1).reset_index()


def senal_perdio_y_orden(po):
    # perdió = el proceso se adjudicó a otro; se sabe desde la primera adjudicación, no desde la convocatoria (160, C4b)
    perd = po[~po["gano_alguna_vez"]][["ruc", "ruc_entidad", "f_resultado"]].dropna().drop_duplicates()
    perd = perd.rename(columns={"f_resultado": "fecha_proceso"})
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["ruc_entidad", "ruc", "fecha", "monto"]).dropna(subset=["fecha"])
    claves = perd[["ruc", "ruc_entidad"]].drop_duplicates()
    o = o.merge(claves, on=["ruc", "ruc_entidad"])
    o = pd.merge_asof(o.sort_values("fecha"), perd.sort_values("fecha_proceso"), left_on="fecha",
                      right_on="fecha_proceso", by=["ruc", "ruc_entidad"], direction="backward",
                      tolerance=pd.Timedelta(days=DIAS))
    o = o.dropna(subset=["fecha_proceso"])
    o["anio"] = o["fecha"].dt.year
    return o.groupby(["ruc", "anio"]).agg(entidades_perdio_y_orden=("ruc_entidad", "nunique"),
                                          ordenes_tras_perder=("fecha", "size"),
                                          monto_tras_perder=("monto", "sum")).reset_index()


def main():
    po, ad = cargar()
    print(f"postulaciones con RUC: {len(po):,} en {po['ocid'].nunique():,} procesos; adjudicaciones: {len(ad):,}")
    pares, procs = pares_por_anio(po)
    s1, persistentes = senal_pares(pares, procs)
    s2 = senal_inhabilitados(po, ad)
    s3 = senal_perdio_y_orden(po)
    s = s1.merge(s2, on=["ruc", "anio"], how="outer").merge(s3, on=["ruc", "anio"], how="outer")
    s = s[s["anio"].isin(ANIOS)]
    num = [c for c in s.columns if c not in ("ruc", "anio", "postor_compania")]
    s[num] = s[num].fillna(0)
    s["postor_compania"] = s["postor_compania"].fillna(False).astype(bool)
    s.to_parquet(SALIDA / "senales_ocds_proveedor_anio.parquet", index=False)
    persistentes.to_parquet(SALIDA / "ocds_pares.parquet", index=False)
    r = s.groupby("anio").agg(con_par=("n_pares_persistentes", lambda x: (x > 0).sum()),
                              con_par_que_rota=("n_pares_rotan", lambda x: (x > 0).sum()),
                              postor_compania=("postor_compania", "sum"),
                              postula_inhabilitado=("postula_inhabilitado", lambda x: (x > 0).sum()),
                              gana_inhabilitado=("gana_inhabilitado", lambda x: (x > 0).sum()),
                              perdio_y_orden=("ordenes_tras_perder", lambda x: (x > 0).sum()))
    print("Proveedores con cada señal, por año:")
    print(r.to_string())
    print(f"Pares persistentes (proveedor-año): {len(persistentes):,}; que rotan: {int(persistentes['rota'].sum()):,}")


if __name__ == "__main__":
    main()
