"""Capa 2, tarjeta territorial a la fecha (HALLAZGOS 128, aprobada por Pedro el 2/11).

- La autoridad misma y el pariente posible salen de la lista del JNE (elecciones de 2014, 2018 y 2022, con DNI y fechas
  de mandato), pública desde la proclamación: son de su época. Se toman de `radar/25` sin cambios.
- La empresa de una autoridad se rehace a la fecha: para las órdenes de cada año, la conformación del OECE del corte más
  reciente hasta el 31/12 de ese año. Para los años que terminan antes del primer corte (11/11/2021) se usa ese primer
  corte y el caso queda marcado (`t_empresa_corte_posterior`), igual que la red de `radar/29`.

Variables por caso (entidad, proveedor y año), en la clase «territorial a la fecha» del diccionario, que `columnas()` no
incluye por defecto (las líneas base no cambian):
    t_autoridad_ordenes, t_empresa_autoridad_ordenes, t_empresa_corte_posterior, t_pariente_ordenes,
    t_pariente_raro_ordenes (par de apellidos con 100 personas o menos en el padrón), tp_territorial_ordenes (todas las
    órdenes territoriales del proveedor en el año, en cualquier entidad).

    python -I -X utf8 capa2_territorial.py
Salida: datos/procesados/v4/x_territorial.parquet (del caso), x_territorial_proveedor.parquet (del proveedor y año) y
filas nuevas en diccionario_x.csv.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from comun import PROC, SALIDA, entidades, norm_serie  # noqa: E402

spec = importlib.util.spec_from_file_location("t25", AQUI.parent / "25_tarjeta_territorial.py")
T25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T25)

V4 = AQUI.parents[1] / "datos" / "procesados" / "v4"
K = ["ruc_entidad", "ruc", "anio"]
ANIOS = range(2018, 2026)
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def empresa_a_la_fecha():
    a = T25.autoridades().rename(columns={"eR": "R", "eP": "P", "eD": "D"})
    cols = ["dni_aut", "CARGO", "nivel", "R", "P", "D", "ini", "fin", "hasta", "todo_el_pais"]
    e = entidades()[["RUC", "DEPARTAMENTO", "PROVINCIA", "DISTRITO"]].rename(columns={"RUC": "ruc_entidad"})
    e["eR"], e["eP"], e["eD"] = (norm_serie(e[c].fillna("")) for c in ("DEPARTAMENTO", "PROVINCIA", "DISTRITO"))
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "monto"])
    o = o[~o["ruc"].str.startswith("10")].merge(e[["ruc_entidad", "eR", "eP", "eD"]], on="ruc_entidad", how="left")
    o["anio"] = o["fecha"].dt.year
    cortes = sorted((PROC / "acc3_conformacion").glob("conformacion_corte*.parquet"))
    fechas = {f: pd.Timestamp(f.stem[-8:]) for f in cortes}
    salidas, cache = [], {}
    for y in ANIOS:
        fin = pd.Timestamp(f"{y}-12-31")
        validos = [f for f in cortes if fechas[f] <= fin]
        f = validos[-1] if validos else cortes[0]
        if f not in cache:
            c = pd.read_parquet(f, columns=["ruc_proveedor", "dni_integrante", "usable_para_cruce"])
            c = c[c["usable_para_cruce"].fillna(False).astype(bool) & c["dni_integrante"].str.fullmatch(r"\d{8}", na=False)]
            cache = {f: c[["ruc_proveedor", "dni_integrante"]].drop_duplicates().rename(columns={"ruc_proveedor": "ruc"})}
        emp = o[o["anio"] == y].merge(cache[f], on="ruc")
        m = T25.ventana(emp.merge(a[cols], left_on="dni_integrante", right_on="dni_aut"))
        m = m.assign(tipo="empresa_de_autoridad", corte=fechas[f], corte_posterior=fechas[f] > fin)
        salidas.append(m[["id", "ruc_entidad", "ruc", "fecha", "anio", "tipo", "corte_posterior"]].drop_duplicates("id"))
        paso(f"   {y}: corte {fechas[f].date()}{' (posterior)' if fechas[f] > fin else ''}; órdenes de empresa de autoridad {len(salidas[-1]):,}")
    return pd.concat(salidas, ignore_index=True)


def main():
    t = pd.read_parquet(SALIDA / "tarjeta_territorial.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "tipo", "freq_par"])
    t = t[t["tipo"].isin(["autoridad_misma", "pariente_posible"])].assign(anio=lambda x: x["fecha"].dt.year, corte_posterior=False)
    paso(f"autoridad misma y pariente posible (lista del JNE, de su época): {t.groupby('tipo').size().to_dict()}")
    emp = empresa_a_la_fecha()
    vieja = pd.read_parquet(SALIDA / "tarjeta_territorial.parquet", columns=["id", "tipo"])
    vieja = set(vieja.loc[vieja["tipo"] == "empresa_de_autoridad", "id"])
    paso(f"empresa de autoridad: {len(emp):,} órdenes a la fecha frente a {len(vieja):,} con la foto de 10/2026; "
         f"en ambas {len(vieja & set(emp['id'])):,}")
    todo = pd.concat([t, emp], ignore_index=True).astype({"ruc_entidad": str, "ruc": str, "anio": int})
    todo = todo[todo["anio"].isin(ANIOS)]
    g = todo.groupby(K)
    x = pd.DataFrame({
        "t_autoridad_ordenes": g["tipo"].apply(lambda s: (s == "autoridad_misma").sum()),
        "t_empresa_autoridad_ordenes": g["tipo"].apply(lambda s: (s == "empresa_de_autoridad").sum()),
        "t_empresa_corte_posterior": g["corte_posterior"].max().astype(float),
        "t_pariente_ordenes": g["tipo"].apply(lambda s: (s == "pariente_posible").sum()),
        "t_pariente_raro_ordenes": todo.assign(r=(todo["tipo"] == "pariente_posible") & (todo["freq_par"] <= 100)).groupby(K)["r"].sum(),
    }).reset_index()
    tp = todo.groupby(["ruc", "anio"]).size().rename("tp_territorial_ordenes").reset_index()
    x.to_parquet(V4 / "x_territorial.parquet", index=False)
    tp.to_parquet(V4 / "x_territorial_proveedor.parquet", index=False)      # del proveedor: se une por RUC y año
    x = x.merge(tp, on=["ruc", "anio"], how="left")
    paso(f"casos con alguna marca territorial por año: {x.groupby('anio').size().to_dict()}")
    # diccionario: clase propia, fuera de las líneas base
    d = pd.read_csv(V4 / "diccionario_x.csv")
    nuevas = [c for c in x.columns if c not in K]
    d = d[~d["variable"].isin(nuevas)]
    fuente = {"t_empresa_autoridad_ordenes": "JNE y conformación del OECE del corte más reciente al 31/12 (antes de 11/2021, el primero)",
              "t_empresa_corte_posterior": "conformación del OECE: el corte es posterior al año (2018 a 2020)"}
    filas = [{"variable": c, "grupo": "alertas", "tipo": "binaria" if c == "t_empresa_corte_posterior" else "conteo o monto",
              "clase_temporal": "territorial a la fecha", "fuente": fuente.get(c, "lista de autoridades electas del JNE (2014, 2018, 2022) y padrón de SUNAT"),
              "lleno_pct": 100.0} for c in nuevas]
    pd.concat([d, pd.DataFrame(filas)], ignore_index=True).to_csv(V4 / "diccionario_x.csv", index=False)
    paso("listo")


if __name__ == "__main__":
    main()
