"""Capa 2, posible familiar de las declaraciones juradas a la fecha de decisión (HALLAZGOS 132, aprobada por Pedro el 2/11).

La alerta de `radar/02` (mismos dos apellidos que un funcionario con cargo de decisión de la entidad que compra, mientras
tiene el cargo, con probabilidad bayesiana de parentesco) usa la foto de las declaraciones de 07/2026. Aquí se le pone a
cada alerta la fecha en que se hizo visible: la primera presentación de una declaración de alguno de los funcionarios
que la generan. Las declaraciones se presentan desde 2022.

A la fecha de decisión T (el cierre del año de prueba), una alerta cuenta si esa fecha es anterior al 31/12/T, para
los casos de entrenamiento y de prueba por igual: es lo que el auditor tendría a mano ese día sobre hechos pasados. Por
eso en los cortes 2020 y 2021 no hay ninguna y en 2023 y en el sellado 2022 sí. La probabilidad se queda con la
calibración de la foto completa (una constante por año); se documenta.

Guarda, por caso, para cada año Y de 2022 a 2025: f_fam_prob_hasta_Y (la mayor probabilidad entre las alertas visibles
al 31/12/Y) y f_fam_ordenes_hasta_Y (cuántas órdenes). `evaluacion.armar_corte` elige el Y del corte y arma
f_familiar_prob y f_familiar_ordenes (clase «familia a la fecha de decisión» del diccionario, fuera de las líneas base).

    python -I -X utf8 capa2_familia.py
Salida: datos/procesados/v4/x_familia.parquet y filas nuevas en diccionario_x.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from comun import SALIDA, entidades, fecha_dji, funcionarios, norm_serie, ruc_por_nombre, ultimo  # noqa: E402

V4 = AQUI.parents[1] / "datos" / "procesados" / "v4"
K = ["ruc_entidad", "ruc", "anio"]
ANIOS_DECISION = range(2022, 2026)
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def presentaciones():
    """Primera presentación de declaración por persona y entidad, con las mismas llaves que comun.funcionarios()."""
    r = pd.read_csv(ultimo("acc4_dji_declaraciones"), dtype=str, encoding="utf-8",
                    usecols=["APELLIDO_PATERNO", "APELLIDO_MATERNO", "NOMBRES", "ENTIDAD", "FECHA_PRESENTACION_DDJJ"])
    r["presentacion"] = fecha_dji(r["FECHA_PRESENTACION_DDJJ"])
    tabla = SALIDA / "entidad_declarante.csv"
    if tabla.exists():
        mapa = pd.read_csv(tabla, dtype={"ruc": str}, keep_default_na=False, na_values=[""])
        r["ruc_entidad"] = r["ENTIDAD"].fillna("").map(mapa.dropna(subset=["ruc"]).set_index("nombre")["ruc"])
    else:
        r["ruc_entidad"] = ruc_por_nombre(r["ENTIDAD"], entidades())
    r["persona"] = (norm_serie(r["APELLIDO_PATERNO"].fillna("")) + " " + norm_serie(r["APELLIDO_MATERNO"].fillna("")) + " "
                    + norm_serie(r["NOMBRES"].fillna(""))).str.strip()
    return r.dropna(subset=["ruc_entidad", "presentacion"]).groupby(["persona", "ruc_entidad"])["presentacion"].min().reset_index()


def main():
    a = pd.read_parquet(SALIDA / "alerta_familiar.parquet", columns=["id", "prob"])
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "nombre", "fecha", "anio"])
    o = o[o["id"].isin(set(a["id"]))].copy()
    o["nombre"] = norm_serie(o["nombre"])
    paso(f"alertas de posible familiar: {len(a):,} órdenes")
    f = funcionarios()
    f = f[f["decide"]].merge(presentaciones(), on=["persona", "ruc_entidad"], how="left")
    paso(f"funcionarios con cargo de decisión: {len(f):,}; con fecha de presentación: {f['presentacion'].notna().mean():.1%}")
    t = o["nombre"].str.split()
    pref = pd.concat([pd.DataFrame({"id": o.loc[t.str.len() > k, "id"], "par": t[t.str.len() > k].str[:k].str.join(" ")})
                      for k in (2, 3, 4)])
    m = pref.merge(o[["id", "ruc_entidad", "fecha"]], on="id").merge(f[["ruc_entidad", "par", "inicio", "cese", "presentacion"]],
                                                                    on=["ruc_entidad", "par"])
    m = m[(m["fecha"] >= m["inicio"]) & (m["fecha"] <= m["cese"])]
    vis = m.groupby("id")["presentacion"].min()
    a = a.merge(o[["id"] + K], on="id").assign(visible=lambda x: x["id"].map(vis))
    paso(f"alertas con fecha de visibilidad: {a['visible'].notna().mean():.1%}; por año en que se hacen visibles: "
         f"{a['visible'].dt.year.value_counts().sort_index().to_dict()}")
    a = a.astype({"ruc_entidad": str, "ruc": str, "anio": int})
    x = a[K].drop_duplicates().set_index(K)
    for y in ANIOS_DECISION:
        v = a[a["visible"] <= pd.Timestamp(f"{y}-12-31")].groupby(K)
        x[f"f_fam_prob_hasta_{y}"] = v["prob"].max()
        x[f"f_fam_ordenes_hasta_{y}"] = v["id"].size()
    x = x.fillna(0).reset_index()
    x.to_parquet(V4 / "x_familia.parquet", index=False)
    paso("casos con alguna alerta visible, por año de decisión: "
         + "; ".join(f"{y}: {int((x[f'f_fam_ordenes_hasta_{y}'] > 0).sum()):,}" for y in ANIOS_DECISION)
         + f" (de {len(x):,} casos con alerta en la foto de 2026)")
    d = pd.read_csv(V4 / "diccionario_x.csv")
    nuevas = ["f_familiar_prob", "f_familiar_ordenes"]
    d = d[~d["variable"].isin(nuevas)]
    fuente = "declaraciones juradas presentadas hasta el 31/12 del año de decisión (radar/02, probabilidad bayesiana)"
    filas = [{"variable": c, "grupo": "alertas", "tipo": "proporción" if c.endswith("prob") else "conteo o monto",
              "clase_temporal": "familia a la fecha de decisión", "fuente": fuente, "lleno_pct": 100.0} for c in nuevas]
    pd.concat([d, pd.DataFrame(filas)], ignore_index=True).to_csv(V4 / "diccionario_x.csv", index=False)
    paso("listo")


if __name__ == "__main__":
    main()
