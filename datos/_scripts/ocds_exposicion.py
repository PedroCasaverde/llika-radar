"""Exposición de cada proveedor en el OCDS, por año (PENDIENTES, sección 8, paso 4: la exposición como offset o estrato).
Un proveedor que postula a muchos procesos tiene más ocasiones de ser sancionado; sin medirlo, el modelo confunde
actividad con riesgo.

  python -X utf8 ocds_exposicion.py

Por RUC: procesos a los que postuló, entidades distintas y si postuló en consorcio, por año de convocatoria; procesos que
ganó y monto adjudicado en soles, por año de adjudicación (HALLAZGOS 160, C4). 2018 queda censurado a la izquierda: el
OCDS casi no trae convocatorias de 2017 adjudicadas en 2018. Los consorcios no traen RUC en el OCDS (HALLAZGOS 68): sus integrantes no
suman aquí. Salida: datos/procesados/ocds/exposicion_proveedor_anio.parquet.
"""
from pathlib import Path

import pandas as pd

OCDS = Path(__file__).resolve().parents[1] / "procesados" / "ocds"


def main():
    pr = pd.read_parquet(OCDS / "procesos.parquet", columns=["ocid", "ruc_entidad", "fecha_convocatoria"])
    pr["anio"] = pd.to_datetime(pr["fecha_convocatoria"].str[:10], errors="coerce").dt.year
    pr = pr.dropna(subset=["anio"]).drop_duplicates("ocid")
    po = pd.read_parquet(OCDS / "postores.parquet", columns=["ocid", "ruc", "es_consorcio"])
    po = po.dropna(subset=["ruc"]).drop_duplicates(["ocid", "ruc"]).merge(pr, on="ocid")
    e = po.groupby(["ruc", "anio"]).agg(procesos=("ocid", "size"), entidades=("ruc_entidad", "nunique"),
                                        en_consorcio=("es_consorcio", "max"))
    # lo ganado va al año en que se adjudicó, no al de la convocatoria (HALLAZGOS 160, C4): el 7,3 % cae en otro año.
    # Si falta la fecha, la primera firma del contrato, y si tampoco hay, la convocatoria.
    ad = pd.read_parquet(OCDS / "adjudicaciones.parquet", columns=["ocid", "adjudicacion_id", "ruc", "fecha", "monto", "moneda"])
    ad = ad.dropna(subset=["ruc"]).merge(pr[["ocid", "anio"]].rename(columns={"anio": "anio_conv"}), on="ocid")
    co = pd.read_parquet(OCDS / "contratos.parquet", columns=["ocid", "adjudicacion_id", "fecha_firma"])
    co["firma"] = pd.to_datetime(co["fecha_firma"].str[:10], errors="coerce")
    ad = ad.merge(co.groupby(["ocid", "adjudicacion_id"])["firma"].min().reset_index(), on=["ocid", "adjudicacion_id"], how="left")
    f_adj = pd.to_datetime(ad["fecha"].str[:10], errors="coerce").fillna(ad["firma"])
    ad["anio"] = f_adj.dt.year.fillna(ad["anio_conv"])
    print(f"adjudicaciones con RUC: {len(ad):,}; en otro año que su convocatoria: {(ad['anio'] != ad['anio_conv']).mean():.1%}; "
          f"sin fecha ni firma: {int(f_adj.isna().sum()):,}")
    ad["monto_pen"] = ad["monto"].where(ad["moneda"].fillna("PEN").eq("PEN"))
    g = ad.groupby(["ruc", "anio"]).agg(ganados=("ocid", "nunique"), monto_adjudicado=("monto_pen", "sum"))
    e = e.join(g, how="outer").fillna({"procesos": 0, "entidades": 0, "ganados": 0, "monto_adjudicado": 0})
    e = e.reset_index().astype({"anio": "int64", "procesos": "int64", "entidades": "int64", "ganados": "int64"})
    e.to_parquet(OCDS / "exposicion_proveedor_anio.parquet", index=False)
    print(f"{len(e):,} pares proveedor-año, {e['ruc'].nunique():,} proveedores, años {e['anio'].min()}-{e['anio'].max()}")
    print(e.groupby("anio").agg(proveedores=("ruc", "nunique"), postulaciones=("procesos", "sum"),
                                ganados=("ganados", "sum")).to_string())


if __name__ == "__main__":
    main()
