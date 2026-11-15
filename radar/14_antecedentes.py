"""Paso 14: antecedentes de incumplimiento de cada proveedor, para usarlos como señal de contexto del radar.

- Multas del Tribunal: unión de las versiones mensuales de la wiki de datos abiertos del OECE (desde 09/2021). El registro
  solo lista las multas vigentes en cada corte, así que las que terminaron antes de 09/2021 no aparecen.
- Penalidades que las entidades aplicaron en sus contratos: unión de las versiones de los dos archivos del OECE
  («penalidades» y «penalidades2018»).
Deja datos/procesados/radar/antecedentes.parquet con una fila por antecedente (ruc, fecha, tipo, monto). Solo imprime
agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 14_antecedentes.py
"""
import glob
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import CRUDOS, SALIDA  # noqa: E402
import lector_penalidades as LP  # noqa: E402

NUEVA_MULTA = ["FECHA_CORTE", "RUC", "NOMBRE", "FECHA_INICIO", "FECHA_FIN", "NUMERO_RESOLUCION", "ID_MOTIVO", "MOTIVO",
               "MONTO"]


def leer_texto(f):
    with open(f, "rb") as h:
        crudo = h.read()
    if not crudo.strip() or crudo[:2] == b"PK":  # vacío, o un Excel guardado con extensión csv
        return None
    return crudo.decode("cp1252", "replace")


def multas():
    filas = []
    for f in sorted(glob.glob(str(CRUDOS / "oece_multas" / "_historia" / "*.csv"))):
        txt = leer_texto(f)
        if txt is None:
            continue
        primera = txt.split("\n", 1)[0]
        if primera.startswith("RUC|Nombre"):  # esquema de 2021: Desde y Hasta
            d = pd.read_csv(io.StringIO(txt), sep="|", dtype=str, index_col=False)
            d = pd.DataFrame({"ruc": d["RUC"], "fecha": pd.to_datetime(d["Desde"], dayfirst=True, errors="coerce"),
                              "resolucion": d["Resolucion"], "monto": d["Monto"]})
        else:
            con_cabecera = primera.startswith("FECHA_CORTE")
            d = pd.read_csv(io.StringIO(txt), sep="|", dtype=str, header=0 if con_cabecera else None, index_col=False,
                            usecols=range(9))
            d.columns = NUEVA_MULTA
            d = pd.DataFrame({"ruc": d["RUC"], "fecha": pd.to_datetime(d["FECHA_INICIO"], format="%Y%m%d", errors="coerce"),
                              "resolucion": d["NUMERO_RESOLUCION"], "monto": d["MONTO"]})
        filas.append(d)
    m = pd.concat(filas).dropna(subset=["ruc", "fecha"]).drop_duplicates(["ruc", "resolucion", "fecha"])
    m["tipo"] = "multa"
    return m


def penalidades():
    """Lector único del registro (HALLAZGOS 160, C1): una fila por (RUC, contrato, fecha), con el monto bien leído."""
    p = LP.penalidades(CRUDOS / "oece_penalidades" / "_historia")
    p = p[~p["es_consorcio"]]
    return pd.DataFrame({"ruc": p["ruc"], "fecha": p["fecha"], "tipo": "penalidad", "monto": p["monto"]})


m, p = multas(), penalidades()
a = pd.concat([m[["ruc", "fecha", "tipo", "monto"]], p[["ruc", "fecha", "tipo", "monto"]]], ignore_index=True)
a["ruc"] = a["ruc"].str.strip()
a = a[a["ruc"].str.fullmatch(r"\d{11}", na=False)]
es_multa = a["tipo"] == "multa"                   # las penalidades ya traen el monto leído por el lector único
a["monto"] = pd.to_numeric(a["monto"].where(~es_multa, a["monto"].astype(str).str.replace(",", "", regex=False)),
                           errors="coerce")
a.to_parquet(SALIDA / "antecedentes.parquet", index=False)
for t, g in a.groupby("tipo"):
    print(f"{t}: {len(g):,} registros de {g['ruc'].nunique():,} RUC; por año: "
          f"{g['fecha'].dt.year.value_counts().sort_index().to_dict()}")
