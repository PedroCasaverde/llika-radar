"""Lector único del registro de penalidades del OECE (HALLAZGOS 160, C1). Lo usan el paso 14 y el paso 29.

Corrige lo que encontró la auditoría en el lector anterior (pandas por defecto):
- separador «|» sin comillas: una comilla suelta en la descripción ya no se traga cientos de filas;
- una fila partida en dos líneas (salto dentro de la descripción) se vuelve a unir;
- el encabezado partido de la versión del 2021-10-02 se reconoce;
- la fecha con año de 2 o de 4 dígitos, y con hora (antes se perdían versiones enteras);
- el monto con coma decimal («31340,64» es 31 340,64, no 3 134 064) y la versión del 2024-02-10, separada por comas,
  con el monto decimal partido en dos campos;
- una penalidad es una llave (RUC, contrato, fecha), con el monto de su última versión.
Los códigos de consorcio (C… o L…, sin RUC) se conservan marcados con es_consorcio.
"""
import csv
import glob
import io
import re
from pathlib import Path

import pandas as pd

COLS = ["contrato", "ruc", "tipo_penalidad", "objeto", "entidad", "fecha", "descripcion", "monto"]
CLAVE = ["ruc", "contrato", "fecha"]


def _fecha(s):
    s = s.astype(str).str.strip().str.split(" ").str[0]    # la versión del 2024-04-06 trae «dd/mm/aaaa 00:00:00»
    f4 = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return f4.fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))


def _filas_tubo(txt):
    lineas = txt.split("\n")
    i, cab = 0, ""
    while i < len(lineas) and cab.count("|") < 7:          # encabezado, aunque venga partido en varias líneas
        cab += lineas[i]
        i += 1
    filas, buf = [], ""
    for l in lineas[i:]:
        buf = l if not buf else buf + " " + l
        p = buf.split("|")
        if len(p) < 8:                                      # salto de línea dentro de la descripción
            continue
        if len(p) > 8:                                      # una «|» dentro de la descripción
            p = p[:6] + ["|".join(p[6:-1])] + [p[-1]]
        filas.append(p)
        buf = ""
    d = pd.DataFrame(filas, columns=COLS)
    d["monto"] = pd.to_numeric(d["monto"].str.strip().str.strip('"').str.replace(",", ".", regex=False), errors="coerce")
    return d


def _filas_coma(txt):
    filas = []
    for p in csv.reader(io.StringIO(txt)):
        if not p or p[0].startswith("ID CONTRATO"):
            continue
        if len(p) == 9:                                     # monto decimal sin comillas: «3730,4» → 3730.4
            p = p[:7] + [p[7] + "." + p[8]]
        if len(p) == 8:
            filas.append(p)
    d = pd.DataFrame(filas, columns=COLS)
    d["monto"] = pd.to_numeric(d["monto"].str.strip(), errors="coerce")
    return d


def versiones(carpeta):
    """Todas las filas de todas las versiones, con la fecha de la versión y su serie."""
    out = []
    for f in sorted(glob.glob(str(Path(carpeta) / "*.csv"))):
        b = open(f, "rb").read()
        if not b.strip() or b[:2] == b"PK":                 # vacío, o un Excel guardado como csv
            continue
        txt = b.decode("cp1252", "replace").replace("\r", "")
        if not txt.lstrip('"').startswith("ID CONTRATO"):
            continue
        d = _filas_tubo(txt) if "|" in txt.split("\n", 1)[0] else _filas_coma(txt)
        for c in ("contrato", "ruc", "entidad", "tipo_penalidad", "objeto"):
            d[c] = d[c].str.strip().str.strip('"').str.strip()
        d["entidad"] = d["entidad"].str.upper()
        d["fecha"] = _fecha(d["fecha"].str.strip('"'))
        d["version"] = pd.Timestamp(re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1))
        d["serie"] = "2018" if Path(f).name.startswith("penalidades2018") else "actual"
        out.append(d.drop(columns=["descripcion"]))
    v = pd.concat(out, ignore_index=True).dropna(subset=["fecha"])
    v = v[v["ruc"].str.fullmatch(r"\d{11}|[CL]\w{6,}", na=False)]
    v["es_consorcio"] = ~v["ruc"].str.fullmatch(r"\d{11}")
    return v


def fechar(p, sin_versiones="2021-11-01", primeras_versiones="2021-12-31"):
    """Fecha en que se supo cada penalidad (HALLAZGOS 160, C2).
    - Desde 11/2021 hay versiones del registro: es la primera versión en que aparece.
    - Antes, se imputa la fecha más la demora mediana de su entidad (medida desde 11/2021; con menos de 20 casos, la
      general). Si la penalidad apareció recién después de las primeras versiones (12/2021), no era pública antes: se
      toma la mayor entre la imputada y esa primera versión.
    Devuelve p con fecha_conocido, conocido_mediana, conocido_p90 y conocido_imputado, y un resumen."""
    p = p.copy()
    antes = p["fecha"] < pd.Timestamp(sin_versiones)
    dem = (p.loc[~antes, "primera_version"] - p.loc[~antes, "fecha"]).dt.days.clip(lower=0)
    med_g, p90_g = float(dem.median()), float(dem.quantile(0.9))
    por_ent = dem.groupby(p.loc[~antes, "entidad"]).agg(["median", "size"])
    med_ent = por_ent.loc[por_ent["size"] >= 20, "median"]
    dias = p["entidad"].map(med_ent).fillna(med_g)
    tardia = antes & (p["primera_version"] > pd.Timestamp(primeras_versiones))
    pv = p["primera_version"]
    for col, imp in (("fecha_conocido", p["fecha"] + pd.to_timedelta(dias, unit="D")),
                     ("conocido_mediana", p["fecha"] + pd.Timedelta(days=med_g)),
                     ("conocido_p90", p["fecha"] + pd.Timedelta(days=p90_g))):
        p[col] = pv.where(~antes, imp.where(~tardia, pd.concat([imp, pv], axis=1).max(axis=1)))
    p["conocido_imputado"] = antes & ~(tardia & (pv >= p["fecha_conocido"]))
    r = {"mediana_dias": med_g, "p90_dias": p90_g, "entidades_con_mediana": len(med_ent),
         "anteriores": int(antes.sum()), "tardias": int(tardia.sum())}
    return p, r


def penalidades(carpeta):
    """Una fila por penalidad (RUC, contrato, fecha): primera versión en que apareció, monto y entidad de la última."""
    v = versiones(carpeta).sort_values("version")
    g = v.groupby(CLAVE, sort=False)
    p = g.last().reset_index()
    p["primera_version"] = g["version"].min().to_numpy()
    p["versiones"] = g.size().to_numpy()
    return p.drop(columns=["version"])
