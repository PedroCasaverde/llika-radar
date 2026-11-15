"""Ubicación (ubigeo) y dirección fiscal de los proveedores y entidades de LLIKA, desde el padrón reducido de SUNAT
(HALLAZGOS 110). Solo se guardan los RUC que aparecen en las órdenes (proveedores y entidades).

    python -X utf8 padron_ubicacion.py
Salida: datos/procesados/radar/padron_ubicacion.parquet (ruc, ubigeo, direccion). Es una foto de 2026: la ubicación cambia
poco, pero no es la del año de cada orden (se marca como dato posterior en el diccionario).
"""
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

DATOS = Path(__file__).resolve().parents[1]
ZIP = sorted((DATOS / "crudos" / "acc6_padron_sunat").glob("padron_reducido_ruc_*.zip"))[-1]
OUT = DATOS / "procesados" / "radar" / "padron_ubicacion.parquet"
COLS = ["RUC", "UBIGEO", "TIPO DE VÍA", "NOMBRE DE VÍA", "CÓDIGO DE ZONA", "TIPO DE ZONA", "NÚMERO", "INTERIOR", "LOTE",
        "DEPARTAMENTO", "MANZANA", "KILÓMETRO"]


def limpiar(s):
    s = s.fillna("").astype(str).str.upper().str.strip()
    return s.where(~s.isin(["-", "--", "S/N", "SN", "0", ""]), "")


def main():
    o = pd.read_parquet(DATOS / "procesados" / "radar" / "ordenes.parquet", columns=["ruc", "ruc_entidad"])
    rucs = set(o["ruc"].astype(str)) | set(o["ruc_entidad"].astype(str))
    partes = []
    with zipfile.ZipFile(ZIP) as z, z.open(z.namelist()[0]) as f:
        texto = io.TextIOWrapper(f, encoding="latin1", errors="replace")
        for d in pd.read_csv(texto, sep="|", dtype=str, usecols=lambda c: c in COLS, chunksize=1_000_000,
                             on_bad_lines="skip", quoting=3):
            d = d[d["RUC"].isin(rucs)]
            if len(d):
                partes.append(d)
    p = pd.concat(partes, ignore_index=True).drop_duplicates("RUC")
    via = limpiar(p["TIPO DE VÍA"]) + " " + limpiar(p["NOMBRE DE VÍA"])

    dire = (via + " N" + limpiar(p["NÚMERO"]) + " INT" + limpiar(p["INTERIOR"]) + " MZ" + limpiar(p["MANZANA"]) + " LT" + limpiar(p["LOTE"])
            + " KM" + limpiar(p["KILÓMETRO"]) + " " + limpiar(p["TIPO DE ZONA"]) + " " + limpiar(p["CÓDIGO DE ZONA"]))
    dire = dire.str.replace(r"\s+", " ", regex=True).str.strip()
    sin = limpiar(p["NOMBRE DE VÍA"]).eq("") & limpiar(p["MANZANA"]).eq("")
    ubigeo = limpiar(p["UBIGEO"])
    out = pd.DataFrame({"ruc": p["RUC"], "ubigeo": ubigeo.where(ubigeo.str.fullmatch(r"\d{6}"), None),
                        "direccion": dire.where(~sin, None)})
    out.to_parquet(OUT, index=False)
    print(f"RUC buscados: {len(rucs):,}; hallados en el padrón: {len(out):,}; con ubigeo: {out['ubigeo'].notna().mean():.1%}; "
          f"con dirección: {out['direccion'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
