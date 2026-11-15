"""Tabla de órdenes de los catálogos electrónicos de Perú Compras, desde los OCDS que baja perucompras_ocds.py.

  python -X utf8 perucompras_ordenes.py
Una fila por orden (la última versión de cada ocid): número, entidad, proveedor, fecha de la orden, monto, acuerdo
marco y estado. Salida: datos/procesados/perucompras/ordenes_catalogo.parquet.
"""
import gzip
import json
import re
from pathlib import Path

import pandas as pd

DATOS = Path(__file__).resolve().parents[1]
ORIGEN = DATOS / "crudos" / "perucompras_ocds"
DEST = DATOS / "procesados" / "perucompras"


def ruc(x):
    m = re.search(r"(\d{11})", str(x or ""))
    return m.group(1) if m else None


def main():
    filas, malos = [], []
    for f in sorted(ORIGEN.glob("*/acuerdo_*.json.gz")):
        acuerdo = f.stem.split("_")[1]
        try:   # algunos archivos traen caracteres de control dentro de los textos: strict=False los acepta
            releases = json.loads(gzip.decompress(f.read_bytes()), strict=False)["releases"]
        except json.JSONDecodeError:
            malos.append(f.name)
            continue
        for r in releases:
            aw = (r.get("awards") or [{}])[0]
            co = (r.get("contracts") or [{}])[0]
            sup = (aw.get("suppliers") or [{}])[0]
            filas.append({"ocid": r.get("ocid"), "version": r.get("date"), "orden": (r.get("tender") or {}).get("id"),
                          "ruc_entidad": ruc((r.get("buyer") or {}).get("id")), "ruc": ruc(sup.get("id")),
                          "proveedor": sup.get("name"), "fecha": (co.get("dateSigned") or "")[:10] or None,
                          "monto": (aw.get("value") or {}).get("amount"), "estado": co.get("statusDetails"),
                          "acuerdo": acuerdo})
    if malos:
        print(f"archivos que no se pudieron leer: {len(malos)}: {malos[:5]}")
    d = pd.DataFrame(filas).sort_values("version").drop_duplicates("ocid", keep="last")
    d["fecha"] = pd.to_datetime(d["fecha"], format="%Y-%m-%d", errors="coerce")
    d["anio"] = d["fecha"].dt.year
    DEST.mkdir(parents=True, exist_ok=True)
    d.drop(columns="version").to_parquet(DEST / "ordenes_catalogo.parquet", index=False)
    print(f"{len(d):,} órdenes de catálogo, {d['ruc'].nunique():,} proveedores, {d['ruc_entidad'].nunique():,} entidades,"
          f" S/ {d['monto'].sum() / 1e9:,.2f} mil millones")
    print(d.groupby("anio").agg(ordenes=("ocid", "size"), monto_millones=("monto", lambda x: round(x.sum() / 1e6))).to_string())


if __name__ == "__main__":
    main()
