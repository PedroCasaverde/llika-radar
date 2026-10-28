# Convierte los archivos mensuales de órdenes de CONOSCE (ACC-1) a Parquet con tipos correctos, y arma la base de
# órdenes emitidas en 2024 que usan la H2, la H1′ y la H3′.
# Uso: python procesar_conosce.py
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

DATOS = Path(__file__).resolve().parents[1]
ENTRADA = DATOS / "crudos" / "acc1_conosce_ordenes"
SALIDA = DATOS / "procesados" / "acc1_conosce"
FECHAS = ["fecha_registro", "fecha_de_emision", "fecha_compromiso_presupuestal", "fecha_de_notificacion"]


def convertir(xlsx: Path) -> str:
    destino = SALIDA / (xlsx.stem + ".parquet")
    if destino.exists() and destino.stat().st_mtime > xlsx.stat().st_mtime:
        return f"ya estaba  {destino.name}"
    df = pd.read_excel(xlsx, engine="calamine", sheet_name=0, dtype=str)
    for c in FECHAS:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["monto_total_orden_original"] = pd.to_numeric(df["monto_total_orden_original"], errors="coerce")
    df["archivo"] = xlsx.stem.replace("ordenes_", "")  # mes de registro del archivo, AAAA-MM
    df.to_parquet(destino, index=False)
    return f"{len(df):>9,} filas  {destino.name}"


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    archivos = sorted(ENTRADA.glob("ordenes_*.xlsx"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for linea in ex.map(convertir, archivos):
            print(linea, flush=True)

    # Base de órdenes emitidas en 2024, juntando todos los archivos de registro (2024 a 2026)
    todo = pd.concat([pd.read_parquet(p) for p in sorted(SALIDA.glob("ordenes_20??-??.parquet"))], ignore_index=True)
    e24 = todo[todo["fecha_de_emision"].dt.year == 2024].copy()
    clave = ["ruc_entidad", "orden"]
    e24["repetida"] = e24.duplicated(clave, keep=False)
    e24.to_parquet(SALIDA / "emitidas_2024.parquet", index=False)
    print(f"\nórdenes con fecha de emisión en 2024: {len(e24):,} (de {len(todo):,} filas leídas)")
    print(f"filas cuya clave entidad-orden aparece más de una vez: {e24['repetida'].sum():,}")
