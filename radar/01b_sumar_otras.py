"""Paso 1b: suma al universo las «Otras contrataciones sin proceso de selección previo» (HALLAZGOS 160, C3).

Tienen el perfil de una compra menor y algunas entidades registran así casi todo (160, sección 11). La parte vieja queda
CONGELADA: los id no cambian, porque 15 tablas se cruzan por ellos. Las órdenes nuevas van al final, con id desde el máximo
más uno. Regla de cada orden nueva:
- de ese tipo de contratación, en soles, no anulada, monto mayor que 0 y hasta el tope por fecha (8 UIT; 9 UIT del 19/07 al
  31/12/2022), emitida de 2018 a 2025;
- una por (entidad, orden), la última por fecha de registro; si la llave ya está en la parte vieja, se queda la vieja.
Se marca, no se borra:
- otra_categoria; fecha_registro (la de la versión guardada) y fecha_registro_primera (la primera versión válida);
- anulada_despues y anulada_mismo_dia_o_despues: la llave tiene una versión anulada posterior o del mismo día;
- ultima_version_otras: orden vieja cuya última versión es «otras».

La primera vez copia radar/ordenes.parquet a radar/ordenes_base.parquet, y desde ahí siempre parte de esa copia.
Después de este paso corre 01_base.py (sin --rehacer), para las sanciones y la alerta de sancionado.

    python -I -X utf8 01b_sumar_otras.py
"""
import glob
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from comun import PROC, SALIDA, UIT  # noqa: E402

T0 = time.time()
ANIOS = sorted(UIT)
TIPO = "Otras contrataciones sin proceso de selección previo"
FINAL = SALIDA / "ordenes.parquet"
VIEJO = SALIDA / "ordenes_base.parquet"
TEMP = SALIDA / "ordenes_lote2_tmp.parquet"
COLS_V = ["ruc_entidad", "depto", "orden", "ruc", "nombre", "fecha", "monto", "servicio", "anio", "id"]


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def tope_expr(anio, fecha):
    uit = anio.replace_strict({a: UIT[a] for a in ANIOS}, default=None, return_dtype=pl.Float64)
    return uit * pl.when((anio == 2022) & (fecha >= pl.datetime(2022, 7, 19))).then(9).otherwise(8)


def main():
    if not VIEJO.exists():
        if "otra_categoria" in pl.scan_parquet(FINAL).collect_schema():
            raise SystemExit("ordenes.parquet ya trae las «otras» y no existe ordenes_base.parquet")
        shutil.copy2(FINAL, VIEJO)
        paso("parte vieja congelada en ordenes_base.parquet")
    fs = sorted(glob.glob(str(PROC / "acc1_conosce" / "ordenes_20??-??.parquet")))
    t, m, fe = pl.col("tipodecontratacion"), pl.col("monto_total_orden_original"), pl.col("fecha_de_emision")
    anio = fe.dt.year()
    uit = anio.replace_strict({a: UIT[a] for a in ANIOS}, default=None, return_dtype=pl.Float64)
    llave = pl.struct(["ruc_entidad", "orden"]).hash(seed=11)
    base = (pl.scan_parquet(fs).with_row_index("fila").filter(anio.is_between(min(ANIOS), max(ANIOS)))
            .with_columns(valida=(pl.col("moneda") == "Nuevos Soles") & (pl.col("estadocontratacion") != "Anulada") & (m > 0),
                          tope_anio=uit * pl.when(anio == 2022).then(9).otherwise(8), tope_fecha=tope_expr(anio, fe))
            .with_columns(ok_U=t.str.to_uppercase().str.starts_with("CONTRATACIONES HASTA") & pl.col("valida") & (m <= pl.col("tope_anio")),
                          ok_O=(t == TIPO) & pl.col("valida") & (m <= pl.col("tope_fecha")),
                          anulada=pl.col("estadocontratacion") == "Anulada", kh=llave))
    # órdenes con versiones válidas en las dos categorías: casi todas empatan en fecha de registro, así que la marca
    # ultima_version_otras (estrictamente posterior) es muy corta. Esta tabla sirve para la sensibilidad (160, sección 13).
    dos = (base.select(["kh", "fecha_registro", "ok_U", "ok_O"]).group_by("kh").agg(
        f_reg_U=pl.col("fecha_registro").filter(pl.col("ok_U")).max(), f_reg_O=pl.col("fecha_registro").filter(pl.col("ok_O")).max())
        .filter(pl.col("f_reg_U").is_not_null() & pl.col("f_reg_O").is_not_null()).collect(engine="streaming"))
    dos = (pl.scan_parquet(VIEJO).select("id", kh=llave).collect(engine="streaming").join(dos, on="kh").drop("kh")
           .with_columns(misma_fecha=pl.col("f_reg_U") == pl.col("f_reg_O"), otras_despues=pl.col("f_reg_O") > pl.col("f_reg_U")))
    dos.write_parquet(SALIDA / "ordenes_dos_categorias.parquet")
    print(f"órdenes viejas con versión en las dos categorías: {dos.height:,}; misma fecha de registro {int(dos['misma_fecha'].sum()):,}; "
          f"«otras» posterior {int(dos['otras_despues'].sum()):,}", flush=True)
    if "--solo-dos-categorias" in sys.argv:
        return
    ver = (base.select(["kh", "fecha_registro", "ok_U", "ok_O", "anulada"]).group_by("kh").agg(
        fecha_registro_primera=pl.col("fecha_registro").filter(pl.col("ok_U") | pl.col("ok_O")).min(),
        f_ult_anulada=pl.col("fecha_registro").filter(pl.col("anulada")).max(),
        f_reg_U=pl.col("fecha_registro").filter(pl.col("ok_U")).max(),
        f_reg_O=pl.col("fecha_registro").filter(pl.col("ok_O")).max()).collect(engine="streaming"))
    paso(f"pasada 1: {ver.height:,} llaves")
    viejo_kh = pl.scan_parquet(VIEJO).select(kh=llave).collect(engine="streaming")
    id0 = int(pl.scan_parquet(VIEJO).select(pl.col("id").max()).collect().item()) + 1
    s = lambda c: pl.col(c).cast(pl.String).fill_null("nan")  # noqa: E731    igual que astype(str) de 01_base
    nuevas = (base.filter(pl.col("ok_O")).select(["ruc_entidad", "departamento_entidad", "orden", "objetocontractual",
                                                  "monto_total_orden_original", "ruc_contratista", "nombre_razon_contratista",
                                                  "fecha_registro", "fecha_de_emision", "fila", "kh"])
              .collect(engine="streaming").sort(["fecha_registro", "fila"]).unique("kh", keep="last", maintain_order=True)
              .join(viejo_kh, on="kh", how="anti")
              .select(ruc_entidad=s("ruc_entidad"), depto=s("departamento_entidad"), orden=s("orden"), ruc=s("ruc_contratista"),
                      nombre=s("nombre_razon_contratista"), fecha=pl.col("fecha_de_emision").dt.truncate("1d"),
                      monto=pl.col("monto_total_orden_original").cast(pl.Float64),
                      servicio=pl.col("objetocontractual").cast(pl.String).str.to_uppercase().str.contains("SERV|CONSULT").fill_null(False),
                      anio=pl.col("fecha_de_emision").dt.year().cast(pl.Int16), kh=pl.col("kh"), fecha_registro=pl.col("fecha_registro"))
              .sort(["anio", "ruc_entidad", "fecha", "orden"]).with_row_index("k")
              .with_columns(id=pl.col("k").cast(pl.Int64) + id0, otra_categoria=pl.lit(True)).drop("k"))
    del viejo_kh
    paso(f"nuevas: {nuevas.height:,}")
    ver_l = ver.lazy()
    viejas = (pl.scan_parquet(VIEJO).with_columns(kh=llave).join(ver_l.select(["kh", "f_reg_U"]), on="kh", how="left", maintain_order="left")
              .rename({"f_reg_U": "fecha_registro"}).with_columns(otra_categoria=pl.lit(False)))
    cols = COLS_V + ["kh", "fecha_registro", "otra_categoria"]
    todo = (pl.concat([viejas.select(cols), nuevas.lazy().select(cols)])
            .join(ver_l.select(["kh", "fecha_registro_primera", "f_ult_anulada", "f_reg_O"]), on="kh", how="left", maintain_order="left")
            .with_columns(anulada_despues=(pl.col("f_ult_anulada") > pl.col("fecha_registro")).fill_null(False),
                          anulada_mismo_dia_o_despues=(pl.col("f_ult_anulada") >= pl.col("fecha_registro")).fill_null(False),
                          ultima_version_otras=(~pl.col("otra_categoria") & (pl.col("f_reg_O") > pl.col("fecha_registro"))).fill_null(False))
            .drop(["f_ult_anulada", "f_reg_O", "kh"]))
    todo.sink_parquet(TEMP)
    del nuevas, ver, ver_l
    paso("tabla escrita en flujo")
    fallas = pruebas(id0)
    print("pruebas:", "todas bien" if not fallas else fallas, flush=True)
    if fallas:
        raise SystemExit(1)
    os.replace(TEMP, FINAL)
    paso("listo: ordenes.parquet reemplazada")


def pruebas(id0):
    h = lambda lf: lf.select(["id", pl.struct(COLS_V).hash(seed=3).alias("h")])  # noqa: E731
    hv = h(pl.scan_parquet(VIEJO)).collect(engine="streaming")
    hn = h(pl.scan_parquet(TEMP).filter(~pl.col("otra_categoria"))).collect(engine="streaming")
    j = hv.join(hn, on="id", how="full", coalesce=True, suffix="_n")
    fallas = []
    dif = j.filter(pl.col("h").is_null() | pl.col("h_n").is_null() | (pl.col("h") != pl.col("h_n"))).height
    if dif:
        fallas.append(f"parte vieja distinta en {dif:,} filas")
    del hv, hn, j
    r = pl.scan_parquet(TEMP).select(
        n=pl.len(), ids=pl.col("id").n_unique(), llaves=pl.struct(["ruc_entidad", "orden"]).hash(seed=11).n_unique(),
        nuevas=pl.col("otra_categoria").sum(), id_min_n=pl.col("id").filter(pl.col("otra_categoria")).min(),
        id_max_n=pl.col("id").filter(pl.col("otra_categoria")).max(),
        no_pos=(pl.col("monto") <= 0).sum(), sin_reg=pl.col("fecha_registro").null_count(),
        sin_reg1=pl.col("fecha_registro_primera").null_count(),
        nuevas_sobre=((pl.col("monto") > tope_expr(pl.col("anio").cast(pl.Int32), pl.col("fecha")) + 1e-6) & pl.col("otra_categoria")).sum(),
        anul_desp=pl.col("anulada_despues").sum(), anul_dia=pl.col("anulada_mismo_dia_o_despues").sum(),
        ult_otras=pl.col("ultima_version_otras").sum()).collect().to_dicts()[0]
    print("resumen:", r, flush=True)
    if r["ids"] != r["n"]:
        fallas.append("id repetidos")
    if r["llaves"] != r["n"]:
        fallas.append("(entidad, orden) repetido")
    if r["id_min_n"] != id0 or r["id_max_n"] != id0 + r["nuevas"] - 1:
        fallas.append("id nuevos no contiguos")
    if r["nuevas_sobre"] or r["no_pos"] or r["sin_reg"] or r["sin_reg1"]:
        fallas.append(f"tope, monto o fechas: {r}")
    por = pl.scan_parquet(TEMP).group_by(["anio", "otra_categoria"]).len().sort(["anio", "otra_categoria"]).collect()
    print("por año y categoría:", por.rows(), flush=True)
    ids = pl.read_parquet(TEMP, columns=["id"])["id"]
    if not bool((ids == pl.Series(range(len(ids)))).all()):
        fallas.append("el orden de las filas no es el del id")
    return fallas


if __name__ == "__main__":
    main()
