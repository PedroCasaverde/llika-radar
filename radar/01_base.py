"""Paso 1: base única de órdenes y de sanciones, y la alerta «sancionado que vende durante su sanción».

Órdenes: contratos menores de CONOSCE emitidos de 2018 a 2025, en soles, sin anuladas, sin repetidas (entidad y número
de orden, que incluye el año) y con monto posible: mayor que cero y no más que el tope del año (9 UIT en 2022 por el
DU 016-2022, 8 UIT los demás).
Sanciones del Tribunal: unión de las 62 versiones de la wiki del OECE (desde 09/2021) y del archivo vigente; las que
no tienen fecha de fin son inhabilitaciones definitivas. Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 01_base.py
"""
import glob
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # con -I la carpeta del script no entra al path
from comun import CRUDOS, PROC, SALIDA, ANIOS, UIT  # noqa: E402

SERVICIO_PUBLICO = (r"TELEFONICA|AMERICA MOVIL|\bCLARO\b|\bENTEL\b|VIETTEL|\bBITEL\b|NEXTEL|\bELECTRO|\bENEL\b|LUZ DEL SUR"
                    r"|HIDRANDINA|SEDAPAL|SEDAPAR|SEDALIB|SEDACUSCO|\bEPS\b|EMAPA|SERVICIO POSTAL|SERPOST")
CABECERA = ["FECHA_CORTE", "RUC", "NOMBRE_RAZONODENOMINACIONSOCIAL", "FECHA_INICIO", "FECHA_FIN", "NUMERO_RESOLUCION",
            "ID_MOTIVO_INFRACCION", "DE_MOTIVO_INFRACCION"]


def leer_sanciones(f):
    """Algunas versiones vienen sin cabecera o con una línea en blanco antes; todas en cp1252."""
    txt = open(f, "rb").read().decode("cp1252", "replace").lstrip("﻿\r\n")
    con_cabecera = txt.split("\n", 1)[0].startswith(("FECHA_CORTE", "RUC"))
    s = pd.read_csv(io.StringIO(txt), sep="|", dtype=str, header=0 if con_cabecera else None, index_col=False)
    if not con_cabecera:
        s = s.iloc[:, :len(CABECERA)]
        s.columns = CABECERA[:s.shape[1]]
    s.columns = [str(c).strip().strip(",") for c in s.columns]
    return s

SALIDA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- órdenes
cols = ["ruc_entidad", "departamento_entidad", "orden", "objetocontractual", "estadocontratacion", "tipodecontratacion",
        "monto_total_orden_original", "moneda", "ruc_contratista", "nombre_razon_contratista", "fecha_registro",
        "fecha_de_emision"]
partes, filas = [], 0
archivos = [] if (SALIDA / "ordenes.parquet").exists() and "--rehacer" not in sys.argv else \
    sorted(glob.glob(str(PROC / "acc1_conosce" / "ordenes_20??-??.parquet")))
for f in archivos:
    d = pd.read_parquet(f, columns=cols)
    filas += len(d)
    anio = d["fecha_de_emision"].dt.year
    tope = anio.map(lambda a: (9 if a == 2022 else 8) * UIT.get(a, 0))
    d = d[anio.isin(ANIOS)
          & d["tipodecontratacion"].astype(str).str.upper().str.startswith("CONTRATACIONES HASTA")
          & d["moneda"].eq("Nuevos Soles") & ~d["estadocontratacion"].eq("Anulada")
          & (d["monto_total_orden_original"] > 0) & (d["monto_total_orden_original"] <= tope)]
    partes.append(d.drop(columns=["tipodecontratacion", "moneda", "estadocontratacion"]))
if archivos:
    o = pd.concat(partes, ignore_index=True)
    antes = len(o)
    o = o.sort_values("fecha_registro").drop_duplicates(["ruc_entidad", "orden"], keep="last")
    o = pd.DataFrame({
        "ruc_entidad": o["ruc_entidad"].astype(str), "depto": o["departamento_entidad"].astype(str),
        "orden": o["orden"].astype(str), "ruc": o["ruc_contratista"].astype(str),
        "nombre": o["nombre_razon_contratista"].astype(str), "fecha": o["fecha_de_emision"].dt.normalize(),
        "monto": o["monto_total_orden_original"].astype(float),
        "servicio": o["objetocontractual"].astype(str).str.upper().str.contains("SERV|CONSULT", regex=True)})
    o["anio"] = o["fecha"].dt.year.astype("int16")
    o = o.sort_values(["anio", "ruc_entidad", "fecha"]).reset_index(drop=True)
    o["id"] = o.index.astype("int64")
    o.to_parquet(SALIDA / "ordenes.parquet", index=False)
    print(f"filas leídas {filas:,}; contratos menores en soles 2018-2025 con monto posible {antes:,}; sin repetidas {len(o):,}")
else:
    o = pd.read_parquet(SALIDA / "ordenes.parquet")
    print(f"órdenes ya armadas: {len(o):,} (para rehacerlas: --rehacer)")
print("órdenes por año:", o.groupby("anio").size().to_dict())
print(f"monto total S/ {o['monto'].sum() / 1e9:,.2f} mil millones; a personas naturales (RUC 10): "
      f"{o.loc[o['ruc'].str.startswith('10'), 'monto'].sum() / o['monto'].sum():.0%}")

# ---------------------------------------------------------------- sanciones
partes = []
for f in sorted(glob.glob(str(CRUDOS / "acc7_sancionados" / "_historia" / "*.csv"))) + [
        sorted(glob.glob(str(CRUDOS / "acc7_sancionados" / "sancionados_*.csv")))[-1]]:
    nombre = Path(f).stem
    corte = pd.to_datetime(nombre[-10:] if "_v" in nombre else nombre[-8:], format="%Y-%m-%d" if "_v" in nombre else "%Y%m%d")
    s = leer_sanciones(f)
    if "Desde" in s.columns:  # formato de 2021
        s = pd.DataFrame({"ruc": s["RUC"], "inicio": pd.to_datetime(s["Desde"].str[:10], format="%d/%m/%Y", errors="coerce"),
                          "fin": pd.to_datetime(s["Hasta"].str[:10], format="%d/%m/%Y", errors="coerce"),
                          "resolucion": s["Resolucion"], "motivo": None})
    else:
        s = pd.DataFrame({"ruc": s["RUC"], "inicio": pd.to_datetime(s["FECHA_INICIO"], format="%Y%m%d", errors="coerce"),
                          "fin": pd.to_datetime(s["FECHA_FIN"], format="%Y%m%d", errors="coerce"),
                          "resolucion": s["NUMERO_RESOLUCION"], "motivo": s["DE_MOTIVO_INFRACCION"]})
    partes.append(s.assign(corte=corte))
s = pd.concat(partes, ignore_index=True)
s["ruc"] = s["ruc"].astype(str).str.strip()
s["resolucion"] = s["resolucion"].astype(str).str.strip().str.upper()
s = s[s["ruc"].str.fullmatch(r"\d{11}") & s["inicio"].notna()]
LLAVE = ["ruc", "inicio", "fin", "resolucion"]
# en qué versiones publicadas aparece cada sanción: si después desaparece, pudo ser anulada o suspendida por un juez
apariciones = s[LLAVE + ["corte"]].drop_duplicates()
apariciones["fin"] = apariciones["fin"].fillna(pd.Timestamp("2100-01-01"))
# el motivo solo viene en el formato nuevo: se completa con cualquier versión que lo traiga
motivo = s.dropna(subset=["motivo"]).drop_duplicates(["ruc", "resolucion"]).set_index(["ruc", "resolucion"])["motivo"]
s = s.drop(columns=["motivo", "corte"]).drop_duplicates(LLAVE)
s = s.join(motivo, on=["ruc", "resolucion"])
s["definitiva"] = s["fin"].isna()
s.to_parquet(SALIDA / "sanciones.parquet", index=False)
apariciones.to_parquet(SALIDA / "sanciones_apariciones.parquet", index=False)
PRIMERA = apariciones["corte"].min()
print(f"\nsanciones distintas: {len(s):,} de {s['ruc'].nunique():,} RUC; definitivas (sin fecha de fin): {int(s['definitiva'].sum()):,}; "
      f"por contratar estando impedido: {int(s['motivo'].fillna('').str.upper().str.contains('IMPEDID').sum()):,}")

# ---------------------------------------------------------------- alerta: sancionado que vende durante su sanción
m = o[["id", "ruc_entidad", "ruc", "fecha", "anio", "monto"]].merge(s[["ruc", "inicio", "fin", "resolucion", "definitiva", "motivo"]], on="ruc")
m = m[(m["fecha"] >= m["inicio"]) & (m["definitiva"] | (m["fecha"] <= m["fin"]))]
# la orden cuenta si el registro publicado en las 6 semanas siguientes todavía mostraba la sanción; antes de la
# primera versión guardada (29/09/2021) basta con que aparezca en esa primera versión
ap = apariciones.assign(fin=lambda x: x["fin"].where(x["fin"] < pd.Timestamp("2100-01-01")))
x = m[["id", "fecha"] + LLAVE].merge(ap, on=LLAVE)
ventana = (x["corte"] >= x["fecha"]) & (x["corte"] <= x["fecha"] + pd.Timedelta(days=45))
temprana = (x["fecha"] < PRIMERA) & (x["corte"] == PRIMERA)
publicadas = set(x.loc[ventana | temprana, "id"])
print(f"órdenes dentro de una sanción: {m['id'].nunique():,}; confirmadas por el registro publicado justo después: "
      f"{len(publicadas):,}")
m = m[m["id"].isin(publicadas)]
a = m.sort_values("definitiva").drop_duplicates("id").copy()  # una orden con sanciones superpuestas cuenta una vez
a["tarjeta"] = "sancionado"
a["evidencia"] = "sancion " + a["resolucion"] + a["definitiva"].map({True: " (definitiva)", False: ""})
# pagos a empresas de servicios públicos (telefonía, luz, agua, correo): casi siempre siguen un contrato o un servicio
# regulado que la entidad no puede cambiar de un día a otro; se marcan aparte y pesan menos
nombres = o.drop_duplicates("ruc").set_index("ruc")["nombre"].str.upper()
a["servicio_publico"] = a["ruc"].map(nombres).str.contains(SERVICIO_PUBLICO, regex=True, na=False)
a[["id", "tarjeta", "definitiva", "servicio_publico", "resolucion", "inicio", "fin", "motivo", "evidencia"]].to_parquet(
    SALIDA / "alerta_sancionado.parquet", index=False)
print("\nALERTA sancionado que vende durante su sanción:")
for sp, g in a.groupby("servicio_publico"):
    print(f"  {'servicio público (telefonía, luz, agua, correo)' if sp else 'resto de proveedores'}: {len(g):,} órdenes, "
          f"{g['ruc'].nunique():,} proveedores, {g['ruc_entidad'].nunique():,} entidades, S/ {g['monto'].sum() / 1e6:,.1f} millones")
for definitiva, g in a.groupby("definitiva"):
    print(f"  {'definitiva' if definitiva else 'con fecha de fin'}: {len(g):,} órdenes, {g['ruc'].nunique():,} proveedores, "
          f"{g['ruc_entidad'].nunique():,} entidades, S/ {g['monto'].sum() / 1e6:,.1f} millones")
print("  por año (órdenes, millones de soles):",
      {int(k): (int(v["n"]), round(v["m"] / 1e6, 1)) for k, v in a.groupby("anio").agg(n=("id", "size"), m=("monto", "sum")).iterrows()})
top = a.groupby(["anio", "ruc"]).size().sort_values(ascending=False).groupby(level=0).head(1)
print("  proveedor con más órdenes alertadas en cada año (órdenes; persona natural o empresa):",
      {int(k[0]): (int(v), "persona" if k[1].startswith("10") else "empresa") for k, v in top.items()})
c24 = a[(a["anio"] == 2024) & ~a["definitiva"]]
print(f"  control: 2024 con fecha de fin = {len(c24):,} órdenes, S/ {c24['monto'].sum() / 1e6:,.1f} millones "
      f"(el E4 dio 4 881 órdenes y S/ 9,4 millones solo con el archivo vigente)")
