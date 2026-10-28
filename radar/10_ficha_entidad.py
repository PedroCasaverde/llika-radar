"""Paso 10: prototipo v0 en código. Arma, para una entidad y un año, el Excel que recibe el OCI: sus casos ordenados por
puntaje y monto, la ficha de cada orden con su evidencia y su norma, la fecha de cada fuente y columnas para que el
especialista confirme o descarte cada caso. Trae datos de personas: se guarda en `datos/procesados/radar/fichas/`, que no
se versiona. En pantalla solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 10_ficha_entidad.py --entidad 20131377224 --anio 2025
    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 10_ficha_entidad.py --ejemplo   (elige una municipalidad con varios tipos de alerta)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import CORTE_DJI, SALIDA, entidades  # noqa: E402

ALERTA = {"sancionado": "Sancionado que vende durante su sanción",
          "empresa_espejo": "Empresa espejo de un sancionado",
          "mismo_dueno": "Fraccionamiento entre RUC de un mismo dueño",
          "funcionario_propio": "Funcionario con cargo de decisión que se contrata",
          "doble_percepcion": "Doble percepción",
          "familiar": "Posible familiar de un funcionario de la entidad"}
NORMA = {"sancionado": "Ley 32069, art. 30: impedido por inhabilitación del Tribunal de Contrataciones",
         "empresa_espejo": "Ley 32069, art. 30: socios o representantes que integran una empresa sancionada",
         "mismo_dueno": "Ley 32069: prohibición de fraccionar; tope de 8 UIT de las compras menores",
         "funcionario_propio": "Ley 32069, art. 30: funcionarios de la entidad, durante su cargo",
         "doble_percepcion": "Ley 28175, art. 3: un solo ingreso del Estado (salvo docencia o un directorio)",
         "familiar": "Ley 32069, art. 30: parientes hasta el segundo grado; Ley 26771: nepotismo"}
FUENTE = {"sancionado": "Registro de sanciones del OECE (versión publicada después de la orden)",
          "empresa_espejo": "Socios y representantes del RNP (corte vigente el día de la orden)",
          "mismo_dueno": "Socios y representantes del RNP (corte vigente el día de la orden)",
          "funcionario_propio": f"Declaraciones juradas de intereses (foto al {CORTE_DJI:%d/%m/%Y})",
          "doble_percepcion": f"Declaraciones juradas de intereses (foto al {CORTE_DJI:%d/%m/%Y})",
          "familiar": f"Declaraciones juradas de intereses (foto al {CORTE_DJI:%d/%m/%Y}) y padrón de SUNAT"}

ap = argparse.ArgumentParser()
ap.add_argument("--entidad")
ap.add_argument("--anio", type=int, default=2025)
ap.add_argument("--ejemplo", action="store_true")
arg = ap.parse_args()

casos = pd.read_parquet(SALIDA / "radar_casos.parquet")
al = pd.read_parquet(SALIDA / "radar_alertas.parquet")
ent = entidades().drop_duplicates("RUC").set_index("RUC")
if arg.ejemplo or not arg.entidad:
    c = casos[(casos["anio"] == arg.anio) & casos["ruc_entidad"].map(ent["NOMBRE_DE_ENTIDAD"]).str.contains("MUNICIPALIDAD", na=False)]
    tipos = al[al["anio"] == arg.anio].groupby("ruc_entidad")["tarjeta"].nunique()
    n = c.groupby("ruc_entidad").size()
    elegibles = n[(n >= 8) & (n <= 15)].index.intersection(tipos[tipos >= 4].index)
    arg.entidad = sorted(elegibles)[0]
nombre_ent = ent["NOMBRE_DE_ENTIDAD"].get(arg.entidad, arg.entidad)

o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "orden", "ruc", "nombre", "anio"])
o = o[(o["ruc_entidad"] == arg.entidad) & (o["anio"] == arg.anio)]
nombres = o.drop_duplicates("ruc").set_index("ruc")["nombre"]
c = casos[(casos["ruc_entidad"] == arg.entidad) & (casos["anio"] == arg.anio)].sort_values("puesto_en_entidad")
a = al[(al["ruc_entidad"] == arg.entidad) & (al["anio"] == arg.anio)].merge(o[["id", "orden"]], on="id")

hoja_casos = pd.DataFrame({
    "Puesto": c["puesto_en_entidad"].to_numpy(),
    "Proveedor": c["ruc"].map(nombres).to_numpy(),
    "RUC": c["ruc"].to_numpy(),
    "Tipo": c["ruc"].str.startswith("10").map({True: "Persona natural", False: "Empresa"}).to_numpy(),
    "Alertas": c["tarjetas"].map(lambda t: "; ".join(ALERTA[x] for x in t.split(", "))).to_numpy(),
    "Puntaje": c["puntaje"].round(2).to_numpy(),
    "Órdenes": c["ordenes"].to_numpy(),
    "Monto (S/)": c["monto"].round(2).to_numpy(),
    "Evidencia": c["evidencia"].to_numpy(),
    "Decisión del OCI (confirma / descarta)": "",
    "Motivo": ""})
puesto = c.set_index("ruc")["puesto_en_entidad"]
hoja_ordenes = pd.DataFrame({
    "Puesto del caso": a["ruc"].map(puesto).to_numpy(),
    "Orden": a["orden"].to_numpy(),
    "Fecha": a["fecha"].dt.date.to_numpy(),
    "Proveedor": a["ruc"].map(nombres).to_numpy(),
    "Monto (S/)": a["monto"].round(2).to_numpy(),
    "Alerta": a["tarjeta"].map(ALERTA).to_numpy(),
    "Probabilidad": a["prob"].round(2).to_numpy(),
    "Evidencia": a["evidencia"].to_numpy(),
    "Norma": a["tarjeta"].map(NORMA).to_numpy(),
    "Fuente": a["tarjeta"].map(FUENTE).to_numpy()}).sort_values(["Puesto del caso", "Fecha"])
hoja_fuentes = pd.DataFrame({
    "Fuente": ["Órdenes de compras menores (CONOSCE)", "Registro de sanciones del Tribunal (OECE)",
               "Socios y representantes de proveedores (RNP, OECE)", "Declaraciones juradas de intereses (Contraloría)",
               "Padrón reducido de RUC (SUNAT)"],
    "Corte usado": ["Mensual, 2018 a 2025", "62 versiones, de 09/2021 a 10/2026", "50 cortes mensuales, de 11/2021 a 10/2026",
                    f"Foto al {CORTE_DJI:%d/%m/%Y}", "Foto de 10/2026"]})

dest = SALIDA / "fichas"
dest.mkdir(exist_ok=True)
archivo = dest / f"ficha_{arg.entidad}_{arg.anio}.xlsx"
with pd.ExcelWriter(archivo, engine="openpyxl") as xw:
    for nombre, df in [("Casos", hoja_casos), ("Órdenes y evidencia", hoja_ordenes), ("Fuentes", hoja_fuentes)]:
        df.to_excel(xw, sheet_name=nombre, index=False)
        ws = xw.sheets[nombre]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            ancho = max(len(str(v.value)) if v.value is not None else 0 for v in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, ancho + 2), 60)
print(f"entidad: {nombre_ent} ({arg.entidad}), {arg.anio}: {len(c)} casos ({(c['ruc'].str.startswith('10')).sum()} de personas), "
      f"{len(a)} alertas en {a['id'].nunique()} órdenes, S/ {c['monto'].sum():,.0f}; tipos: {a['tarjeta'].value_counts().to_dict()}")
print(f"Excel: {archivo}")
