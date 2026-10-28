"""Paso 11: muestra para verificar a mano la alerta de posible familiar en la consulta pública de declaraciones juradas
de la Contraloría (appdji.contraloria.gob.pe/djic/, «Buscar familiares»), que tiene captcha y no se puede bajar en masa.

Toma las parejas persona-entidad de 2025 con alerta de familiar y saca 20 al azar de cada grupo de probabilidad
(menos de 50 %, de 50 % a 80 % y 80 % o más). Para cada una, lista al funcionario o funcionarios de la entidad con los
mismos apellidos y su cargo. Quien verifica anota si la persona figura como pariente en la declaración del funcionario.
La medida es la proporción confirmada en cada grupo: debe subir con la probabilidad. Trae nombres: se guarda en
`datos/procesados/radar/verificacion/`, que no se versiona. En pantalla solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 11_muestra_parentesco.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, entidades, funcionarios, norm_serie  # noqa: E402

ANIO, POR_GRUPO = 2025, 20
fa = pd.read_parquet(SALIDA / "alerta_familiar.parquet", columns=["id", "prob"])
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "nombre", "fecha", "anio", "monto"])
a = fa.merge(o[o["anio"] == ANIO], on="id")
p = a.groupby(["ruc_entidad", "ruc"]).agg(nombre=("nombre", "first"), prob=("prob", "max"), desde=("fecha", "min"),
                                           hasta=("fecha", "max"), ordenes=("id", "size"), monto=("monto", "sum")).reset_index()
p["grupo"] = pd.cut(p["prob"], [-0.01, 0.5, 0.8, 1.0], labels=["menos de 50 %", "50 % a 80 %", "80 % o más"])
print(f"parejas persona-entidad con alerta de familiar en {ANIO}: {len(p):,}; por grupo: "
      f"{p['grupo'].value_counts().sort_index().to_dict()}")
m = pd.concat([g.sample(n=min(POR_GRUPO, len(g)), random_state=44) for _, g in p.groupby("grupo", observed=True)],
              ignore_index=True)

# funcionarios de la entidad con los mismos apellidos y cargo durante las órdenes
f = funcionarios()
m["nombre_n"] = norm_serie(m["nombre"])
t = m["nombre_n"].str.split()
filas = []
for k in (2, 3, 4):
    ok = t.str.len() > k
    filas.append(pd.DataFrame({"fila": m.index[ok], "par": t[ok].str[:k].str.join(" ")}))
pares = pd.concat(filas).merge(m[["ruc_entidad", "desde", "hasta"]], left_on="fila", right_index=True)
cand = pares.merge(f[["persona", "par", "ruc_entidad", "cargo", "decide", "inicio", "cese"]], on=["ruc_entidad", "par"])
cand = cand[(cand["inicio"] <= cand["hasta"]) & (cand["cese"] >= cand["desde"])]
cand = cand.sort_values("decide", ascending=False).drop_duplicates(["fila", "persona", "cargo"])
cand["texto"] = (cand["persona"] + " (" + cand["cargo"] + ", " + cand["inicio"].dt.strftime("%d/%m/%Y") + " a "
                 + cand["cese"].dt.strftime("%d/%m/%Y") + ")")
func = cand.groupby("fila")["texto"].agg(" | ".join)
ent = entidades().drop_duplicates("RUC").set_index("RUC")["NOMBRE_DE_ENTIDAD"]
salida = pd.DataFrame({
    "N.°": np.arange(1, len(m) + 1),
    "Grupo de probabilidad": m["grupo"].astype(str),
    "Probabilidad": m["prob"].round(2),
    "Entidad": m["ruc_entidad"].map(ent),
    "RUC de la entidad": m["ruc_entidad"],
    "Persona que vende": m["nombre"],
    "DNI (del RUC 10)": m["ruc"].str[2:10],
    "Órdenes en el año": m["ordenes"],
    "Monto (S/)": m["monto"].round(2),
    "Funcionario(s) con los mismos apellidos y cargo": m.index.map(func),
    "¿Figura como pariente en la declaración? (sí / no / declaración no encontrada)": "",
    "Parentesco declarado": "",
    "Fecha de consulta": "",
    "Quién consultó": ""})
dest = SALIDA / "verificacion"
dest.mkdir(exist_ok=True)
archivo = dest / f"muestra_parentesco_{ANIO}.xlsx"
with pd.ExcelWriter(archivo, engine="openpyxl") as xw:
    salida.to_excel(xw, sheet_name="Muestra", index=False)
    ws = xw.sheets["Muestra"]
    ws.freeze_panes = "A2"
    for col in ws.columns:
        ancho = max(len(str(v.value)) if v.value is not None else 0 for v in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(10, ancho + 2), 70)
print(f"muestra: {len(salida)} parejas; probabilidad media por grupo: "
      f"{m.groupby('grupo', observed=True)['prob'].mean().round(2).to_dict()}; "
      f"con funcionario encontrado: {salida['Funcionario(s) con los mismos apellidos y cargo'].notna().sum()}")
print(f"Excel: {archivo}")
