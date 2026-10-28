"""Paso 6: puntaje de prioridad, tabla final del radar y prueba de los pesos con Lasso.

Puntaje por puntos (gravedad legal x probabilidad), por alerta:
- 3: sancionado que vende durante su sanción (1 si es un servicio público: telefonía, luz, agua, correo), empresa espejo
  (1 si el vínculo pasa por un gremio), funcionario con cargo de decisión que le vende a su entidad;
- 2: doble percepción (1 si le paga una entidad educativa: posible docencia), compra partida entre RUC del mismo dueño,
  posible familiar (3 si el funcionario tiene cargo de decisión), multiplicado por la probabilidad de parentesco.
Un caso es un proveedor en una entidad y un año; su puntaje suma, una vez por tipo, la alerta más alta de cada tipo
(como los índices de banderas rojas, que suman banderas). Se ordena por puntaje y, si empatan, por monto.

La prueba de los pesos con Lasso está en 08_lasso.py. Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 06_puntaje.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA  # noqa: E402

TARJETAS = ["sancionado", "empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "anio", "monto"])

partes = []
for t in TARJETAS:
    f = SALIDA / f"alerta_{t}.parquet"
    if not f.exists():
        print(f"(falta la alerta {t}: se omite)")
        continue
    a = pd.read_parquet(f)
    a["tarjeta"] = t
    a["prob"] = a["prob"] if "prob" in a else 1.0
    g = {"sancionado": np.where(a.get("servicio_publico", False), 1, 3),
         "empresa_espejo": np.where(a.get("gremio", False), 1, 3), "funcionario_propio": 3,
         "doble_percepcion": np.where(a.get("docencia", False), 1, 2), "mismo_dueno": 2,
         "familiar": np.where(a.get("decide", False), 3, 2)}[t]
    a["gravedad"] = g
    partes.append(a[["id", "tarjeta", "prob", "gravedad", "evidencia"]])
al = pd.concat(partes, ignore_index=True).merge(o, on="id")
al["puntos"] = al["gravedad"] * al["prob"]
al.to_parquet(SALIDA / "radar_alertas.parquet", index=False)

# ---------------------------------------------------------------- casos: proveedor x entidad x año
por_tipo = al.groupby(["ruc_entidad", "ruc", "anio", "tarjeta"]).agg(puntos=("puntos", "max"), evidencia=("evidencia", "first"))
casos = por_tipo.groupby(level=[0, 1, 2]).agg(puntaje=("puntos", "sum"),
                                              tarjetas=("puntos", lambda x: ", ".join(sorted(x.index.get_level_values(3)))),
                                              evidencia=("evidencia", lambda x: " | ".join(x)))
montos = al.drop_duplicates("id").groupby(["ruc_entidad", "ruc", "anio"]).agg(monto=("monto", "sum"), ordenes=("id", "size"))
casos = casos.join(montos).reset_index().sort_values(["anio", "ruc_entidad", "puntaje", "monto"], ascending=[True, True, False, False])
casos["puesto_en_entidad"] = casos.groupby(["anio", "ruc_entidad"]).cumcount() + 1
casos.to_parquet(SALIDA / "radar_casos.parquet", index=False)
print(f"RADAR: {len(al):,} alertas en {al['id'].nunique():,} órdenes; {len(casos):,} casos (proveedor, entidad y año) en "
      f"{casos['ruc_entidad'].nunique():,} entidades; S/ {casos['monto'].sum() / 1e6:,.1f} millones")
print("  casos por número de tipos de alerta:", casos["tarjetas"].str.count(",").add(1).value_counts().sort_index().to_dict())
print("  casos por año:", casos.groupby("anio").size().to_dict())
print("  distribución del puntaje:", casos["puntaje"].describe(percentiles=[.5, .9, .99]).round(2).to_dict())
