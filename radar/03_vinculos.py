"""Paso 3: alertas que salen de los socios, representantes y administradores de los proveedores (OECE, un corte por
mes desde noviembre de 2021), con el vínculo vigente el día de cada orden.

- Empresa espejo: una empresa le vende a una entidad mientras una persona que la integra está sancionada (persona
  natural) o integra, en el mismo corte, una empresa sancionada; la sanción, vigente y publicada (paso 1).
- Mismo dueño con varios RUC: en una entidad y un año, una persona y su empresa (o dos o más empresas de la misma
  persona) quedan cada una bajo el tope y juntas lo pasan: la compra queda partida entre RUC del mismo dueño.
Para las órdenes anteriores al primer corte (11/11/2021) se usa ese primer corte y la alerta lo indica.
Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 03_vinculos.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import PROC, SALIDA, UIT  # noqa: E402
from comun_sanciones import vigentes_publicadas  # noqa: E402


def resumen(nombre, a):
    print(f"  {nombre}: {len(a):,} órdenes, {a['ruc'].nunique():,} proveedores, {a['ruc_entidad'].nunique():,} entidades, "
          f"S/ {a['monto'].sum() / 1e6:,.1f} millones; por año: {a.groupby('anio').size().to_dict()}")


# ---------------------------------------------------------------- vínculos persona-empresa por corte
fs = sorted(glob.glob(str(PROC / "acc3_conformacion" / "conformacion_corte*.parquet")))
partes = []
for f in fs:
    v = pd.read_parquet(f, columns=["fecha_corte", "ruc_proveedor", "dni_integrante", "usable_para_cruce"])
    v = v[v["usable_para_cruce"] & v["dni_integrante"].notna()]
    partes.append(pd.DataFrame({"corte": v["fecha_corte"].astype("datetime64[ns]"), "emp": v["ruc_proveedor"].astype(str),
                                "dni": v["dni_integrante"].astype(str)}).drop_duplicates())
L = pd.concat(partes, ignore_index=True)
cortes = np.sort(L["corte"].unique())
print(f"{len(fs)} cortes, de {pd.Timestamp(cortes[0]).date()} a {pd.Timestamp(cortes[-1]).date()}; vínculos persona-empresa "
      f"(persona con DNI, fila usable): {len(L):,}")

o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "anio", "monto"])
o["fecha"] = o["fecha"].astype("datetime64[ns]")
o = pd.merge_asof(o.sort_values("fecha"), pd.DataFrame({"corte": cortes}), left_on="fecha", right_on="corte", direction="backward")
o["aprox"] = o["corte"].isna()
o["corte"] = o["corte"].fillna(pd.Timestamp(cortes[0]))
tope = o["anio"].map(lambda a: (9 if a == 2022 else 8) * UIT[a])

# ---------------------------------------------------------------- empresa espejo
s = pd.read_parquet(SALIDA / "sanciones.parquet")
nat = s[s["ruc"].str.startswith("10")].assign(dni=lambda d: d["ruc"].str[2:10])
jur = s[s["ruc"].str.startswith("20")].merge(L, left_on="ruc", right_on="emp").drop(columns="emp")  # personas de la sancionada, por corte
riesgo = set(nat["dni"]) | set(jur["dni"])
Lr = L[L["dni"].isin(riesgo)]
oo = o[o["ruc"].str.startswith("20")].merge(Lr, left_on=["ruc", "corte"], right_on=["emp", "corte"])  # personas de la proveedora ese día
a = oo.merge(nat.drop(columns="ruc").assign(ruc_sanc=nat["ruc"]), on="dni")
a = a[a["ruc"] != a["ruc_sanc"]]
b = oo.merge(jur.rename(columns={"ruc": "ruc_sanc"}), on=["dni", "corte"])  # la misma persona, en el mismo corte
b = b[b["ruc"] != b["ruc_sanc"]]
x = pd.concat([a.assign(camino="sancionado persona natural"), b.assign(camino="integra una empresa sancionada")], ignore_index=True)
x = x.rename(columns={"ruc": "ruc_prov"}).assign(ruc=lambda d: d["ruc_sanc"])
x = vigentes_publicadas(x).drop(columns="ruc").rename(columns={"ruc_prov": "ruc"})
x = x.sort_values("camino").drop_duplicates("id")
x["tarjeta"] = "empresa_espejo"
# si la sancionada es un gremio (cámara de comercio, sociedad nacional, asociación), sus directivos están en muchas
# empresas: el vínculo es legal pero débil, y pesa menos
nom = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["ruc", "nombre"]).drop_duplicates("ruc").set_index("ruc")["nombre"].str.upper()
x["gremio"] = x["ruc_sanc"].map(nom).str.contains(r"CAMARA DE COMERCIO|SOCIEDAD NACIONAL|ASOCIACION|CONFEDERACION|GREMIO",
                                                  regex=True, na=False)
x["evidencia"] = "un integrante esta vinculado a " + x["ruc_sanc"] + ", sancionado (" + x["resolucion"] + ")" + \
    x["aprox"].map({True: "; vinculo tomado del primer corte (11/11/2021)", False: ""})
x[["id", "tarjeta", "camino", "gremio", "ruc_sanc", "resolucion", "aprox", "evidencia"]].to_parquet(
    SALIDA / "alerta_empresa_espejo.parquet", index=False)
print("\nALERTA empresa espejo (vende mientras un integrante está vinculado a un sancionado):")
resumen("todas", x)
for c, g in x.groupby("camino"):
    resumen(c, g)
resumen("vía un gremio (pesa menos)", x[x["gremio"]])
c24 = x[x["anio"] == 2024]
print(f"  control 2024: {c24['ruc'].nunique():,} empresas, {len(c24):,} órdenes, S/ {c24['monto'].sum() / 1e6:,.1f} millones "
      f"(E4 con el archivo vigente: 539 empresas, 6 475 órdenes, S/ 69,2 millones)")

# ---------------------------------------------------------------- mismo dueño con varios RUC
ventas = o.groupby(["anio", "ruc_entidad", "ruc"]).agg(monto=("monto", "sum")).reset_index()
ventas["tope"] = ventas["anio"].map(lambda a: (9 if a == 2022 else 8) * UIT[a])
emp_cortes = o[o["ruc"].str.startswith("20")][["anio", "ruc_entidad", "ruc", "corte"]].drop_duplicates()
cp = emp_cortes.merge(L, left_on=["ruc", "corte"], right_on=["emp", "corte"])[["anio", "ruc_entidad", "ruc", "dni"]].drop_duplicates()
cp = cp.merge(ventas, on=["anio", "ruc_entidad", "ruc"])
# U1: la persona vende con su RUC 10 y su empresa también, en la misma entidad y año
r10 = ventas[ventas["ruc"].str.startswith("10")].assign(dni=lambda d: d["ruc"].str[2:10])
u1 = r10.merge(cp, on=["anio", "ruc_entidad", "dni"], suffixes=("_p", "_e"))
u1 = u1[(u1["monto_p"] <= u1["tope_p"]) & (u1["monto_e"] <= u1["tope_p"]) & (u1["monto_p"] + u1["monto_e"] > u1["tope_p"])]
# U2: dos o más empresas de la misma persona, cada una bajo el tope y el grupo sobre el tope
g = cp.groupby(["anio", "ruc_entidad", "dni"]).agg(n=("ruc", "nunique"), suma=("monto", "sum"), maximo=("monto", "max"),
                                                   tope=("tope", "first")).reset_index()
u2 = g[(g["n"] >= 2) & (g["maximo"] <= g["tope"]) & (g["suma"] > g["tope"])]
grupos = pd.concat([
    u1[["anio", "ruc_entidad", "ruc_p", "dni"]].rename(columns={"ruc_p": "ruc"}),
    u1[["anio", "ruc_entidad", "ruc_e", "dni"]].rename(columns={"ruc_e": "ruc"}),
    cp.merge(u2[["anio", "ruc_entidad", "dni"]], on=["anio", "ruc_entidad", "dni"])[["anio", "ruc_entidad", "ruc", "dni"]]])
grupos = grupos.drop_duplicates(["anio", "ruc_entidad", "ruc"])
y = o.merge(grupos, on=["anio", "ruc_entidad", "ruc"])
y["tarjeta"] = "mismo_dueno"
y["evidencia"] = "el mismo dueño vende con varios RUC a la entidad en el año; por separado bajo el tope, juntos lo pasan"
y[["id", "tarjeta", "aprox", "evidencia"]].to_parquet(SALIDA / "alerta_mismo_dueno.parquet", index=False)
print("\nALERTA mismo dueño con varios RUC (compra partida entre RUC del mismo dueño):")
print(f"  casos persona y su empresa: {len(u1):,}; casos de dos o más empresas: {len(u2):,}")
resumen("órdenes involucradas", y)
for k, v in {2024: (1372, 1281)}.items():
    print(f"  control {k}: persona y su empresa {int((u1['anio'] == k).sum()):,}, dos o más empresas {int((u2['anio'] == k).sum()):,} "
          f"(E4 con vínculo vigente: {v[0]:,} y {v[1]:,})")
