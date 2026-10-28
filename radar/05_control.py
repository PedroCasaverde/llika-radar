"""Paso 5: control de confiabilidad del radar (el corte de las 3 horas, con criterios fijados antes de mirar).

1. Cifras por tipo de alerta, de 2018 a 2025.
2. Muestra al azar de 10 alertas por tipo, sin nombres ni RUC de personas, para leerlas.
3. Prueba externa: los proveedores marcados, ¿reciben después más sanciones del Tribunal que el resto? Se compara con
   todos los proveedores del mismo tipo (persona o empresa) que vendieron: sanción que empieza después de la primera
   alerta (o de la primera venta, en la base). Intervalos de Wilson.
4. Probabilidad de parentesco: ¿las alertas con más probabilidad tienen más sanciones posteriores por impedimento?

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 05_control.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, entidades, wilson  # noqa: E402

TARJETAS = ["sancionado", "empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "anio", "monto", "servicio"])
ent = entidades().set_index("RUC")["NOMBRE_DE_ENTIDAD"].str.upper()
tipo_ent = np.select([ent.str.contains("MUNICIPALIDAD", na=False), ent.str.contains("GOBIERNO REGIONAL", na=False)],
                     ["municipalidad", "gobierno regional"], "otra entidad")
tipo_ent = pd.Series(tipo_ent, index=ent.index)
s = pd.read_parquet(SALIDA / "sanciones.parquet")
s["impedido"] = s["motivo"].fillna("").str.upper().str.contains("IMPEDID")

alertas = {}
for t in TARJETAS:
    f = SALIDA / f"alerta_{t}.parquet"
    if f.exists():
        alertas[t] = pd.read_parquet(f).merge(o, on="id")

print("1) CIFRAS POR TIPO DE ALERTA, 2018-2025")
for t, a in alertas.items():
    print(f"  {t:<20}{len(a):>9,} órdenes{a['ruc'].nunique():>8,} proveedores{a['ruc_entidad'].nunique():>7,} entidades"
          f"   S/ {a['monto'].sum() / 1e6:>7,.1f} millones")

print("\n2) MUESTRA AL AZAR DE 10 ALERTAS POR TIPO (sin nombres ni RUC de personas)")
rng = np.random.default_rng(23)
for t, a in alertas.items():
    m = a.iloc[rng.choice(len(a), size=min(10, len(a)), replace=False)]
    print(f"\n  [{t}]")
    for _, r in m.iterrows():
        prov = "persona natural" if r["ruc"].startswith("10") else f"empresa {r['ruc']}"
        ev = r.get("evidencia", "")
        print(f"   {r['fecha'].date()} | {tipo_ent.get(r['ruc_entidad'], 'otra entidad'):<17} | {prov:<20} | S/ {r['monto']:>9,.0f} | {ev}")

print("\n3) PRUEBA EXTERNA: SANCIÓN DEL TRIBUNAL QUE EMPIEZA DESPUÉS DE LA ALERTA")
primera_venta = o.groupby("ruc")["fecha"].min()
ini_sancion = s.groupby("ruc")["inicio"].apply(list)
ini_imped = s[s["impedido"]].groupby("ruc")["inicio"].apply(list)


def sancionado_despues(rucs_fechas, tabla):
    return np.array([any(i > f for i in tabla.get(r, [])) for r, f in rucs_fechas.items()])


for tipo, pref in [("personas", "10"), ("empresas", "20")]:
    base = primera_venta[primera_venta.index.str.startswith(pref)]
    b_any, b_imp = sancionado_despues(base, ini_sancion), sancionado_despues(base, ini_imped)
    print(f"  base, {tipo} que vendieron ({len(base):,}): cualquier sanción posterior {b_any.mean():.3%}, "
          f"por impedimento {b_imp.mean():.3%}")
    for t, a in alertas.items():
        aa = a[a["ruc"].str.startswith(pref)]
        if len(aa) == 0:
            continue
        pr = aa.groupby("ruc")["fecha"].min()
        x_any, x_imp = sancionado_despues(pr, ini_sancion), sancionado_despues(pr, ini_imped)
        lo, hi = wilson(int(x_any.sum()), len(x_any))
        print(f"    {t:<20}{len(pr):>7,} {tipo}: cualquier sanción posterior {x_any.mean():.2%} "
              f"(IC {lo:.2%} a {hi:.2%}; {x_any.mean() / max(b_any.mean(), 1e-9):.1f} veces la base); "
              f"por impedimento {int(x_imp.sum())} ({x_imp.mean() / max(b_imp.mean(), 1e-9):.1f} veces)")

if "familiar" in alertas:
    print("\n4) PROBABILIDAD DE PARENTESCO Y SANCIONES POSTERIORES (por persona)")
    fa = alertas["familiar"].groupby("ruc").agg(prob=("prob", "max"), fecha=("fecha", "min"))
    fa["grupo"] = pd.cut(fa["prob"], [-0.01, 0.2, 0.5, 0.8, 1.0], labels=["0-20 %", "20-50 %", "50-80 %", "80-100 %"])
    for gname, g in fa.groupby("grupo", observed=True):
        xi = sancionado_despues(g["fecha"], ini_sancion)
        print(f"  probabilidad {gname}: {len(g):,} personas; con sanción posterior {int(xi.sum())} ({xi.mean():.2%})")
