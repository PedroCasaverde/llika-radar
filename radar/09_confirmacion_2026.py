"""Paso 9: confirmación con las órdenes de 2026 (enero a setiembre), que el radar no usó. Criterios en HALLAZGOS 44,
fijados antes de correr.

1. H1: personas con los dos apellidos de un funcionario de la entidad que les compra, frente a 100 permutaciones de
   funcionarios dentro de su departamento (mismo método que 02_funcionarios.py). 2025 se corre como control.
2. H2 fuera de tiempo: los 1 000 primeros proveedores por puntaje con las alertas de 2024-2025 (empate: mayor monto),
   ¿reciben desde el 01/01/2026 más sanciones del Tribunal que la base? Se excluye a quien ya tenía una sanción que
   empezó antes de 2026.
3. Órdenes de 2026 a sancionados durante su sanción, confirmadas por el registro publicado (descriptivo).
Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 09_confirmacion_2026.py
"""
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import PROC, SALIDA, funcionarios, norm_serie, wilson  # noqa: E402
from comun_sanciones import LLAVE, vigentes_publicadas  # noqa: E402

t0 = time.time()
UIT_2026 = 5500
N_PERM = 100
CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
SERVICIO_PUBLICO = (r"TELEFONICA|AMERICA MOVIL|\bCLARO\b|\bENTEL\b|VIETTEL|\bBITEL\b|NEXTEL|\bELECTRO|\bENEL\b|LUZ DEL SUR"
                    r"|HIDRANDINA|SEDAPAL|SEDAPAR|SEDALIB|SEDACUSCO|\bEPS\b|EMAPA|SERVICIO POSTAL|SERPOST")

# ---------------------------------------------------------------- órdenes de 2026, con los filtros de 01_base.py
cols = ["ruc_entidad", "departamento_entidad", "orden", "estadocontratacion", "tipodecontratacion",
        "monto_total_orden_original", "moneda", "ruc_contratista", "nombre_razon_contratista", "fecha_registro",
        "fecha_de_emision"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(PROC / "acc1_conosce" / "ordenes_2026-??.parquet")))],
              ignore_index=True)
d = d[d["fecha_de_emision"].dt.year.eq(2026)
      & d["tipodecontratacion"].astype(str).str.upper().str.startswith("CONTRATACIONES HASTA")
      & d["moneda"].eq("Nuevos Soles") & ~d["estadocontratacion"].eq("Anulada")
      & (d["monto_total_orden_original"] > 0) & (d["monto_total_orden_original"] <= 8 * UIT_2026)]
d = d.sort_values("fecha_registro").drop_duplicates(["ruc_entidad", "orden"], keep="last")
o26 = pd.DataFrame({"ruc_entidad": d["ruc_entidad"].astype(str), "depto": d["departamento_entidad"].astype(str),
                    "orden": d["orden"].astype(str),
                    "ruc": d["ruc_contratista"].astype(str), "nombre": d["nombre_razon_contratista"].astype(str),
                    "fecha": d["fecha_de_emision"].dt.normalize(), "monto": d["monto_total_orden_original"].astype(float)})
o26["anio"] = 2026
print(f"órdenes de 2026: {len(o26):,} (emitidas del {o26['fecha'].min().date()} al {o26['fecha'].max().date()}), "
      f"S/ {o26['monto'].sum() / 1e6:,.0f} millones; por mes: {o26.groupby(o26['fecha'].dt.month).size().to_dict()} "
      f"({time.time() - t0:.0f} s)", flush=True)

# ---------------------------------------------------------------- 1) familiares sobre el azar
f = funcionarios()
personas_func = set(f["persona"])


def exceso_familiares(op, y):
    """op: órdenes a personas naturales del año y, con nombre normalizado. Igual que el bucle de 02_funcionarios.py."""
    ini, fin = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    s = f[(f["inicio"] <= fin) & (f["cese"] >= ini)].reset_index(drop=True)
    cand = op[~op["nombre"].isin(personas_func)]
    pv = cand[["ruc_entidad", "depto", "ruc", "nombre"]].drop_duplicates(["ruc_entidad", "ruc"]).reset_index(drop=True)
    t = pv["nombre"].str.split()
    pref = pd.concat([pd.DataFrame({"i": pv.index[t.str.len() > k], "par": t[t.str.len() > k].str[:k].str.join(" ")})
                      for k in (2, 3, 4)])
    pares = set(s["par"])
    pref = pref[pref["par"].isin(pares)].drop_duplicates()
    cod_ent = {r: i for i, r in enumerate(sorted(set(s["ruc_entidad"]) | set(pv["ruc_entidad"])))}
    cod_par = {p: i for i, p in enumerate(sorted(pares))}
    B = np.int64(len(cod_par) + 1)
    llave = pv["ruc_entidad"].map(cod_ent).to_numpy(np.int64)[pref["i"].to_numpy()] * B + pref["par"].map(cod_par).to_numpy(np.int64)
    idx = pref["i"].to_numpy()
    sp, se = s["par"].map(cod_par).to_numpy(np.int64), s["ruc_entidad"].map(cod_ent).to_numpy(np.int64)

    def coinc(a):
        return np.isin(llave, np.unique(a * B + sp))

    obs = np.unique(idx[coinc(se)]).size
    dep = s["depto"].fillna("SIN DEPTO").to_numpy()
    grupos = [np.flatnonzero(dep == x) for x in np.unique(dep)]
    perm = []
    for _ in range(N_PERM):
        a = se.copy()
        for g in grupos:
            a[g] = se[g][np.random.default_rng(len(perm) + 1000 * y).permutation(len(g))]
        perm.append(np.unique(idx[coinc(a)]).size)
    perm = np.array(perm)
    return {"año": y, "funcionarios": len(s), "personas que venden": len(pv), "observados": obs,
            "azar_media": round(perm.mean(), 1), "azar_p95": float(np.percentile(perm, 95)),
            "veces": round(obs / max(perm.mean(), 1), 2), "p": round((1 + (perm >= obs).sum()) / (1 + N_PERM), 3),
            "cumple": obs > np.percentile(perm, 95)}


o25 = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["ruc_entidad", "depto", "ruc", "nombre", "anio"])
o25 = o25[(o25["anio"] == 2025) & o25["ruc"].str.startswith("10")].copy()
o25["nombre"] = norm_serie(o25["nombre"])
p26 = o26[o26["ruc"].str.startswith("10")].copy()
p26["nombre"] = norm_serie(p26["nombre"])
filas = [exceso_familiares(o25, 2025), exceso_familiares(p26, 2026)]
print("\n1) H1, FAMILIARES SOBRE EL AZAR (2025 es el control; debe dar lo mismo que 02_funcionarios.py)")
print(pd.DataFrame(filas).to_string(index=False), flush=True)
cargos_26 = f[(f["cese"] >= "2026-01-01")]
print(f"  cargos vigentes en 2026 según la foto de las declaraciones (al 15/07/2026): {len(cargos_26):,}", flush=True)

# ---------------------------------------------------------------- 2) anticipa sanciones, fuera de tiempo
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["ruc", "anio", "monto"])
al = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["ruc", "anio", "tarjeta", "puntos"])
sx = pd.read_parquet(SALIDA / "sanciones.parquet")
corte = pd.Timestamp("2026-01-01")
X = o[o["anio"].isin([2024, 2025])].groupby("ruc").agg(monto=("monto", "sum"))
piv = al[al["anio"].isin([2024, 2025])].pivot_table(index="ruc", columns="tarjeta", values="puntos", aggfunc="max")
for c in CARTAS:
    X[c] = piv[c].reindex(X.index).fillna(0) if c in piv else 0.0
X["puntos"] = X[CARTAS].sum(axis=1)
X = X[~X.index.isin(set(sx.loc[sx["inicio"] < corte, "ruc"]))]
X["y"] = X.index.isin(set(sx.loc[sx["inicio"] >= corte, "ruc"])).astype(int)
base = X["y"].mean()
print(f"\n2) H2 FUERA DE TIEMPO: alertas de 2024-2025, sanciones que empiezan del 01/01/2026 al "
      f"{sx['inicio'].max().date()} (última versión del registro: 03/10/2026)")
print(f"  base: {len(X):,} proveedores que vendieron en 2024-2025 sin sanción previa; {X['y'].sum():,} con sanción en 2026 ({base:.3%})")
top = X.sort_values(["puntos", "monto"], ascending=False).head(1000)
k = int(top["y"].sum())
lo, hi = wilson(k, len(top))
print(f"  1 000 primeros por puntaje: {k} con sanción en 2026 ({k / 1000:.1%}; IC {lo:.1%} a {hi:.1%}; "
      f"{k / 1000 / base:.1f} veces la base); cumple: {k / 1000 >= 3 * base and lo > base}")
top_m = X.sort_values("monto", ascending=False).head(1000)  # línea de base: lo que se hace hoy, mirar lo más caro
print(f"  comparación descriptiva, 1 000 primeros por monto vendido: {int(top_m['y'].sum())} con sanción en 2026 "
      f"({top_m['y'].mean():.1%}); proveedores en ambas listas: {len(top.index.intersection(top_m.index))}; "
      f"sancionados en 2026 que encuentra solo el radar: {int(top.loc[~top.index.isin(top_m.index), 'y'].sum())}")
for nombre, g in [("cualquier alerta", X[X["puntos"] > 0])] + [(c, X[X[c] > 0]) for c in CARTAS]:
    kk = int(g["y"].sum())
    lo, hi = wilson(kk, len(g))
    print(f"  {nombre:<20}{len(g):>8,} proveedores; {kk:>4} con sanción en 2026 ({g['y'].mean():.2%}; IC {lo:.2%} a {hi:.2%}; "
          f"{g['y'].mean() / base:.1f} veces la base)")
for tipo, pref in [("personas", "10"), ("empresas", "20")]:
    g = X[X.index.str.startswith(pref)]
    print(f"  base de {tipo}: {g['y'].mean():.3%}; con alguna alerta: {g.loc[g['puntos'] > 0, 'y'].mean():.2%}")

# ---------------------------------------------------------------- 3) sancionados que venden en 2026
m = o26.merge(sx[LLAVE + ["definitiva"]], on="ruc")
m = vigentes_publicadas(m).drop_duplicates(["ruc_entidad", "orden"])
m["servicio_publico"] = m["nombre"].str.upper().str.contains(SERVICIO_PUBLICO, regex=True, na=False)
print("\n3) SANCIONADOS QUE VENDEN EN 2026, CONFIRMADOS POR EL REGISTRO PUBLICADO")
for sp, g in m.groupby("servicio_publico"):
    print(f"  {'servicio público' if sp else 'resto'}: {len(g):,} órdenes, {g['ruc'].nunique():,} proveedores, "
          f"{g['ruc_entidad'].nunique():,} entidades, S/ {g['monto'].sum() / 1e6:,.2f} millones")
print(f"  (las órdenes de setiembre solo se confirman si una versión de las 6 semanas siguientes las cubre) "
      f"[{time.time() - t0:.0f} s]")
