"""Paso 2: alertas que salen de las declaraciones juradas de los funcionarios (foto al 15/07/2026).

- Funcionario que le vende a su propia entidad: mismo nombre completo, durante su cargo. Los colaboradores y locadores
  que declaran no son alerta: es la planilla paralela, que se cuenta aparte.
- Doble percepción (Ley 28175, art. 3): funcionario con cargo en una entidad que cobra órdenes de otra durante su cargo,
  con nombre completo único en el país según el padrón de SUNAT.
- Posible familiar: persona con los mismos dos apellidos que un funcionario de la entidad que le compra, mientras el
  funcionario tiene el cargo. Cada caso lleva una probabilidad de parentesco (Bayes empírico):
    * azar esperado del par: lambda = funcionarios de la entidad en el año x frecuencia del par de apellidos en el
      departamento (proveedores del departamento, encogida hacia la frecuencia nacional del padrón);
    * probabilidad = r / (r + lambda), y r se calibra para que la suma de probabilidades iguale el exceso sobre el azar
      que mide la permutación (funcionarios repartidos entre entidades de su departamento, 100 veces por año).
Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 02_funcionarios.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, ANIOS, entidades, funcionarios, norm_serie  # noqa: E402

N_PERM = 100
M_PREVIO = 1000  # peso, en personas, de la frecuencia nacional al encoger la del departamento
DOCENCIA = r"UNIVERSIDAD|INSTITUTO|ESCUELA|COLEGIO|\bUGEL\b|EDUCACION"


def resumen(nombre, a):
    print(f"  {nombre}: {len(a):,} órdenes, {a['ruc'].nunique():,} personas, {a['ruc_entidad'].nunique():,} entidades, "
          f"S/ {a['monto'].sum() / 1e6:,.1f} millones; por año: {a.groupby('anio').size().to_dict()}")


o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "depto", "ruc", "nombre", "fecha", "anio", "monto"])
o = o[o["ruc"].str.startswith("10")].copy()
o["nombre"] = norm_serie(o["nombre"])
f = funcionarios()
ent = entidades().set_index("RUC")["NOMBRE_DE_ENTIDAD"].str.upper()
pn = pd.read_parquet(SALIDA / "padron_nombres.parquet").set_index("nombre")["n"]
o["nombre_unico"] = o["nombre"].map(pn).fillna(1).le(1)
print(f"órdenes a personas naturales: {len(o):,}; funcionarios-cargo con entidad y fechas: {len(f):,} "
      f"({f['persona'].nunique():,} personas, {f['ruc_entidad'].nunique():,} entidades); con cargo de decisión: {f['decide'].mean():.0%}")

# ---------------------------------------------------------------- 1) funcionario que le vende a su propia entidad
m = o.merge(f[["persona", "ruc_entidad", "cargo", "decide", "locador", "inicio", "cese"]],
            left_on=["nombre", "ruc_entidad"], right_on=["persona", "ruc_entidad"])
m = m[(m["fecha"] >= m["inicio"]) & (m["fecha"] <= m["cese"])]
m = m.sort_values(["decide", "locador"], ascending=[False, True]).drop_duplicates("id")
m["nivel"] = np.select([m["decide"], ~m["locador"]], ["cargo de decisión", "otro cargo"], "colaborador o locador")
propios = set(m["id"])
print("\nALERTA funcionario que le vende a su propia entidad, durante su cargo:")
for nivel, g in m.groupby("nivel"):
    resumen(nivel, g)
# alerta solo con cargo de decisión: con «otro cargo» muchos son personal contratado por servicios (control del 23/10)
a1 = m[m["nivel"] == "cargo de decisión"].assign(tarjeta="funcionario_propio", prob=1.0)
a1["evidencia"] = "es funcionario de la entidad (" + a1["cargo"] + ")"
a1[["id", "tarjeta", "prob", "decide", "cargo", "evidencia"]].to_parquet(SALIDA / "alerta_funcionario_propio.parquet", index=False)
m[m["nivel"] != "cargo de decisión"][["id", "nivel"]].to_parquet(SALIDA / "contexto_planilla_paralela.parquet", index=False)

# ---------------------------------------------------------------- 2) doble percepción
fn = f[~f["locador"]][["persona", "ruc_entidad", "cargo", "decide", "inicio", "cese"]].rename(columns={"ruc_entidad": "ent_cargo"})
m2 = o[o["nombre_unico"] & ~o["id"].isin(propios)].merge(fn, left_on="nombre", right_on="persona")
m2 = m2[(m2["fecha"] >= m2["inicio"]) & (m2["fecha"] <= m2["cese"]) & (m2["ent_cargo"] != m2["ruc_entidad"])]
# si la persona también declara en la entidad que le paga (cualquier cargo) en esa fecha, no es otra entidad
ya = o.merge(f[["persona", "ruc_entidad", "inicio", "cese"]], left_on=["nombre", "ruc_entidad"], right_on=["persona", "ruc_entidad"])
ya = set(ya.loc[(ya["fecha"] >= ya["inicio"]) & (ya["fecha"] <= ya["cese"]), "id"])
m2 = m2[~m2["id"].isin(ya)].sort_values("decide", ascending=False).drop_duplicates("id")
m2["docencia"] = m2["ruc_entidad"].map(ent).str.contains(DOCENCIA, regex=True, na=False)
print("\nALERTA doble percepción (cargo en una entidad, órdenes de otra, en las mismas fechas):")
for d, g in m2.groupby("docencia"):
    resumen("le paga una entidad educativa (posible docencia, excepción de ley)" if d else "le paga otra entidad", g)
a2 = m2.assign(tarjeta="doble_percepcion", prob=1.0)
a2["evidencia"] = "tiene cargo en otra entidad (" + a2["cargo"] + ", RUC " + a2["ent_cargo"] + ")"
a2[["id", "tarjeta", "prob", "decide", "docencia", "cargo", "ent_cargo", "evidencia"]].to_parquet(
    SALIDA / "alerta_doble_percepcion.parquet", index=False)

# ---------------------------------------------------------------- 3) posibles familiares, con probabilidad bayesiana
nac = pd.read_parquet(SALIDA / "padron_pares.parquet").groupby("par")["n"].sum()
N_NAC = int(pd.read_parquet(SALIDA / "padron_deptos.parquet")["personas"].sum())
personas_func = set(f["persona"])
cand = o[~o["nombre"].isin(personas_func)]
# frecuencia local: personas distintas que le venden a entidades del departamento, por par de apellidos
loc = cand[["depto", "ruc", "nombre"]].drop_duplicates(["depto", "ruc"])
toks = loc["nombre"].str.split()
filas = []
for k in (2, 3, 4):
    ok = toks.str.len() > k
    filas.append(pd.DataFrame({"depto": loc.loc[ok, "depto"], "par": toks[ok].str[:k].str.join(" ")}))
local = pd.concat(filas).groupby(["depto", "par"]).size()
n_local = loc.groupby("depto").size()

alertas, filas_anio = [], []
for y in ANIOS:
    ini, fin = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    s = f[(f["inicio"] <= fin) & (f["cese"] >= ini)].reset_index(drop=True)
    pv = cand[cand["anio"] == y][["ruc_entidad", "depto", "ruc", "nombre"]].drop_duplicates(["ruc_entidad", "ruc"]).reset_index(drop=True)
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

    hit = coinc(se)
    obs = np.unique(idx[hit]).size
    dep = s["depto"].fillna("SIN DEPTO").to_numpy()
    grupos = [np.flatnonzero(dep == x) for x in np.unique(dep)]
    perm = []
    for _ in range(N_PERM):
        a = se.copy()
        for g in grupos:
            a[g] = se[g][np.random.default_rng(len(perm) + 1000 * y).permutation(len(g))]
        perm.append(np.unique(idx[coinc(a)]).size)
    perm = np.array(perm)
    exceso = max(obs - perm.mean(), 0.0)

    # probabilidad bayesiana por persona-entidad coincidente
    h = pref[hit].merge(pv, left_on="i", right_index=True)
    h = h.sort_values("par", key=lambda x: x.str.len(), ascending=False).drop_duplicates("i")  # el par más largo que calza
    n_e = s.groupby("ruc_entidad").size()
    p_nac = h["par"].map(nac).fillna(1) / N_NAC
    n_loc = pd.Series([local.get((d, p), 0) for d, p in zip(h["depto"], h["par"])], index=h.index)
    p_loc = (n_loc + M_PREVIO * p_nac) / (h["depto"].map(n_local).fillna(0) + M_PREVIO)
    lam = (h["ruc_entidad"].map(n_e) * p_loc).clip(lower=1e-12).to_numpy()
    if exceso > 0:
        r = brentq(lambda r: (r / (r + lam)).sum() - exceso, 1e-12, 1e3)
        h["prob"] = r / (r + lam)
    else:
        r, h["prob"] = 0.0, 0.0
    filas_anio.append({"año": y, "funcionarios": len(s), "observados": obs, "azar_media": round(perm.mean(), 1),
                       "azar_p95": float(np.percentile(perm, 95)), "veces": round(obs / max(perm.mean(), 1), 2),
                       "p": round((1 + (perm >= obs).sum()) / (1 + N_PERM), 3), "prob_media": round(float(h["prob"].mean()), 3)})

    # órdenes de esas personas en esa entidad mientras algún funcionario con ese par tenía el cargo
    oo = o[(o["anio"] == y)][["id", "ruc_entidad", "ruc", "fecha"]].merge(h[["ruc_entidad", "ruc", "par", "prob"]], on=["ruc_entidad", "ruc"])
    oo = oo.merge(s[["ruc_entidad", "par", "inicio", "cese", "decide", "cargo"]], on=["ruc_entidad", "par"])
    oo = oo[(oo["fecha"] >= oo["inicio"]) & (oo["fecha"] <= oo["cese"])]
    oo = oo.sort_values("decide", ascending=False).drop_duplicates("id")
    alertas.append(oo)

print("\nPOSIBLES FAMILIARES (mismos dos apellidos que un funcionario de la entidad que les compra):")
print(pd.DataFrame(filas_anio).to_string(index=False))
a3 = pd.concat(alertas, ignore_index=True).merge(o[["id", "anio", "monto"]], on="id")
a3["tarjeta"] = "familiar"
a3["evidencia"] = "mismos apellidos que un funcionario de la entidad (" + a3["cargo"] + "); probabilidad de parentesco " + \
    (a3["prob"] * 100).round(0).astype(int).astype(str) + " %"
a3[["id", "tarjeta", "prob", "decide", "cargo", "evidencia"]].to_parquet(SALIDA / "alerta_familiar.parquet", index=False)
resumen("todas, dentro del cargo del funcionario", a3)
for umbral in (0.5, 0.8):
    resumen(f"con probabilidad de parentesco de {umbral:.0%} o más", a3[a3["prob"] >= umbral])
resumen("con funcionario de cargo de decisión", a3[a3["decide"]])
