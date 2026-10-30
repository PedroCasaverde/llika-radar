"""Paso 18: el costo de no usar LLIKA, en tiempo y dinero (HALLAZGOS 56).

Parte A, el costo del error con el examen del E5 (alertas de 2022-2023 frente a sanciones desde 2024). Usa las tablas
del taller, que son el mismo examen sin RUC (942 164 proveedores, 1 996 con sanción después), y repite el v0 del E5
con su receta congelada (regresión L1, C = 0,01). Para las tres listas (lo más caro primero, regla del radar y v0) mide:
- la precisión en distintos K y el rendimiento marginal por tramos de revisiones;
- cuántas revisiones hacen falta para encontrar los mismos aciertos que el v0 encuentra en 1 000;
- lo que cuestan en horas esas revisiones, para tres supuestos de minutos por revisión;
- el costo del error en unidades del negocio: revisiones sin acierto (falsos positivos) y sancionados que la lista no
  trajo (falsos negativos), con el monto que vendieron.
Todo con el mismo bootstrap pareado del E5 (pesos de Poisson, 1 000 repeticiones, semilla 11).

Parte B: cuánto dinero pasa por órdenes con alerta, año por año, cómo se concentra por entidad, y cuánto trabajo es
para un OCI revisar sus casos frente a cruzar a mano a todos sus proveedores. Parte C: lo que valen esas horas en soles,
con sueldos oficiales de auditores. Parte D: lo que paga el Estado en órdenes con alerta, por capa de certeza legal.
Parte E: el dinero en contratos con alerta que cada lista pone delante del auditor con el mismo presupuesto.

Las partes B, D y E necesitan los datos completos del radar. Con ellos, --exportar guarda sus agregados sin RUC ni DNI
en la carpeta de las tablas del taller (costos_*.parquet y costos_resumen.json, con huella sha256); sin ellos, como en
el repositorio público, el script lee esos agregados y comprueba su huella.

Un acierto es un proveedor con una sanción posterior del Tribunal: no es una irregularidad confirmada. Los minutos por
revisión y por cruce son supuestos hasta la prueba de uso (14/11).

    python radar/18_costo_de_no_usar.py [carpeta con las tablas del taller] [--exportar]

Guarda los agregados en datos/procesados/radar/costo_de_no_usar.json.
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

t0 = time.time()
K, B, SEMILLA, LARGO = 1000, 1000, 11, 15_000
KS = [100, 250, 500, 1000, 2000, 3000, 5000, 10000]
TRAMOS = [0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000]
CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
ANTEC = ["log_multas", "log_penalidades", "evasion", "reciente"]
C_V0 = 0.01
# Supuestos, a medir en la prueba de uso
MIN_REVISION = (15, 30, 60)      # minutos que toma revisar un caso (abrirlo, mirar sus fuentes y decidir)
MIN_CRUCE = 20                   # minutos para cruzar a mano un proveedor en cuatro fuentes (5 por fuente)
# horas trabajadas al año en el sector público: 8 h de lunes a viernes. En 2026 hay 261 días hábiles, 13 feriados
# nacionales caen en día hábil y las vacaciones (30 días calendario) son 22 días hábiles: 226 días de 8 horas
HORAS_AUDITOR = 226 * 8
COSTO_ANUAL_USD = 400            # servidor encendido todo el año, unos USD 33 al mes (precios del 23/10, por confirmar)
CAMBIO = 4.0                     # soles por dólar, supuesto


def es(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def pct(x, d=1):
    return es(100 * x, d) + " %"


# las tablas del taller: en el repositorio están en taller/datos; en el curso, en el clon local del repositorio
_CANDIDATAS = [Path(__file__).resolve().parents[1] / "taller" / "datos", Path(r"D:\UTEC\llika-radar\taller\datos")]
ap = argparse.ArgumentParser(description="El costo de no usar LLIKA, en tiempo y dinero")
ap.add_argument("carpeta", nargs="?", help="carpeta con las tablas del taller (por defecto, taller/datos)")
ap.add_argument("--exportar", action="store_true",
                help="con los datos completos, guarda en esa carpeta los agregados sin RUC de las partes B, D y E")
ARGS = ap.parse_args()
TALLER = Path(ARGS.carpeta) if ARGS.carpeta else next(
    (c for c in _CANDIDATAS if (c / "examen.parquet").exists()), _CANDIDATAS[0])
tr = pd.read_parquet(TALLER / "aprendizaje.parquet")
te = pd.read_parquet(TALLER / "examen.parquet")
pares = pd.read_parquet(TALLER / "examen_entidades.parquet")
y = te["y"].to_numpy()
monto = te["monto"].to_numpy(float)
P = int(y.sum())
print(f"examen: {es(len(te))} proveedores, {es(P)} con sanción después, monto de esos {es(monto[y == 1].sum() / 1e6, 1)} millones")


def orden(*claves):
    return np.lexsort(tuple(-np.asarray(c, dtype=float) for c in reversed(claves)))


# --- las tres listas, como en el taller ---
ytr = tr["y"].to_numpy()
rng = np.random.default_rng(7)
neg = np.flatnonzero(ytr == 0)
fil = np.sort(np.concatenate([np.flatnonzero(ytr == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))
cols = CARTAS + ANTEC
sc = StandardScaler().fit(tr.iloc[fil][cols])
m = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, C=C_V0)
m.fit(sc.transform(tr.iloc[fil][cols]), ytr[fil])
puntaje = m.predict_proba(sc.transform(te[cols]))[:, 1]
listas = {"hoy": orden(monto), "regla": orden(te[CARTAS].sum(axis=1), monto), "v0": orden(puntaje, monto)}
NOM = list(listas)
esperado = {"hoy": 37, "regla": 71, "v0": 143}
for n in NOM:
    h = int(y[listas[n][:K]].sum())
    assert h == esperado[n], f"{n}: {h} aciertos en {K}, el E5 da {esperado[n]}"
print("control: 37, 71 y 143 aciertos en las 1 000 primeras, igual que el E5\n")

prof = {n: idx[:LARGO] for n, idx in listas.items()}
yp = {n: y[prof[n]].astype(float) for n in NOM}
mp = {n: monto[prof[n]] for n in NOM}
U1000 = np.union1d(listas["hoy"][:K], listas["v0"][:K])
M500 = np.union1d(listas["hoy"][:K // 2], listas["v0"][:K // 2])


def medir(w):
    """Todas las métricas para un vector de pesos (unos, o un sorteo del bootstrap)."""
    wy, wm = float((w * y).sum()), float((w * y * monto).sum())
    c = {}
    for n in NOM:
        ww = w[prof[n]]
        c[n] = (np.cumsum(ww), np.cumsum(ww * yp[n]), np.cumsum(ww * yp[n] * mp[n]))
    r = {}
    objetivo = {}
    for n in NOM:
        cw, ch, cm = c[n]
        js = np.searchsorted(cw, KS)
        r[f"prec_{n}"] = ch[js] / cw[js]
        r[f"hall_{n}"] = ch[js]
        r[f"monto_{n}"] = cm[js]
        r[f"recall_{n}"] = ch[js] / wy
        r[f"recall_monto_{n}"] = cm[js] / wm
        jt = np.searchsorted(cw, TRAMOS[1:])
        h_e, r_e = np.concatenate([[0.0], ch[jt]]), np.concatenate([[0.0], cw[jt]])
        r[f"tramo_{n}"] = np.diff(h_e) / np.diff(r_e)
        j1 = int(np.searchsorted(cw, K))
        objetivo[n] = ch[j1]
    for etiqueta, T in (("v0", objetivo["v0"]), ("hoy", objetivo["hoy"])):
        for n in NOM:
            cw, ch, _ = c[n]
            j = int(np.searchsorted(ch, T))
            r[f"revs_para_{etiqueta}_{n}"] = cw[j] if j < len(ch) else np.nan
    for etiqueta, U in (("u1000", U1000), ("m500", M500)):
        u = w[U]
        r[f"{etiqueta}_revs"], r[f"{etiqueta}_hall"] = float(u.sum()), float((u * y[U]).sum())
        r[f"{etiqueta}_monto"] = float((u * y[U] * monto[U]).sum())
    return r


punto = medir(np.ones(len(y)))
rng = np.random.default_rng(SEMILLA)
boot = {k: [] for k in punto}
for _ in range(B):
    w = rng.poisson(1.0, size=len(y)).astype(float)
    for k, v in medir(w).items():
        boot[k].append(v)
boot = {k: np.asarray(v) for k, v in boot.items()}
print(f"bootstrap de {B} repeticiones en {time.time() - t0:.0f} s")
sin_alcanzar = {k: float(np.isnan(v).mean()) for k, v in boot.items() if k.startswith("revs_para") and np.isnan(v).any()}
if sin_alcanzar:
    print("OJO, repeticiones donde la lista no llegó al objetivo dentro de", LARGO, sin_alcanzar)


def rango(k, i=None):
    a = boot[k] if i is None else boot[k][:, i]
    p = punto[k] if i is None else punto[k][i]
    return [float(p), float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


R = {"supuestos": {"min_revision": list(MIN_REVISION), "min_cruce": MIN_CRUCE, "horas_auditor_anio": HORAS_AUDITOR,
                   "costo_anual_usd": COSTO_ANUAL_USD, "cambio": CAMBIO, "K": K, "B": B, "semilla": SEMILLA},
     "examen": {"proveedores": len(te), "sancionados": P, "monto_sancionados": float(monto[y == 1].sum())}}

# 1. precisión en distintos K
print("\n1. Precisión en los K primeros (rango 95 %)")
R["precision"] = {}
for n in NOM:
    R["precision"][n] = {str(k): rango(f"prec_{n}", i) for i, k in enumerate(KS)}
print(f"{'K':>7} " + " ".join(f"{n:>26}" for n in NOM))
for i, k in enumerate(KS):
    print(f"{k:>7} " + " ".join(
        f"{pct(rango(f'prec_{n}', i)[0]):>9} [{pct(rango(f'prec_{n}', i)[1])} a {pct(rango(f'prec_{n}', i)[2])}]".rjust(26)
        for n in NOM))

# 2. rendimiento marginal por tramos
print("\n2. Hallazgos por cada 100 revisiones, por tramo de la lista (la lista por monto frente al v0)")
R["tramos"] = {"bordes": TRAMOS, "lista": {}}
for n in NOM:
    R["tramos"]["lista"][n] = [rango(f"tramo_{n}", t) for t in range(len(TRAMOS) - 1)]
R["tramos"]["p_v0_mayor_que_hoy"] = [float((boot["tramo_v0"][:, t] > boot["tramo_hoy"][:, t]).mean()) for t in range(len(TRAMOS) - 1)]
for t in range(len(TRAMOS) - 1):
    a, b_ = TRAMOS[t], TRAMOS[t + 1]
    print(f"  {a:>5} a {b_:<5} hoy {es(100 * punto['tramo_hoy'][t], 1):>5}  regla {es(100 * punto['tramo_regla'][t], 1):>5}  "
          f"v0 {es(100 * punto['tramo_v0'][t], 1):>5}   (el v0 rinde más en {pct(R['tramos']['p_v0_mayor_que_hoy'][t], 0)} de las repeticiones)")
cruce = next((TRAMOS[t] for t in range(len(TRAMOS) - 1) if punto["tramo_v0"][t] < punto["tramo_hoy"][t]), None)
R["tramos"]["primer_tramo_donde_hoy_rinde_mas"] = cruce
print("  primer tramo donde ordenar por monto rinde más que el v0 empieza en:", cruce)
print("  acumulado: aciertos (y millones de soles que vendieron en 2022-2023) en los K primeros")
R["acumulado"] = {}
for i, k in enumerate(KS):
    p_ = float((boot["hall_v0"][:, i] > boot["hall_hoy"][:, i]).mean())
    R["acumulado"][str(k)] = {n: {"aciertos": rango(f"hall_{n}", i), "monto": rango(f"monto_{n}", i)} for n in NOM}
    R["acumulado"][str(k)]["p_v0_adelante"] = p_
    print(f"    K={k:>6}: " + "  ".join(f"{n} {es(punto[f'hall_{n}'][i])} (S/ {es(punto[f'monto_{n}'][i] / 1e6, 1)} M)" for n in NOM)
          + f"  | el v0 va adelante en {pct(p_, 0)} de las repeticiones")

# 3. revisiones para igualar
print("\n3. Revisiones necesarias para encontrar los mismos aciertos")
R["igualar"] = {}
for etiqueta, nombre in (("v0", "los que el v0 encuentra en 1 000"), ("hoy", "los que hoy encuentra la lista por monto en 1 000")):
    print(f"  Para llegar a {nombre}:")
    R["igualar"][etiqueta] = {"aciertos": float(punto[f"hall_{etiqueta}"][KS.index(K)]), "revisiones": {}}
    for n in NOM:
        rr = rango(f"revs_para_{etiqueta}_{n}")
        R["igualar"][etiqueta]["revisiones"][n] = rr
        print(f"    {n:<6} {es(rr[0]):>7} revisiones  [{es(rr[1])} a {es(rr[2])}]")
razon = boot["revs_para_v0_hoy"] / boot["revs_para_v0_v0"]
R["igualar"]["razon_hoy_sobre_v0"] = [float(punto["revs_para_v0_hoy"] / punto["revs_para_v0_v0"]),
                                      float(np.nanpercentile(razon, 2.5)), float(np.nanpercentile(razon, 97.5))]
print(f"  la lista por monto necesita {es(R['igualar']['razon_hoy_sobre_v0'][0], 1)} veces las revisiones del v0 "
      f"[{es(R['igualar']['razon_hoy_sobre_v0'][1], 1)} a {es(R['igualar']['razon_hoy_sobre_v0'][2], 1)}]")
i1 = KS.index(K)
rph = {n: K / punto[f"hall_{n}"][i1] for n in NOM}
rz = boot["prec_v0"][:, i1] / boot["prec_hoy"][:, i1]
R["razon_precision_1000"] = [float(punto["prec_v0"][i1] / punto["prec_hoy"][i1]), float(np.percentile(rz, 2.5)), float(np.percentile(rz, 97.5))]
print(f"  precisión del v0 sobre la de hoy en las 1 000 primeras: {es(R['razon_precision_1000'][0], 1)} veces "
      f"[{es(R['razon_precision_1000'][1], 1)} a {es(R['razon_precision_1000'][2], 1)}]")
R["revisiones_por_acierto_1000"] = {n: float(v) for n, v in rph.items()}
print("  revisiones por acierto en las 1 000 primeras:", {n: es(v, 1) for n, v in rph.items()})

# 4. horas
print("\n4. Horas, según los minutos por revisión (supuesto)")
R["horas"] = {}
rv0, rhoy = R["igualar"]["v0"]["revisiones"], None
for mm in MIN_REVISION:
    f = mm / 60
    fila = {"por_acierto_hoy": rph["hoy"] * f, "por_acierto_regla": rph["regla"] * f, "por_acierto_v0": rph["v0"] * f,
            "para_objetivo_hoy": [x * f for x in rv0["hoy"]], "para_objetivo_v0": [x * f for x in rv0["v0"]]}
    fila["diferencia"] = [(rv0["hoy"][0] - rv0["v0"][0]) * f,
                          float(np.nanpercentile((boot["revs_para_v0_hoy"] - boot["revs_para_v0_v0"]) * f, 2.5)),
                          float(np.nanpercentile((boot["revs_para_v0_hoy"] - boot["revs_para_v0_v0"]) * f, 97.5))]
    fila["auditores_anio"] = fila["diferencia"][0] / HORAS_AUDITOR
    R["horas"][str(mm)] = fila
    print(f"  {mm} min: por acierto hoy {es(fila['por_acierto_hoy'], 1)} h, regla {es(fila['por_acierto_regla'], 1)} h, "
          f"v0 {es(fila['por_acierto_v0'], 1)} h | para el objetivo del v0: hoy {es(fila['para_objetivo_hoy'][0])} h, "
          f"v0 {es(fila['para_objetivo_v0'][0])} h | diferencia {es(fila['diferencia'][0])} h "
          f"[{es(fila['diferencia'][1])} a {es(fila['diferencia'][2])}] = {es(fila['auditores_anio'], 2)} auditores-año")

# 5. costo del error en unidades del negocio
print("\n5. El costo del error en las 1 000 primeras revisiones")
R["error"] = {}
vend = monto[y == 1]
R["monto_por_sancionado"] = {"media": float(vend.mean()), "mediana": float(np.median(vend)), "p90": float(np.percentile(vend, 90))}
print(f"  un sancionado que no entra a la lista vendió en 2022-2023, en promedio, S/ {es(vend.mean() / 1e3, 0)} mil "
      f"(mediana S/ {es(np.median(vend) / 1e3, 0)} mil, p90 S/ {es(np.percentile(vend, 90) / 1e3, 0)} mil)")
for n in NOM:
    h, hm = punto[f"hall_{n}"][i1], punto[f"monto_{n}"][i1]
    rec, recm = rango(f"recall_{n}", i1), rango(f"recall_monto_{n}", i1)
    R["error"][n] = {"aciertos": rango(f"hall_{n}", i1), "falsos_positivos": K - h, "falsos_negativos": P - h,
                     "monto_aciertos": rango(f"monto_{n}", i1), "monto_perdido": float(monto[y == 1].sum() - hm),
                     "recall": rec, "recall_monto": recm}
    print(f"  {n:<6} aciertos {es(h)} [{es(R['error'][n]['aciertos'][1])} a {es(R['error'][n]['aciertos'][2])}] | "
          f"revisiones sin acierto {es(K - h)} | sancionados que no trajo {es(P - h)} ({pct(1 - rec[0], 1)}) | "
          f"vendieron S/ {es(hm / 1e6, 1)} M de S/ {es(monto[y == 1].sum() / 1e6, 1)} M ({pct(recm[0], 1)} del monto de los sancionados)")
for nombre, e in (("unión de las dos listas (1 000 de cada una)", "u1000"), ("500 del v0 y 500 por monto", "m500")):
    rr, hh, mo = rango(f"{e}_revs"), rango(f"{e}_hall"), rango(f"{e}_monto")
    R["error"][e] = {"revisiones": rr, "aciertos": hh, "monto_aciertos": mo}
    print(f"  {nombre}: {es(rr[0])} proveedores, {es(hh[0])} aciertos [{es(hh[1])} a {es(hh[2])}] ({pct(hh[0] / rr[0])}), "
          f"S/ {es(mo[0] / 1e6, 1)} M")
print("  Cociente entre lo que vale un acierto perdido y una revisión sin acierto que justifica parar donde el rendimiento marginal es p:")
R["cociente_costo"] = {}
for t in range(len(TRAMOS) - 1):
    p_ = punto["tramo_v0"][t]
    if p_ > 0:
        R["cociente_costo"][f"{TRAMOS[t]}-{TRAMOS[t + 1]}"] = float((1 - p_) / p_)
print("   ", {k: es(v, 0) for k, v in R["cociente_costo"].items()})

# 6. segmentos de la lista del v0 por motivo de sanción
if "motivo" in te.columns:
    print("\n6. Revisiones por acierto, según el motivo de la sanción (en las 1 000 primeras)")
    R["motivos"] = {}
    for n in NOM:
        mo = te.iloc[listas[n][:K]].query("y == 1")["motivo"].value_counts().to_dict()
        R["motivos"][n] = {k: int(v) for k, v in mo.items()}
        print(f"  {n:<6}", {k: (v, es(K / v, 0) + " rev/acierto") for k, v in mo.items()})

# --- Agregados de los datos completos (partes B, D y E) ---
# Con los datos completos del radar se calculan aquí. Con --exportar se guardan, sin RUC ni DNI, en la carpeta de las
# tablas del taller (costos_*.parquet y costos_resumen.json, con su huella sha256): así el repositorio público corre las
# partes B, D y E sin los datos completos.
CAPA = {"sancionado": 1, "funcionario_propio": 1, "doble_percepcion": 1, "empresa_espejo": 2, "mismo_dueno": 3, "familiar": 3}
INCIDENCIA_CGR = 0.127
TABLAS_COSTOS = ("anio", "tarjetas", "entidades", "entidad_anio", "proveedores")


def agregados_completos():
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto"])
    al = pd.read_parquet(SALIDA / "radar_alertas.parquet",
                         columns=["id", "tarjeta", "gravedad", "prob", "anio", "monto", "ruc_entidad", "ruc"])
    casos = pd.read_parquet(SALIDA / "radar_casos.parquet", columns=["ruc_entidad", "ruc", "anio"])
    s_ = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio"])
    al["capa"] = al["tarjeta"].map(CAPA)
    # una fila por orden con alerta, en su capa más cierta
    uni = al.sort_values("capa").groupby("id").agg(anio=("anio", "first"), monto=("monto", "first"),
                                                   ent=("ruc_entidad", "first"), ruc=("ruc", "first"),
                                                   capa=("capa", "min"), grav=("gravedad", "max"))

    def por_capa(df, f, indice="anio"):
        return df.pivot_table(index=indice, columns="capa", values="monto", aggfunc=f).reindex(columns=[1, 2, 3]).fillna(0)

    anio = o.groupby("anio").agg(ordenes_total=("id", "size"), monto_total=("monto", "sum"))
    anio = anio.join(uni.groupby("anio").agg(ordenes_alerta=("monto", "size"), monto_alerta=("monto", "sum")))
    anio = anio.join(por_capa(uni, "sum").rename(columns=lambda c: f"capa{c}"))
    anio = anio.join(por_capa(uni, "count").rename(columns=lambda c: f"ordenes_capa{c}"))
    anio["casos"] = casos.groupby("anio").size()
    anio = anio.fillna(0).reset_index()
    tarjetas = (al.drop_duplicates(["id", "tarjeta"]).groupby("tarjeta").agg(ordenes=("id", "size"), monto=("monto", "sum"))
                .reset_index())
    entidades = uni.groupby("ent")["monto"].sum().sort_values(ascending=False).reset_index(drop=True).to_frame("monto_alerta")
    eny = o.groupby(["ruc_entidad", "anio"]).agg(ordenes=("id", "size"), proveedores=("ruc", "nunique"))
    eny = eny.join(casos.groupby(["ruc_entidad", "anio"]).size().rename("casos"), how="left").fillna({"casos": 0})
    entidad_anio = eny.reset_index()[["anio", "ordenes", "proveedores", "casos"]]
    # proveedores del examen, en el orden de las tablas del taller: RUC ordenados, sin los sancionados antes de 2024
    X = o[(o["anio"] >= 2022) & (o["anio"] <= 2023)].groupby("ruc").agg(monto=("monto", "sum"))
    X = X[~X.index.isin(set(s_.loc[s_["inicio"] < pd.Timestamp("2024-01-01"), "ruc"]))]
    mal = por_capa(uni[(uni["anio"] >= 2022) & (uni["anio"] <= 2023)], "sum", indice="ruc").reindex(X.index).fillna(0)
    proveedores = pd.DataFrame({"monto": X["monto"].to_numpy(), **{f"capa{c}": mal[c].to_numpy() for c in (1, 2, 3)}})
    fam = al[al["tarjeta"] == "familiar"].groupby("id").agg(monto=("monto", "first"), prob=("prob", "max"))
    g3 = uni[uni["grav"] >= 3]
    resumen = {"entidades_total": int(o["ruc_entidad"].nunique()),
               "gravedad_3": {"ordenes": int(len(g3)), "monto": float(g3["monto"].sum())},
               "familiar": {"bruto": float(fam["monto"].sum()), "ponderado": float((fam["monto"] * fam["prob"]).sum())}}
    try:
        ej = json.loads((SALIDA / "evaluacion_e5.json").read_text(encoding="utf-8"))["confirmacion_2026"]
        resumen["confirmacion_2026"] = {"hoy": ej["filas"]["hoy"]["precision"], "v0": ej["filas"]["v0"]["precision"],
                                        "dif": ej["diferencias"]["v0-hoy"]}
    except (OSError, KeyError):
        pass
    agg = {"anio": anio, "tarjetas": tarjetas, "entidades": entidades, "entidad_anio": entidad_anio,
           "proveedores": proveedores}
    return agg, resumen


def exportar(agg, resumen, carpeta):
    archivos = {}
    for nombre in TABLAS_COSTOS:
        p = carpeta / f"costos_{nombre}.parquet"
        agg[nombre].to_parquet(p, index=False)
        archivos[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "MB": round(p.stat().st_size / 1e6, 2)}
    doc = {"nota": "Agregados sin RUC ni DNI para las partes B, D y E de radar/18_costo_de_no_usar.py. Las filas de "
                   "costos_proveedores siguen el orden de examen.parquet.", **resumen, "archivos": archivos}
    (carpeta / "costos_resumen.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nagregados exportados en {carpeta}: " + ", ".join(f"{k} ({v['MB']} MB)" for k, v in archivos.items()))


def agregados_taller(carpeta):
    doc = json.loads((carpeta / "costos_resumen.json").read_text(encoding="utf-8"))
    agg = {}
    for nombre in TABLAS_COSTOS:
        p = carpeta / f"costos_{nombre}.parquet"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == doc["archivos"][p.name]["sha256"], f"la huella de {p.name} no coincide"
        agg[nombre] = pd.read_parquet(p)
    return agg, doc


COMPLETOS = (SALIDA / "ordenes.parquet").exists() and (SALIDA / "radar_alertas.parquet").exists()
AGG, RES = None, {}
if COMPLETOS:
    AGG, RES = agregados_completos()
    if ARGS.exportar:
        exportar(AGG, RES, TALLER)
elif (TALLER / "costos_resumen.json").exists():
    AGG, RES = agregados_taller(TALLER)
    print("\nlas partes B, D y E usan los agregados del taller (sin RUC), con su huella comprobada")

# confirmación con 2026, ya calculada en el paso 12
if "confirmacion_2026" in RES:
    c26 = RES["confirmacion_2026"]
    R["confirmacion_2026"] = {"hoy": c26["hoy"], "v0": c26["v0"], "dif": c26["dif"],
                              "revisiones_por_acierto_hoy": 1 / c26["hoy"], "revisiones_por_acierto_v0": 1 / c26["v0"]}
    print(f"\n7. Confirmación con 2026 (paso 12): hoy {pct(c26['hoy'])}, v0 {pct(c26['v0'])}; "
          f"{es(1 / c26['hoy'])} y {es(1 / c26['v0'])} revisiones por acierto; diferencia {es(100 * c26['dif']['dif'], 1)} p.p. "
          f"[{es(100 * c26['dif']['rango'][0], 1)} a {es(100 * c26['dif']['rango'][1], 1)}]")

# --- Parte B: el dinero con alerta, las entidades y el trabajo de un OCI ---
if AGG is not None:
    print("\n--- Parte B: el dinero con alerta, las entidades y el trabajo de un OCI ---")
    A = AGG["anio"].set_index("anio")
    ex = A[["ordenes_alerta", "monto_alerta"]].rename(columns={"ordenes_alerta": "ordenes", "monto_alerta": "monto"})
    ex["ordenes"] = ex["ordenes"].astype(int)
    ex["pct_ordenes"] = ex["ordenes"] / A["ordenes_total"]
    ex["pct_monto"] = ex["monto"] / A["monto_total"]
    n_al, m_al = int(A["ordenes_alerta"].sum()), float(A["monto_alerta"].sum())
    n_o, m_o = int(A["ordenes_total"].sum()), float(A["monto_total"].sum())
    R["exposicion"] = {
        "ordenes": n_al, "monto": m_al, "ordenes_total": n_o, "monto_total": m_o,
        "por_anio": {int(a): {"ordenes": int(r_.ordenes), "monto": float(r_.monto), "pct_monto": float(r_.pct_monto),
                              "monto_total_anio": float(A.loc[a, "monto_total"])} for a, r_ in ex.iterrows()}}
    anios = len(ex)
    R["exposicion"]["monto_por_anio"] = m_al / anios
    print(f"órdenes con alerta: {es(n_al)} de {es(n_o)} ({pct(n_al / n_o, 2)}), S/ {es(m_al / 1e6, 0)} M de "
          f"S/ {es(m_o / 1e6, 0)} M ({pct(m_al / m_o, 2)}); S/ {es(m_al / anios / 1e6, 0)} M por año")
    print(ex.assign(monto=(ex["monto"] / 1e6).round(1), pct_ordenes=(100 * ex["pct_ordenes"]).round(2),
                    pct_monto=(100 * ex["pct_monto"]).round(2)).to_string())
    g3 = RES["gravedad_3"]
    R["exposicion"]["gravedad_3"] = {"ordenes": int(g3["ordenes"]), "monto": float(g3["monto"])}
    print(f"con gravedad legal 3 (impedimento directo): {es(g3['ordenes'])} órdenes, S/ {es(g3['monto'] / 1e6, 1)} M")
    ta = AGG["tarjetas"].set_index("tarjeta")
    R["exposicion"]["por_tarjeta"] = {k: {"ordenes": int(r_.ordenes), "monto": float(r_.monto)} for k, r_ in ta.iterrows()}
    print((ta.assign(monto=(ta["monto"] / 1e6).round(1))).to_string())
    pe = AGG["entidades"]["monto_alerta"]
    n_ent = int(RES["entidades_total"])
    R["exposicion"]["entidades_con_alerta"] = int(len(pe))
    R["exposicion"]["entidades_total"] = n_ent
    cum = pe.cumsum() / pe.sum()
    R["exposicion"]["concentracion"] = {"top_10": float(cum.iloc[9]), "top_100": float(cum.iloc[99]),
                                        "top_10_pct_entidades": float(cum.iloc[max(int(0.1 * len(pe)) - 1, 0)])}
    print(f"entidades con alguna alerta: {es(len(pe))} de {es(n_ent)}; las 10 primeras concentran {pct(cum.iloc[9], 0)} del monto "
          f"con alerta, las 100 primeras {pct(cum.iloc[99], 0)}, el 10 % de las entidades {pct(R['exposicion']['concentracion']['top_10_pct_entidades'], 0)}")

    # trabajo de un OCI: casos de su entidad frente a cruzar a mano a todos sus proveedores
    d = AGG["entidad_anio"]
    n_casos = int(A["casos"].sum())
    q = lambda s: {"mediana": float(s.median()), "p75": float(s.quantile(.75)), "p90": float(s.quantile(.90)), "p99": float(s.quantile(.99)), "max": float(s.max())}
    R["oci"] = {"entidad_anio": int(len(d)), "ordenes": q(d["ordenes"]), "proveedores": q(d["proveedores"]),
                "con_casos": int((d["casos"] > 0).sum()), "casos": q(d.loc[d["casos"] > 0, "casos"])}
    print(f"\nentidad-año: {es(len(d))}; órdenes por entidad-año: mediana {es(R['oci']['ordenes']['mediana'])}, p90 {es(R['oci']['ordenes']['p90'])}")
    print(f"proveedores distintos por entidad-año: mediana {es(R['oci']['proveedores']['mediana'])}, p90 {es(R['oci']['proveedores']['p90'])}, p99 {es(R['oci']['proveedores']['p99'])}")
    print(f"entidad-años con al menos un caso: {es(R['oci']['con_casos'])} ({pct(R['oci']['con_casos'] / len(d), 0)}); casos por entidad-año con casos: "
          f"mediana {es(R['oci']['casos']['mediana'])}, p90 {es(R['oci']['casos']['p90'])}, máx {es(R['oci']['casos']['max'])}")
    R["oci"]["horas"] = {}
    for mm in MIN_REVISION:
        h_casos = d.loc[d["casos"] > 0, "casos"] * mm / 60
        h_mano = d["proveedores"] * MIN_CRUCE / 60
        R["oci"]["horas"][str(mm)] = {"casos_mediana": float(h_casos.median()), "casos_p90": float(h_casos.quantile(.9)),
                                      "mano_mediana": float(h_mano.median()), "mano_p90": float(h_mano.quantile(.9))}
        print(f"  {mm} min por caso: revisar los casos de LLIKA toma {es(h_casos.median(), 1)} h (mediana) y {es(h_casos.quantile(.9), 0)} h (p90) por entidad-año; "
              f"cruzar a mano a todos los proveedores a {MIN_CRUCE} min toma {es(h_mano.median(), 0)} h y {es(h_mano.quantile(.9), 0)} h")
    n_anios = int(d["anio"].nunique())
    mano_total = float(d["proveedores"].sum()) * MIN_CRUCE / 60
    R["oci"]["nacional"] = {"anios": n_anios, "pares_proveedor_entidad_anio": int(d["proveedores"].sum()),
                            "cruce_a_mano_horas_por_anio": mano_total / n_anios,
                            "cruce_a_mano_auditores_anio": mano_total / n_anios / HORAS_AUDITOR, "casos": n_casos}
    for mm in MIN_REVISION:
        R["oci"]["nacional"][f"casos_{mm}_auditores_anio"] = n_casos * mm / 60 / n_anios / HORAS_AUDITOR
    print(f"\ntodo el país, por año: cruzar a mano a todos los proveedores de todas las entidades ({es(d['proveedores'].sum() / n_anios)} pares "
          f"proveedor-entidad) a {MIN_CRUCE} min son {es(mano_total / n_anios)} h, o {es(mano_total / n_anios / HORAS_AUDITOR)} auditores-año")
    print("  revisar todos los casos del radar (" + es(n_casos / n_anios) + " por año): " + ", ".join(
        f"{mm} min: {es(n_casos * mm / 60 / n_anios)} h = {es(R['oci']['nacional'][f'casos_{mm}_auditores_anio'], 1)} auditores-año"
        for mm in MIN_REVISION))
    # costo de operar frente a lo expuesto
    anual_sol = COSTO_ANUAL_USD * CAMBIO
    R["equilibrio"] = {"costo_anual_soles": anual_sol, "monto_expuesto_anual": R["exposicion"]["monto_por_anio"],
                       "fraccion": anual_sol / R["exposicion"]["monto_por_anio"]}
    print(f"\ncosto anual de operar: USD {COSTO_ANUAL_USD} = S/ {es(anual_sol)} = {pct(R['equilibrio']['fraccion'], 4)} de lo expuesto por año")

# --- Parte C: lo que valen esas horas en soles ---
# Sueldos mensuales brutos de auditores de OCI.
# Fuentes oficiales: Contraloría General de la República, Concurso Público de Méritos N° 06-2026-CG (DL 728, Anexo 2,
# puestos y posiciones, set. 2026; 1 049 plazas, media ponderada S/ 5 887): Auditor/a Analista I 5 500, Analista II 6 500,
# Especialista I 7 500 y Especialista II 8 500 (el monto más alto de esa convocatoria). CENEPRED, CAS 082 y 084-2024:
# Especialista en Control Gubernamental, 8 000.
# Sin fuente oficial hallada: 9 500 y 12 500 (rangos de mercado para especialista senior y directivo) y 15 000, una cota
# alta de sensibilidad para jefatura.
SUELDOS = (5500, 6500, 7500, 8500, 9500, 12500, 15000)
# sueldos al año que paga la entidad en el régimen DL 728: 12 sueldos, 2 gratificaciones con su bonificación de 9 %,
# la CTS (un sueldo más un sexto de cada gratificación) y EsSalud (9 % de los 12 sueldos)
CARGA = (12 + 2 * 1.09 + (1 + 2 / 12) + 12 * 0.09) / 12
SUELDO_BASE, MIN_BASE = 7500, 30   # cifra base: Auditor/a Especialista I de la Contraloría (2026) y 30 minutos por revisión
HORAS_DESCARGA_MES = 4      # supuesto: horas al mes de la descarga manual de las fuentes (por medir)


def costo_hora(s):
    return s * 12 * CARGA / HORAS_AUDITOR


print("\n--- Parte C: lo que valen esas horas, según el sueldo mensual bruto (carga de %s sobre el sueldo anual) ---" % es(CARGA, 2))
R["soles"] = {"sueldos": list(SUELDOS), "carga": CARGA, "horas_auditor_anio": HORAS_AUDITOR,
              "costo_hora": {str(s): costo_hora(s) for s in SUELDOS}, "por_minutos": {}}
print("  costo por hora de auditor: " + "; ".join(f"sueldo S/ {es(s)}: S/ {es(costo_hora(s), 0)}" for s in SUELDOS))
for mm in MIN_REVISION:
    f = R["horas"][str(mm)]
    fila = {}
    for s in SUELDOS:
        ch = costo_hora(s)
        fila[str(s)] = {"acierto_hoy": f["por_acierto_hoy"] * ch, "acierto_v0": f["por_acierto_v0"] * ch,
                        "diferencia": [x * ch for x in f["diferencia"]]}
        if "oci" in R:
            h = R["oci"]["horas"][str(mm)]
            fila[str(s)].update({"oci_mano_mediana": h["mano_mediana"] * ch, "oci_mano_p90": h["mano_p90"] * ch,
                                 "oci_casos_mediana": h["casos_mediana"] * ch, "oci_casos_p90": h["casos_p90"] * ch})
    R["soles"]["por_minutos"][str(mm)] = fila
    print(f"  {mm} min por revisión:")
    for s in SUELDOS:
        r_ = fila[str(s)]
        linea = (f"    sueldo S/ {es(s):>6}: cada acierto cuesta S/ {es(r_['acierto_hoy'], 0)} por monto y S/ {es(r_['acierto_v0'], 0)} con el v0; "
                 f"los 143 aciertos, S/ {es(r_['diferencia'][0] / 1e3, 0)} mil menos [{es(r_['diferencia'][1] / 1e3, 0)} a {es(r_['diferencia'][2] / 1e3, 0)} mil]")
        if "oci_mano_mediana" in r_:
            linea += (f" | por entidad-año: cruzar a mano S/ {es(r_['oci_mano_mediana'], 0)} (mediana) y S/ {es(r_['oci_mano_p90'], 0)} (p90) "
                      f"frente a S/ {es(r_['oci_casos_mediana'], 0)} y S/ {es(r_['oci_casos_p90'], 0)} con los casos de LLIKA")
        print(linea)
if "oci" in R and "nacional" in R["oci"]:
    nac = R["oci"]["nacional"]
    R["soles"]["pais_por_anio"] = {str(s): {"mano": nac["cruce_a_mano_auditores_anio"] * s * 12 * CARGA,
                                            **{f"casos_{mm}": nac[f"casos_{mm}_auditores_anio"] * s * 12 * CARGA for mm in MIN_REVISION}}
                                   for s in SUELDOS}
    print("  todo el país, por año (millones de soles): " + "; ".join(
        f"S/ {es(s)}: a mano {es(v['mano'] / 1e6, 1)}, casos de LLIKA a 30 min {es(v['casos_30'] / 1e6, 2)}"
        for s, v in ((int(k), v) for k, v in R["soles"]["pais_por_anio"].items())))
# la cifra base
chb = costo_hora(SUELDO_BASE)
fb = R["horas"][str(MIN_BASE)]
R["base"] = {"sueldo": SUELDO_BASE, "min_revision": MIN_BASE, "horas_al_anio": HORAS_AUDITOR, "carga": CARGA,
             "costo_hora": chb, "hora_solo_sueldo": SUELDO_BASE / (8 * 261 / 12),
             "acierto_hoy": fb["por_acierto_hoy"] * chb, "acierto_v0": fb["por_acierto_v0"] * chb,
             "revision_en_vano": MIN_BASE / 60 * chb, "ahorro_143": [x * chb for x in fb["diferencia"]]}
print(f"\n  CIFRA BASE (sueldo S/ {es(SUELDO_BASE)}, {MIN_BASE} min por revisión, {HORAS_AUDITOR} h al año, carga {es(CARGA, 3)}):")
print(f"    hora de auditor S/ {es(chb, 1)} (solo el sueldo: S/ {es(R['base']['hora_solo_sueldo'], 1)}); revisión en vano S/ {es(R['base']['revision_en_vano'], 0)}; "
      f"cada acierto S/ {es(R['base']['acierto_hoy'], 0)} por monto y S/ {es(R['base']['acierto_v0'], 0)} con el v0; "
      f"los 143 aciertos, S/ {es(R['base']['ahorro_143'][0] / 1e3, 1)} mil menos [{es(R['base']['ahorro_143'][1] / 1e3, 0)} a {es(R['base']['ahorro_143'][2] / 1e3, 0)} mil]")
if "oci" in R:
    h = R["oci"]["horas"][str(MIN_BASE)]
    R["base"].update({"oci_mano_mediana": h["mano_mediana"] * chb, "oci_mano_p90": h["mano_p90"] * chb,
                      "oci_casos_mediana": h["casos_mediana"] * chb, "oci_casos_p90": h["casos_p90"] * chb,
                      "pais_mano": R["oci"]["nacional"]["cruce_a_mano_auditores_anio"] * SUELDO_BASE * 12 * CARGA,
                      "pais_casos": R["oci"]["nacional"][f"casos_{MIN_BASE}_auditores_anio"] * SUELDO_BASE * 12 * CARGA})
    print(f"    por entidad-año: cruzar a mano S/ {es(R['base']['oci_mano_mediana'], 0)} (mediana) y S/ {es(R['base']['oci_mano_p90'], 0)} (p90); "
          f"con LLIKA S/ {es(R['base']['oci_casos_mediana'], 0)} y S/ {es(R['base']['oci_casos_p90'], 0)}; todo el país por año: "
          f"S/ {es(R['base']['pais_mano'] / 1e6, 1)} M a mano frente a S/ {es(R['base']['pais_casos'] / 1e6, 2)} M")
# punto de equilibrio en horas de auditor: el costo de operar (servidor más descarga manual) dividido entre el costo por hora
operar = COSTO_ANUAL_USD * CAMBIO + HORAS_DESCARGA_MES * 12 * chb
R["soles"]["equilibrio"] = {"costo_operar_anual": operar, "horas_descarga_mes": HORAS_DESCARGA_MES,
                            "horas_de_auditor_para_pagarse": {str(s): operar / costo_hora(s) for s in SUELDOS}}
print(f"  operar LLIKA un año: S/ {es(COSTO_ANUAL_USD * CAMBIO)} de servidor más {HORAS_DESCARGA_MES * 12} h de descarga (a S/ {es(chb, 0)} la hora) = "
      f"S/ {es(operar)}; se paga con " + ", ".join(f"{es(operar / costo_hora(s), 0)} h de auditor (sueldo S/ {es(s)})" for s in (5500, 8500, 12500, 15000)))

# --- Parte D: lo que paga el Estado en órdenes con alerta ---
# Cada orden va a su capa más cierta:
# 1, prohibición directa con cruce exacto: proveedor sancionado a la fecha de la orden, funcionario que se contrata en su
#    entidad y doble percepción;
# 2, impedimento probable por vínculo societario: empresa espejo de un sancionado;
# 3, indicios: el mismo dueño con varios RUC (posible fraccionamiento) y el posible familiar (con su probabilidad).
# No es pérdida: es lo que se pagó a quien no debía contratar (capas 1 y 2) o en condiciones que piden revisión (capa 3).
# Como referencia de escala, la Contraloría estimó que la corrupción y la inconducta funcional costaron en 2023 el 12,7 %
# del presupuesto ejecutado (S/ 24 268 millones; citado por la prensa): no es una tasa medida en estas órdenes.
if AGG is not None:
    print("\n--- Parte D: lo que paga el Estado en órdenes con alerta ---")
    A = AGG["anio"].set_index("anio")
    pc = A[["capa1", "capa2", "capa3"]]
    ult = int(A.index.max())
    casos_ult = int(A.loc[ult, "casos"])
    rev_ult = casos_ult * MIN_BASE / 60 * chb
    tot_ult = float(pc.loc[ult].sum())
    R["estado"] = {
        "por_anio": {int(a): {"capa1": float(r_["capa1"]), "capa2": float(r_["capa2"]), "capa3": float(r_["capa3"]),
                              "total": float(r_.sum()), "pct_del_anio": float(r_.sum() / A.loc[a, "monto_total"])}
                     for a, r_ in pc.iterrows()},
        "total": {c: float(pc[c].sum()) for c in pc.columns},
        "ordenes": {f"capa{c}": int(A[f"ordenes_capa{c}"].sum()) for c in (1, 2, 3)},
        "familiar_bruto": float(RES["familiar"]["bruto"]), "familiar_ponderado": float(RES["familiar"]["ponderado"]),
        "ultimo_anio": ult, "casos_ultimo_anio": casos_ult, "revisar_casos_ultimo_anio": rev_ult,
        "fraccion_que_paga_la_revision": rev_ult / tot_ult,
        "referencia_cgr": {"incidencia": INCIDENCIA_CGR,
                           "capas_1_2_ultimo_anio": INCIDENCIA_CGR * float(pc.loc[ult, "capa1"] + pc.loc[ult, "capa2"]),
                           "total_ultimo_anio": INCIDENCIA_CGR * tot_ult},
        "sancion_posterior": {"vendieron_2022_2023": float(monto[y == 1].sum()),
                              "en_las_1000": {n: float(punto[f"monto_{n}"][i1]) for n in NOM}},
    }
    print("  millones de soles por año (capa 1 prohibición directa; 2 impedimento probable; 3 indicios):")
    for a, r_ in R["estado"]["por_anio"].items():
        print(f"    {a}: {es(r_['capa1'] / 1e6, 1):>6} {es(r_['capa2'] / 1e6, 1):>7} {es(r_['capa3'] / 1e6, 1):>7}  total {es(r_['total'] / 1e6, 1):>6} "
              f"({pct(r_['pct_del_anio'], 2)} de las compras menores del año)")
    t_ = R["estado"]["total"]
    print(f"  2018-2025: capa 1 S/ {es(t_['capa1'] / 1e6, 1)} M ({es(R['estado']['ordenes']['capa1'])} órdenes); capa 2 S/ {es(t_['capa2'] / 1e6, 1)} M "
          f"({es(R['estado']['ordenes']['capa2'])}); capa 3 S/ {es(t_['capa3'] / 1e6, 1)} M ({es(R['estado']['ordenes']['capa3'])})")
    print(f"  posible familiar: S/ {es(R['estado']['familiar_bruto'] / 1e6, 1)} M en bruto, S/ {es(R['estado']['familiar_ponderado'] / 1e6, 1)} M ponderado por su probabilidad")
    print(f"  {ult}: S/ {es(tot_ult / 1e6, 1)} M en órdenes con alerta; revisar sus {es(casos_ult)} casos a {MIN_BASE} min y S/ {es(chb, 0)} la hora cuesta "
          f"S/ {es(rev_ult / 1e3, 0)} mil, el {pct(rev_ult / tot_ult, 3)} de lo pagado")
    ref = R["estado"]["referencia_cgr"]
    print(f"  referencia de escala (12,7 % de la Contraloría): S/ {es(ref['capas_1_2_ultimo_anio'] / 1e6, 1)} M en las capas 1 y 2 de {ult}; "
          f"S/ {es(ref['total_ultimo_anio'] / 1e6, 1)} M en todo lo marcado")
    sp_ = R["estado"]["sancion_posterior"]
    print(f"  los 1 996 proveedores sancionados después vendieron S/ {es(sp_['vendieron_2022_2023'] / 1e6, 1)} M en 2022-2023; "
          + ", ".join(f"{n} trae S/ {es(v / 1e6, 1)} M en sus 1 000 primeros" for n, v in sp_["en_las_1000"].items()))

# --- Parte E: el gasto en contratos con alerta que cada lista pone delante del auditor ---
# Con el mismo presupuesto de revisiones, el tiempo del auditor cuesta lo mismo con cualquier lista; lo que cambia es
# cuánto dinero en contratos con alerta (órdenes de 2022-2023, por capa) queda a la vista y cuánto queda sin revisar.
# Las filas de costos_proveedores siguen el orden de las tablas del taller, y se comprueba con el monto de cada proveedor.
if AGG is not None:
    pv = AGG["proveedores"]
    assert len(pv) == len(te) and np.allclose(pv["monto"].to_numpy(), monto), "el orden de proveedores no coincide con el taller"
    capas = {c: pv[f"capa{c}"].to_numpy(float) for c in (1, 2, 3)}
    m12 = capas[1] + capas[2]
    tot_mal = {c: float(v.sum()) for c, v in capas.items()}
    R["contratos_malos"] = {"total_2022_2023": tot_mal, "listas": {}}
    print("\n--- Parte E: dinero en contratos con alerta de 2022-2023 que cada lista pone delante del auditor ---")
    print(f"  proveedores del examen, total con alerta: capa 1 S/ {es(tot_mal[1] / 1e6, 1)} M; capa 2 S/ {es(tot_mal[2] / 1e6, 1)} M; "
          f"capa 3 S/ {es(tot_mal[3] / 1e6, 1)} M")
    for k in (1000, 2000, 5000):
        rev = k * MIN_BASE / 60 * chb
        print(f"  con {es(k)} revisiones (S/ {es(rev / 1e3, 0)} mil de tiempo de auditor a {MIN_BASE} min y S/ {es(chb, 0)} la hora):")
        for n in NOM:
            top = listas[n][:k]
            c1, c2, c3 = (float(capas[c][top].sum()) for c in (1, 2, 3))
            fuera = float(m12[(y == 1)].sum() - m12[top][y[top] == 1].sum())
            R["contratos_malos"]["listas"].setdefault(n, {})[str(k)] = {
                "capa1": c1, "capa2": c2, "capa3": c3, "capas12": c1 + c2, "total": c1 + c2 + c3,
                "pct_capas12": (c1 + c2) / (tot_mal[1] + tot_mal[2]), "soles_por_sol_de_auditor_capas12": (c1 + c2) / rev,
                "sancionados_capas12_fuera": fuera}
            print(f"    {n:<6} capa 1 S/ {es(c1 / 1e6, 1):>5} M, capa 2 S/ {es(c2 / 1e6, 1):>5} M, capa 3 S/ {es(c3 / 1e6, 1):>5} M | "
                  f"capas 1-2: {pct((c1 + c2) / (tot_mal[1] + tot_mal[2]), 1)} del total, S/ {es((c1 + c2) / rev, 0)} por cada sol de auditor")
    # rango del 95 % de las diferencias frente a hoy en las capas 1 y 2, con 1 000 revisiones (pesos de Poisson, semilla 11)
    rng_e = np.random.default_rng(SEMILLA)
    tops = {n: listas[n][:K] for n in NOM}
    sumas = {n: [] for n in NOM}
    for _ in range(B):
        w = rng_e.poisson(1.0, size=len(y)).astype(float)
        for n in NOM:
            sumas[n].append(float((w[tops[n]] * m12[tops[n]]).sum()))
    sumas = {n: np.asarray(v) for n, v in sumas.items()}
    for n in ("v0", "regla"):
        d_b = sumas[n] - sumas["hoy"]
        d_p = float(m12[tops[n]].sum() - m12[tops["hoy"]].sum())
        R["contratos_malos"][f"dif_{n}_hoy_capas12_1000"] = [d_p, float(np.percentile(d_b, 2.5)), float(np.percentile(d_b, 97.5))]
        print(f"  diferencia {n} menos hoy en capas 1-2, con 1 000 revisiones: S/ {es(d_p / 1e6, 1)} M "
              f"[{es(np.percentile(d_b, 2.5) / 1e6, 1)} a {es(np.percentile(d_b, 97.5) / 1e6, 1)}]")

SALIDA.mkdir(parents=True, exist_ok=True)
(SALIDA / "costo_de_no_usar.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\nguardado en {SALIDA / 'costo_de_no_usar.json'} ({time.time() - t0:.0f} s)")
