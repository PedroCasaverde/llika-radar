# %% [markdown]
# # LLIKA: el experimento mínimo del taller (31/10/2026)
#
# **Qué es.** LLIKA es un radar que ordena a los proveedores del Estado para que el órgano de control institucional
# (OCI) revise primero a los que tienen más probabilidad de terminar sancionados. Este notebook responde una sola
# pregunta: ¿esa lista es mejor que lo que se hace hoy, revisar primero lo más caro? Además deja la corrida registrada
# para que cualquiera la repita y obtenga los mismos números.
#
# **De dónde salen los datos.** El radar completo (carpeta `radar/`) cruza las compras menores de CONOSCE, las sanciones
# del Tribunal, los socios y representantes del RNP, las declaraciones juradas de intereses y el padrón de SUNAT. Esos
# datos pesan unos 28 GB y traen RUC y DNI, así que no se pueden repetir en una mañana ni publicar. El script
# `exportar_datos_taller.py` resume la salida del radar en tres tablas livianas, una fila por proveedor y sin RUC, con la
# misma función que usa la evaluación del E5 (`radar/12_evaluacion_e5.py`). No son una muestra: están todos los
# proveedores. Cada tabla tiene su huella sha256 en `datos/huellas.json`; si alguien cambiara un solo dato, la huella no
# coincidiría.
#
# **Cómo correrlo.** En Colab, *Entorno de ejecución → Ejecutar todas*: el notebook baja solo las tablas. En una laptop,
# `pip install -r requirements.txt` y luego ejecutar las celdas de arriba abajo, o `python correr_taller.py`. En la PC
# donde se probó tardó unos 30 segundos.
#
# **El recorrido.**
# 1. Lo que declaramos antes de correr (pre-registro).
# 2. Por qué no hay fuga: con qué años se aprende y con cuáles se examina.
# 3. Las fórmulas con que se mide una lista.
# 4. La referencia: revisar primero lo más caro.
# 5. El candidato: el modelo v0 del E5, con su receta congelada.
# 6. Cuánto se mueve el número: la incertidumbre.
# 7. Dónde acierta y dónde falla cada lista.
# 8. Los KPIs del E3.
# 9. La comprobación contra el E5.
# 10. La línea de la plenaria y el registro de la corrida.
#
# Si las tablas no se pueden cargar, `MODO_DEMO = True` usa el 10 % de los proveedores. Los números cambian y hay que
# declararlo en la bitácora.

# %%
MODO_DEMO = False     # True solo si no se pueden cargar las tablas completas (hay que declararlo)
SEMILLA_BOOT = 11     # la misma del E5
K, B = 1000, 1000     # los 1000 primeros de cada lista y 1000 repeticiones del bootstrap
C_V0 = 0.01           # la regularización que eligió la validación cruzada del E5; hoy no se busca nada
URL = "https://raw.githubusercontent.com/PedroCasaverde/llika-radar/main/taller/datos/"

import hashlib, json, platform, sys, time, urllib.request, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):               # consolas de Windows y Git Bash que no usan UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", module="sklearn")   # avisos de versiones nuevas que no cambian el resultado

CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
ANTEC = ["log_multas", "log_penalidades", "evasion", "reciente"]


def pct(x, d=1):
    """Porcentaje con coma decimal: 0.143 -> '14,3 %'."""
    return f"{x * 100:.{d}f} %".replace(".", ",")


def ent(n):
    """Entero con espacio de miles: 942164 -> '942 164'."""
    return f"{int(n):,}".replace(",", " ")


def pp(x):
    """Diferencia en puntos porcentuales con signo: 0.106 -> '+10,6'."""
    return f"{x * 100:+.1f}".replace(".", ",")

# %%
# Busca las tablas junto al notebook; si no están, las baja del repositorio
ARCHIVOS = ["aprendizaje.parquet", "examen.parquet", "examen_entidades.parquet", "huellas.json"]
DATOS = next((d for d in (Path("datos"), Path("taller/datos")) if (d / "examen.parquet").exists()), Path("datos"))
DATOS.mkdir(exist_ok=True)
for f in ARCHIVOS:
    if not (DATOS / f).exists():
        print("bajando", f)
        urllib.request.urlretrieve(URL + f, DATOS / f)

huellas = json.loads((DATOS / "huellas.json").read_text(encoding="utf-8"))
for f, h in huellas["archivos"].items():
    real = hashlib.sha256((DATOS / f).read_bytes()).hexdigest()
    print(f"{f:<26} {'OK, es el conjunto congelado' if real == h['sha256'] else 'OJO: no coincide la huella'}")

tr = pd.read_parquet(DATOS / "aprendizaje.parquet")
te = pd.read_parquet(DATOS / "examen.parquet")
pares = pd.read_parquet(DATOS / "examen_entidades.parquet")
N_EXAMEN, SANC_EXAMEN = len(te), int(te["y"].sum())   # antes de cualquier muestra, para comprobar el conjunto

if MODO_DEMO:
    # muestra al azar del 10 % de los proveedores; K baja en la misma proporción
    rng_demo = np.random.default_rng(2026)
    te = te[rng_demo.random(len(te)) < 0.10].reset_index(drop=True)
    K = 100
    print("MODO_DEMO: 10 % de los proveedores del examen y K = 100. Declararlo en la bitácora.")

y = te["y"].to_numpy()
VENTANAS = {
    "aprendizaje": {"compras": "2018-2021", "etiqueta": "sanción que empieza en 2022 o 2023 (hasta el 31/12/2023)"},
    "examen": {"compras": "2022-2023", "etiqueta": "sanción que empieza desde el 1/1/2024"},
}

# %% [markdown]
# ## 1. Lo que declaramos antes de correr (pre-registro)
# Antes de ver un solo resultado se fija qué se compara, sobre qué conjunto, con qué métrica y con cuántas repeticiones.
# Así nadie puede escoger después la métrica que más le conviene.
#
# - La **referencia** es lo que hace hoy el control: revisar primero a los proveedores que más venden.
# - El **candidato** es el modelo v0 del E5.
# - La **métrica** es la precisión en los 1000 primeros: de los 1000 proveedores que cada lista pone arriba, qué
#   porcentaje recibió después una sanción del Tribunal de Contrataciones.
# - El **contrapeso** es la carga por entidad, que no debe empeorar: si la lista mejora pero le carga 40 casos a una sola
#   oficina, no sirve.
#
# La celda guarda la declaración en `bitacora/preregistro.json`.

# %%
PRE_REGISTRO = {
    "equipo": "G5 · LLIKA (carril A: medir la referencia)",
    "referencia": "lo que se hace hoy: los proveedores ordenados por monto total, de mayor a menor",
    "candidato": "modelo v0 del E5: regresión logística L1 con alertas de vínculo y antecedentes, receta congelada",
    "conjunto": {"descripcion": "proveedores con órdenes en 2022-2023, sin sanción previa a 2024; etiqueta = sanción desde 2024",
                 "n": int(len(te)), "con_sancion": int(y.sum()), "sha256": huellas["archivos"]["examen.parquet"]["sha256"]},
    "ventanas": VENTANAS,
    "metrica": f"precisión en los {K} primeros: de los {K} primeros de la lista, qué porcentaje tuvo sanción posterior",
    "contrapeso": "carga por entidad: cuántos de esos proveedores le tocan a cada entidad (mediana y percentil 90)",
    "incertidumbre": f"bootstrap de proveedores con pesos de Poisson, {B} repeticiones, semilla {SEMILLA_BOOT}, pareado entre listas",
    "receta_v0": {"modelo": "regresión logística L1 (liblinear)", "C": C_V0, "variables": CARTAS + ANTEC,
                  "muestra": "todos los sancionados y 100 000 no sancionados al azar (semilla 7)"},
    "modo_demo": MODO_DEMO,
    "fecha": datetime.now().isoformat(timespec="seconds"),
}
Path("bitacora").mkdir(exist_ok=True)
Path("bitacora/preregistro.json").write_text(json.dumps(PRE_REGISTRO, ensure_ascii=False, indent=1), encoding="utf-8")
for k_, v_ in PRE_REGISTRO.items():
    print(f"{k_:>13}: {v_}")

# %% [markdown]
# ## 2. Por qué no hay fuga: con qué años se aprende y con cuáles se examina
# Un modelo hace trampa si, al aprender, ve algo que solo se conoce en el examen. Aquí eso se evita con el tiempo:
#
# - el modelo **aprende** con las compras de 2018 a 2021 y con las sanciones que empezaron en 2022 o 2023;
# - se **examina** con las compras de 2022 y 2023 frente a las sanciones que empezaron desde el 1/1/2024.
#
# Las etiquetas del aprendizaje terminan el 31/12/2023 y las del examen empiezan el 1/1/2024, así que no se cruzan. Cada
# variable usa solo información hasta el 31 de diciembre del último año de su ventana, y en el examen no entra nadie que
# ya estuviera sancionado antes de 2024. Nada del examen se usa para escoger algo: la regla no aprende y el v0 llega con
# su receta fijada en el E5. La celda comprueba que las tablas cargadas son las declaradas.

# %%
fin_etiquetas_aprendizaje, inicio_etiquetas_examen = pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01")
assert fin_etiquetas_aprendizaje < inicio_etiquetas_examen, "las etiquetas se cruzan"
assert (len(tr), int(tr["y"].sum())) == (huellas["aprendizaje"]["n"], huellas["aprendizaje"]["con_sancion"])
assert (N_EXAMEN, SANC_EXAMEN) == (huellas["examen"]["n"], huellas["examen"]["con_sancion"])
print(pd.DataFrame({
    "compras": [VENTANAS["aprendizaje"]["compras"], VENTANAS["examen"]["compras"]],
    "etiqueta": [VENTANAS["aprendizaje"]["etiqueta"], VENTANAS["examen"]["etiqueta"]],
    "proveedores": [ent(len(tr)), ent(N_EXAMEN)],
    "sancionados": [ent(tr["y"].sum()), ent(SANC_EXAMEN)],
    "base": [pct(tr["y"].mean(), 3), pct(SANC_EXAMEN / N_EXAMEN, 3)],
}, index=["aprendizaje", "examen"]).to_string())
print("\nOK: las etiquetas no se cruzan y las tablas son las declaradas en huellas.json")

# %% [markdown]
# ## 3. Las fórmulas con que se mide una lista
# Son las mismas de `radar/12_evaluacion_e5.py`:
#
# - **Orden**: cada lista ordena a los proveedores de mayor a menor; los empates se rompen por monto.
# - **Precisión en los K primeros**: cuántos de los K primeros recibieron después una sanción, dividido entre K.
# - **Intervalo de Wilson**: el rango plausible de esa proporción, con una fórmula cerrada.
# - **Bootstrap pareado**: se repite el examen 1000 veces. En cada repetición, cada proveedor entra 0, 1, 2 o más veces
#   (pesos de Poisson), y todas las listas usan el mismo sorteo para que la diferencia entre ellas sea justa.
# - **Carga por entidad**: a cuántas entidades les tocan los K primeros y cuántos casos recibe cada una.

# %%
def orden(*claves):
    # ordena de mayor a menor por la primera clave; las siguientes desempatan
    return np.lexsort(tuple(-np.asarray(c, dtype=float) for c in reversed(claves)))


def bootstrap(y, listas, rng, k, b, largo=None):
    largo = largo or 4 * k
    res = {n: np.empty(b) for n in listas}
    for r in range(b):
        w = rng.poisson(1.0, size=len(y))          # cada proveedor entra 0, 1, 2... veces
        for n, idx in listas.items():
            ww = w[idx[:largo]]
            cw = np.cumsum(ww)
            j = int(np.searchsorted(cw, k))
            res[n][r] = (ww[: j + 1] * y[idx[: j + 1]]).sum() / cw[j]
    return res


def ic(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, c - m)), float(min(1.0, c + m))]


def carga(top_ids):
    por_ent = pares[pares["id"].isin(top_ids)].groupby("entidad").size()
    return {"entidades": int(len(por_ent)), "mediana": float(por_ent.median()), "p90": float(por_ent.quantile(0.9))}


def medir(idx):
    top = idx[:K]
    k = int(y[top].sum())
    return {"precision": k / K, "con_sancion": k, "wilson": wilson(k, K), "carga": carga(te["id"].to_numpy()[top])}

# %% [markdown]
# ## 4. La referencia: revisar primero lo más caro
# Es lo que contaron los especialistas de control en las entrevistas: el plan mira primero las compras de mayor monto.
# Aquí se ordena a los proveedores por lo que vendieron en 2022-2023 y se cuenta cuántos de los 1000 primeros fueron
# sancionados después. No hay modelo ni nada que entrenar. Es la tarea del carril A: la referencia corriendo, con su
# número.

# %%
listas = {"hoy": orden(te["monto"])}
res = {"hoy": medir(listas["hoy"])}
r = res["hoy"]
print(f"Referencia (lo más caro primero): {pct(r['precision'])} de los {K} primeros tuvo sanción posterior "
      f"({r['con_sancion']} proveedores). Al azar sería {pct(y.mean(), 2)}.")
print(f"Carga por entidad: mediana {r['carga']['mediana']:.0f}, percentil 90 {r['carga']['p90']:.0f}")

# %% [markdown]
# ## 5. El candidato: el modelo v0 del E5, con su receta congelada
# Hoy no se afina nada: se repite el modelo del E5 tal como quedó fijado el 28/10.
#
# - **La regla por puntos** del radar no aprende: suma la gravedad legal de las alertas del proveedor y desempata por
#   monto.
# - **El modelo v0** es una regresión logística con penalización L1 (Lasso). Aprende, con la ventana de aprendizaje,
#   cuánto pesa cada pista para anticipar una sanción, y deja en cero las que no ayudan. Usa las cinco alertas de
#   vínculo y cuatro variables de antecedentes: multas del Tribunal, penalidades de las entidades, evasión (la empresa
#   espejo empezó a vender después de la sanción de su vinculada) y antecedente reciente.
# - La regularización es C = 0,01, el valor que eligió la validación cruzada del E5. Como en el E5, aprende con todos
#   los sancionados y 100 000 no sancionados sorteados con semilla 7.

# %%
CORRER_CANDIDATO = True   # con False se mide solo la referencia y la regla

listas["regla"] = orden(te[CARTAS].sum(axis=1), te["monto"])
res["regla"] = medir(listas["regla"])

coef = None
if CORRER_CANDIDATO:
    t0 = time.time()
    ytr = tr["y"].to_numpy()
    rng = np.random.default_rng(7)                    # todos los sancionados y 100 000 no sancionados, como en el E5
    neg = np.flatnonzero(ytr == 0)
    fil = np.sort(np.concatenate([np.flatnonzero(ytr == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))
    cols = CARTAS + ANTEC
    sc = StandardScaler().fit(tr.iloc[fil][cols])
    m = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, C=C_V0)
    m.fit(sc.transform(tr.iloc[fil][cols]), ytr[fil])
    coef = {c: round(float(v), 3) for c, v in zip(cols, m.coef_[0])}
    puntaje = m.predict_proba(sc.transform(te[cols]))[:, 1]
    listas["v0"] = orden(puntaje, te["monto"])
    res["v0"] = medir(listas["v0"])
    print(f"v0 entrenado en {time.time() - t0:.0f} s con C = {C_V0}")
    print("pesos:", {c: v for c, v in coef.items() if v != 0}, "| en cero:", [c for c, v in coef.items() if v == 0])

for n_, r_ in res.items():
    print(f"{n_:<6} {pct(r_['precision'])}  ({r_['con_sancion']} de {K})  carga mediana {r_['carga']['mediana']:.0f}")

# %% [markdown]
# ## 6. Cuánto se mueve el número: la incertidumbre
# Con otros 942 164 proveedores parecidos, el resultado no sería idéntico. El bootstrap repite el examen 1000 veces con
# sorteos distintos y mira entre qué valores cae el 95 % de las repeticiones. Lo que importa es el rango de la
# **diferencia** con la referencia: si no toca el cero, la ventaja no es suerte del sorteo.

# %%
t0 = time.time()
bs = bootstrap(y, listas, np.random.default_rng(SEMILLA_BOOT), K, B)
for n_ in res:
    res[n_]["rango"] = ic(bs[n_])
dif = {f"{c}-hoy": {"dif": res[c]["precision"] - res["hoy"]["precision"], "rango": ic(bs[c] - bs["hoy"])}
       for c in res if c != "hoy"}
print(f"bootstrap de {B} repeticiones en {time.time() - t0:.0f} s")
for n_, r_ in res.items():
    print(f"{n_:<6} {pct(r_['precision'])}  rango 95 % [{pct(r_['rango'][0])} a {pct(r_['rango'][1])}]")
for n_, d_ in dif.items():
    print(f"diferencia {n_}: {pp(d_['dif'])} puntos porcentuales [{pp(d_['rango'][0])} a {pp(d_['rango'][1])}]")

# %% [markdown]
# ## 7. Dónde acierta y dónde falla cada lista
# Un promedio puede esconder dónde falla una lista. Se separan los 1000 primeros en grupos (con o sin alerta de vínculo,
# persona natural o empresa, proveedores que venden a una sola entidad) y se cuenta cuántos fueron sancionados en cada
# grupo. Al final se ve el motivo de las sanciones que encontró cada lista: falsedad, incumplimiento o impedimento.

# %%
def segmentos(idx):
    T = te.iloc[idx[:K]]
    yy = T["y"]
    con_alerta = T[CARTAS].sum(axis=1).gt(0)
    grupos = {"con alerta de vínculo": con_alerta, "sin alerta de vínculo": ~con_alerta,
              "persona natural": T["persona"].eq(1), "empresa": T["persona"].eq(0),
              "vende a una sola entidad": T["entidades"].eq(1)}
    return {g_: {"n": int(mm.sum()), "con_sancion": int(yy[mm].sum()),
                 "precision": (float(yy[mm].mean()) if mm.sum() else None)} for g_, mm in grupos.items()}


seg = {n_: segmentos(idx) for n_, idx in listas.items()}
print(pd.DataFrame({n_: {g_: f"{v_['con_sancion']}/{v_['n']}" for g_, v_ in s_.items()} for n_, s_ in seg.items()}))
motivos = {n_: te.iloc[idx[:K]].query("y == 1")["motivo"].value_counts().to_dict() for n_, idx in listas.items()}
print("\nmotivo de la sanción entre los sancionados de cada lista:")
for n_, mo_ in motivos.items():
    print(f"  {n_:<6} {mo_}")

# %% [markdown]
# ## 8. Los KPIs del E3
# Son los indicadores de impacto y el guardarraíl de tamaño que definimos en el E3, medidos sobre el mismo conjunto
# congelado:
#
# - aciertos entre los 50 y los 1000 primeros, frente a ordenar por monto y frente al azar;
# - cuántos de la lista no habrían entrado si se ordenara por monto;
# - el peso de los proveedores pequeños dentro de la lista, frente a su peso en el total.

# %%
K50 = 5 if MODO_DEMO else 50
mediana_monto = float(te["monto"].median())
pequeno = te["monto"].to_numpy() < mediana_monto     # pequeño = vendió menos que la mitad de los proveedores
peso_universo = float(pequeno.mean())


def kpis_e3(k):
    top_monto = set(listas["hoy"][:k])
    fila = {}
    for n_, idx in listas.items():
        top = idx[:k]
        a = int(y[top].sum())
        fila[n_] = {"k": k, "aciertos": a, "precision": a / k, "wilson": wilson(a, k),
                    "veces_el_azar": (a / k) / float(y.mean()),
                    "fuera_del_monto": len(set(top) - top_monto) / k,
                    "peso_pequenos": float(pequeno[top].mean()) / peso_universo}
    return fila


KPI_E3 = {"top50": kpis_e3(K50), "top1000": kpis_e3(K), "mediana_monto": mediana_monto,
          "peso_pequenos_universo": peso_universo}
for clave in ("top50", "top1000"):
    print(f"\n{clave}")
    for n_, r_ in KPI_E3[clave].items():
        peso_ = f"{r_['peso_pequenos']:.2f}".replace(".", ",")
        print(f"  {n_:<6} {r_['aciertos']}/{r_['k']} = {pct(r_['precision'])} "
              f"[{pct(r_['wilson'][0])} a {pct(r_['wilson'][1])}] · {r_['veces_el_azar']:.0f} veces el azar · "
              f"{pct(r_['fuera_del_monto'])} no habría entrado por monto · pequeños: {peso_} de su peso")
print(f"\nPequeño = vendió en 2022-2023 menos de S/ {ent(mediana_monto)} (la mediana); son el {pct(peso_universo, 0)} "
      "de los proveedores. «De su peso» = su parte en la lista dividida entre su parte en el total.")

# %% [markdown]
# ## 9. La comprobación contra el E5
# Si las tablas son las congeladas, los números tienen que ser los mismos del E5. Si alguno no coincide, algo cambió en
# los datos o en el código, y hay que revisarlo antes de seguir.

# %%
ESPERADO = {"hoy": 0.037, "regla": 0.071, "v0": 0.143}
if MODO_DEMO:
    print("MODO_DEMO: no se compara con el E5.")
else:
    for n_, e_ in ESPERADO.items():
        if n_ in res:
            ok = abs(res[n_]["precision"] - e_) < 0.0015
            print(f"{n_:<6} E5 {pct(e_)}, ahora {pct(res[n_]['precision'])}: {'coincide' if ok else 'NO coincide, revisar'}")

# %% [markdown]
# ## 10. La línea de la plenaria y el registro de la corrida
# La plenaria pide una línea por equipo: referencia, candidato, diferencia y dispersión. La celda la arma, guarda la
# corrida completa en `bitacora/` (fecha, versiones, semillas, huellas y todos los resultados) y suma una fila a
# `bitacora/BITACORA.md`. El auditor elige una corrida de esa tabla, corre el notebook y compara su JSON con el de la
# fila.

# %%
cand = "v0" if "v0" in res else "regla"
d_ = dif[cand + "-hoy"]
linea = (f"Referencia (lo más caro primero) {pct(res['hoy']['precision'])} · candidato ({cand}) "
         f"{pct(res[cand]['precision'])} · diferencia {pp(d_['dif'])} p.p. · dispersión: rango 95 % de la diferencia "
         f"[{pp(d_['rango'][0])} a {pp(d_['rango'][1])}] p.p. ({B} repeticiones, semilla {SEMILLA_BOOT})")
print(linea)

corrida = {"fecha": datetime.now().isoformat(timespec="seconds"), "modo_demo": MODO_DEMO,
           "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "sklearn": sklearn.__version__, "semillas": {"bootstrap": SEMILLA_BOOT, "negativos": 7},
           "receta_v0": PRE_REGISTRO["receta_v0"], "k": K, "repeticiones": B,
           "huellas": {f: h["sha256"] for f, h in huellas["archivos"].items()},
           "resultados": res, "diferencias": dif, "segmentos": seg, "motivos": motivos, "pesos_v0": coef,
           "kpis_e3": KPI_E3, "linea_plenaria": linea}
nombre = Path(f"bitacora/corrida_{datetime.now():%Y%m%d_%H%M%S}.json")
nombre.write_text(json.dumps(corrida, ensure_ascii=False, indent=1), encoding="utf-8")

bit = Path("bitacora/BITACORA.md")
if not bit.exists():
    bit.write_text("# Bitácora del experimento mínimo\n\n## Corridas\n\n| Fecha | Modo | Referencia | Candidato | "
                   "Diferencia en p.p. [rango 95 %] | Entorno | Archivo |\n|---|---|---|---|---|---|---|\n",
                   encoding="utf-8")
with bit.open("a", encoding="utf-8") as f_:
    f_.write(f"| {corrida['fecha'][:16].replace('T', ' ')} | {'demo (10 %)' if MODO_DEMO else 'completo'} | "
             f"{pct(res['hoy']['precision'])} | {pct(res[cand]['precision'])} ({cand}) | "
             f"{pp(d_['dif'])} [{pp(d_['rango'][0])} a {pp(d_['rango'][1])}] | Python {platform.python_version()}, "
             f"scikit-learn {sklearn.__version__} | `{nombre.name}` |\n")
print("guardado:", nombre, "y una fila más en", bit)
