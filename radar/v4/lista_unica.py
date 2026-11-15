"""La lista única de LLIKA, en la versión que aprobó Pedro (HALLAZGOS 160, secciones 71 a 78). No toca corrida_sellada.py ni los veredictos de sus exámenes.

Por cada entidad y año, cinco filas:
    primero, vínculo: autoridad, su empresa y parientes (regla, sin peso);
    luego, R × log(1 + M) × C_local, donde R une dos modelos por proveedor (sanción y penalidad) y C_local es la parte de lo que vende el proveedor que va a esa entidad;
    el quinto puesto, la compra más grande con bandera; si no hay, la más grande.
Fuera, desde el ajuste: servicios básicos, telefonía y entidades públicas. La data cruda no se toca.

Métrica única: precisión en 5 por entidad-año, en todas las entidades-año, con señal adversa documentada, contando a cada proveedor una vez por año.

    python -I -X utf8 lista_unica.py exploracion   # 2020, 2021 y 2023; debe reproducir la sonda
    python -I -X utf8 lista_unica.py segunda       # 2022 y 2024, una sola vez: segunda medición, no ciega (sección 78)
    python -I -X utf8 lista_unica.py congelar      # 2025, una sola vez y sin etiquetas (sección 78)
Salida, fuera de git: resultados/lista_unica/. Las listas traen RUC y nombres: no salen de datos/.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import p1_tablas as P1  # noqa: E402
from casos import DIR, OUT, SALIDA, TIPOS, etiquetas_caso, eventos, top  # noqa: E402
from d15_variables import NUEVAS, lista_121  # noqa: E402
from d17_configuracion import alerta  # noqa: E402
from etiquetas import BASE, HOY_EXPLORACION, K, Hechos  # noqa: E402
from evaluacion import armar_corte  # noqa: E402
from m6_proveedor import NUEVAS_P, Fuentes, modelo_proveedor  # noqa: E402
sys.argv = _a

EXPLORACION = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
SEGUNDA = {2022: [2018, 2019, 2020], 2024: [2018, 2019, 2020, 2021, 2022]}     # los mismos ajustes de los exámenes
FECHA_SEGUNDA = "2026-10-31"                                                    # la misma fecha de prueba de los exámenes
ANIO_LISTA, AJUSTE_LISTA, CORTE_LISTA = 2025, [2018, 2019, 2020, 2021, 2022, 2023], "2025-12-31"
TELEFONIA = r"WIN EMPRESAS|GTD PER|AMERICATEL|OPTICAL NETWORKS|VIETTEL|BITEL|AMERICA MOVIL|ENTEL PERU|TELEFONICA"
MOCHILA_ALTA = 0.98          # el 2 % más alto del país: solo cambia el motivo que se muestra, no el orden
VERDADES = {"señal adversa": "y", "impedimento": "y_imp_real", "sanción": "y_S", "penalidad": "y_P", "hallazgo": "y_rev"}
ETIQUETAS = ["y", "y_imp_real", "y_S", "y_P", "y_rev"]
SALIDA_L = OUT / "lista_unica"
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def huella(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def sig(s):
    return 1 / (1 + np.exp(-np.clip(np.asarray(s, dtype=float), -30, 30)))


def fuera_de(t, nombres):
    return t["u_servicio_basico"].astype(bool) | t["u_proveedor_publico"].astype(bool) | t["ruc"].map(nombres).fillna("").str.upper().str.contains(TELEFONIA, regex=True)


def compras(a):
    """Por caso: banderas y la orden que se muestra. Bandera: 30 días o cerca del tope."""
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "monto", "tope"], filters=[("anio", "==", a)]).astype({"ruc_entidad": str, "ruc": str})
    q = pl.from_pandas(o).sort(["ruc_entidad", "ruc", "fecha", "id"]).with_columns(pl.col("fecha").cast(pl.Datetime("ms")), pl.col("monto").cast(pl.Float64), pl.col("tope").cast(pl.Float64), pl.lit(1).cast(pl.Int64).alias("uno"))
    q = q.with_columns(pl.col("monto").rolling_sum_by("fecha", "30d", closed="right").over(["ruc_entidad", "ruc"]).alias("s"), pl.col("uno").rolling_sum_by("fecha", "30d", closed="right").over(["ruc_entidad", "ruc"]).alias("n"))
    q = q.with_columns(((pl.col("n") >= 2) & (pl.col("s") > pl.col("tope"))).fill_null(False).alias("b30"), (pl.col("monto") >= 0.9 * pl.col("tope")).fill_null(False).alias("cerca"))
    q = q.with_columns((pl.col("b30") | pl.col("cerca")).alias("bandera"))
    k = ["ruc_entidad", "ruc"]
    p = q.sort("monto", descending=True)
    g = p.group_by(k, maintain_order=True).agg(pl.col("b30").any().alias("f30"), pl.col("cerca").any().alias("f_cerca"), pl.len().alias("ordenes_caso"), pl.col("id").first().alias("orden_mayor"), pl.col("monto").first().alias("monto_orden_mayor"))
    b = p.filter(pl.col("bandera")).group_by(k, maintain_order=True).agg(pl.col("id").first().alias("orden_con_bandera"), pl.col("monto").first().alias("monto_orden_con_bandera"))
    g = g.join(b, on=k, how="left").to_pandas()
    g["anio"] = a
    return g.astype(TIPOS)


def relacion_previa(casos, a):
    """Para la ficha: lo que el proveedor le vendió antes a esa entidad. No entra al puntaje (sección 73)."""
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc_entidad", "ruc", "anio", "monto"], filters=[("anio", "<", a)]).astype({"ruc_entidad": str, "ruc": str, "anio": int})
    r = o.groupby(["ruc_entidad", "ruc"], as_index=False).agg(monto_antes=("monto", "sum"), desde=("anio", "min"), anios_antes=("anio", "nunique"))
    return casos.merge(r, on=["ruc_entidad", "ruc"], how="left").fillna({"monto_antes": 0.0, "anios_antes": 0})


def armar_anio(a, anios, H, E, Pa, cprov, nombres, fecha, con_etiquetas=True):
    corte = f"{a}-12-31"
    tr, te = armar_corte(H, anios, a, fecha_prueba=fecha)
    tr, te = tr.reset_index(drop=True).astype({"ruc": str, "ruc_entidad": str, "anio": int}), te.reset_index(drop=True).astype({"ruc": str, "ruc_entidad": str, "anio": int})
    nuevas = Pa[["ruc", "anio"] + NUEVAS_P]
    tr, te = tr.merge(nuevas, on=["ruc", "anio"], how="left"), te.merge(nuevas, on=["ruc", "anio"], how="left")
    v = etiquetas_caso(E, tr, corte, ventana=None)
    f = fuera_de(tr, nombres).to_numpy()
    t2 = tr[~f].reset_index(drop=True)
    paso(f"ETAPA {a}: {len(t2):,} casos de ajuste; fuera {int(f.sum()):,}; {len(te):,} casos en la lista")
    s = modelo_proveedor(t2, v["y_S"].to_numpy()[~f], te, cprov, (100,))[100][0]
    paso(f"ETAPA {a}: modelo de sanción listo")
    p = modelo_proveedor(t2, v["y_P"].to_numpy()[~f], te, cprov, (100,))[100][0]
    paso(f"ETAPA {a}: modelo de penalidad listo")
    d = te[K + ["grupo", "c_monto", "c_ordenes", "p_monto", "p_exp_ocds_monto", "p_cat_monto", "t_autoridad_ordenes", "t_empresa_autoridad_ordenes", "t_pariente_raro_ordenes", "t_empresa_pariente_aut_ordenes", "h_prov_penalidades", "h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones"]].copy()
    if con_etiquetas:
        d = d.join(etiquetas_caso(E, d, fecha))
    d["alerta"] = alerta(te)
    d["fuera"] = fuera_de(te, nombres).to_numpy()
    d["M_p"] = d.groupby(["ruc", "anio"])["c_monto"].transform("sum")
    d["pS"], d["pP"] = sig(s), sig(p)
    d["s_S"], d["s_P"] = s, p
    return d


def preparar_lista(d, a):
    q = d[~d["fuera"]].copy()
    q = q.merge(compras(a), on=K, how="left")
    q[["f30", "f_cerca"]] = q[["f30", "f_cerca"]].fillna(False).astype(bool)
    V = q["alerta"].to_numpy(dtype=bool)
    fuerte = (q["t_autoridad_ordenes"].fillna(0) > 0) | (q["t_empresa_autoridad_ordenes"].fillna(0) > 0)
    pariente = (q["t_pariente_raro_ordenes"].fillna(0) > 0) | (q["t_empresa_pariente_aut_ordenes"].fillna(0) > 0)
    q["grado"] = np.select([~V, fuerte, pariente], [0, 3, 2], 1)
    q["R"] = 1 - (1 - q["pS"]) * (1 - q["pP"])
    q["c_local"] = q["c_monto"] / q["M_p"].clip(lower=1e-9)
    q["logM"] = np.log1p(q["c_monto"].astype(float))
    q["puntaje"] = q["R"] * q["logM"] * q["c_local"]
    q["s_lista"] = 1e4 * (q["grado"] >= 2) + q["puntaje"]
    u = q.drop_duplicates(["ruc", "anio"])[["ruc", "anio", "R"]].copy()
    u["percentil_historial"] = u.groupby("anio")["R"].rank(pct=True)
    q = q.merge(u[["ruc", "anio", "percentil_historial"]], on=["ruc", "anio"], how="left")
    q["s_monto"] = q["c_monto"].astype(float)
    q["s_tamano"] = np.log1p(q[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1))
    q["s_alerta_monto"] = 100 * V + q["logM"]
    return q


def lista(q):
    """Cuatro por el puntaje; el quinto, la compra más grande con bandera; si no hay, la más grande. Con 5 casos o menos, entran todos."""
    filas, quinto = [], []
    for _, g in q.groupby("grupo", sort=False):
        if len(g) <= 5:
            filas.extend(g.index.tolist())
            continue
        el = g.nlargest(4, "s_lista").index.tolist()
        r = g.drop(index=el)
        b = r[r["f30"] | r["f_cerca"]]
        i = (b if len(b) else r)["c_monto"].idxmax()
        filas.extend(el + [i])
        quinto.append(i)
    L = q.loc[filas].copy()
    L["es_quinto"] = L.index.isin(quinto)
    return L


def ficha(L, a, nombres, con_etiquetas=True):
    """El motivo de cada fila, en palabras, y lo que se muestra. El puntaje no se muestra: R no es una probabilidad."""
    con_bandera = L["f30"] | L["f_cerca"]
    L["motivo"] = np.select([L["grado"] == 3, L["grado"] == 2, L["es_quinto"] & con_bandera, L["es_quinto"], L["percentil_historial"] >= MOCHILA_ALTA], ["vínculo: autoridad o su empresa", "vínculo: pariente, es una coincidencia", "compra con bandera", "sin señal fuerte: compra más grande", "historial del proveedor"], "sin señal fuerte: peso en la entidad")
    L["orden_visible"] = np.where(L["es_quinto"] & con_bandera, L["orden_con_bandera"], L["orden_mayor"])
    L["monto_orden_visible"] = np.where(L["es_quinto"] & con_bandera, L["monto_orden_con_bandera"], L["monto_orden_mayor"])
    L["riesgo_mayor"] = np.where(L["pS"] >= L["pP"], "sanción", "penalidad")
    L["vinculo_declaracion"] = L["grado"] == 1
    L["nombre"] = L["ruc"].map(nombres)
    L = relacion_previa(L, a)
    cols = K + ["grupo", "nombre", "motivo", "orden_visible", "monto_orden_visible", "ordenes_caso", "c_monto", "c_local", "percentil_historial", "riesgo_mayor", "h_prov_penalidades", "h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones", "f30", "f_cerca", "vinculo_declaracion", "monto_antes", "desde", "anios_antes"]
    return L[cols + (ETIQUETAS if con_etiquetas else [])]


def medida(L, y):
    return {"filas": len(L), "proveedores": int(L[L[y]].groupby(["ruc", "anio"]).ngroups), "por_fila": int(L[y].sum())}


def ic_proveedores(A, B, y="y", n=2000, semilla=0):
    """Las entidades-año son un censo: lo incierto es qué proveedores tienen señal. Pesos de Poisson por proveedor, con las filas fijas."""
    def p(L):
        return L[L[y]].groupby(["ruc", "anio"]).size().reset_index()["ruc"]
    a, b = p(A), p(B)
    if len(b) == 0:
        return {"razon": None, "ic90": None}
    f, u = pd.factorize(pd.concat([a, b]))
    c = len(a)
    rng = np.random.default_rng(semilla)
    bs = []
    for _ in range(n):
        w = rng.poisson(1.0, len(u))
        d0 = w[f[c:]].sum() / len(B)
        bs.append((w[f[:c]].sum() / len(A)) / d0 if d0 > 0 else np.nan)
    return {"razon": (len(a) / len(A)) / (len(b) / len(B)), "ic90": [float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))]}


def tabla(partes):
    """Una sola medida; el desglose por verdad es la misma medida."""
    listas = {n: pd.concat([q[n] for q in partes.values()]) for n in next(iter(partes.values()))}
    r = {"por_verdad": {}, "por_anio": {}, "razones": {}, "razones_por_anio": {}, "motivos": {}}
    for v, y in VERDADES.items():
        r["por_verdad"][v] = {n: medida(L, y) for n, L in listas.items()}
    for a, p in partes.items():
        r["por_anio"][a] = {n: medida(L, "y") for n, L in p.items()}
        r["razones_por_anio"][a] = {c: ic_proveedores(p["lista única"], p[c]) for c in ("monto", "tamaño", "alerta y monto")}
    for y_n, y in VERDADES.items():
        r["razones"][y_n] = {c: ic_proveedores(listas["lista única"], listas[c], y) for c in ("monto", "tamaño", "alerta y monto")}
    return r


def motivos(fichas):
    F = pd.concat(fichas)
    t = {}
    for m, g in F.groupby("motivo"):
        t[m] = {"filas": len(g), "proveedores_con_senal": int(g[g["y"]].groupby(["ruc", "anio"]).ngroups), "por_fila": int(g["y"].sum())}
    return t


def md(r, titulo):
    orden = ["azar", "tamaño", "monto", "alerta y monto", "lista única"]
    s = [f"# {titulo}", "", "Proveedores distintos con señal, contando a cada proveedor una vez por año, sobre las filas del Top 5 de todas las entidades-año.", ""]
    s += ["| Verdad | " + " | ".join(orden) + " |", "|---|" + "---|" * len(orden)]
    for v, f in r["por_verdad"].items():
        s.append(f"| {v} | " + " | ".join(str(f[n]["proveedores"]) for n in orden) + " |")
    n_f = r["por_verdad"]["señal adversa"]["lista única"]["filas"]
    s += ["", f"Filas: {n_f:,}.", "", "| Año | " + " | ".join(orden) + " |", "|---|" + "---|" * len(orden)]
    for a, f in r["por_anio"].items():
        s.append(f"| {a} | " + " | ".join(str(f[n]["proveedores"]) for n in orden) + " |")
    s += ["", "| Lista única frente a | Razón | IC al 90 % |", "|---|---|---|"]
    for c, x in r["razones"]["señal adversa"].items():
        s.append(f"| {c} | {x['razon']:.2f} | {x['ic90'][0]:.2f} a {x['ic90'][1]:.2f} |")
    return "\n".join(s) + "\n"


def comunes():
    c116 = [c for c in lista_121() if c not in NUEVAS]
    assert len(c116) == 116, f"variables: {len(c116)}"
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    n = pd.read_parquet(BASE / "proveedores.parquet", columns=["ruc", "nombre"]).astype({"ruc": str}).drop_duplicates("ruc").set_index("ruc")["nombre"]
    return cprov, n


def partes_de(q, L):
    az = q.assign(_a=np.random.default_rng(0).random(len(q)))
    return {"lista única": L, "monto": top(q, "s_monto"), "tamaño": top(q, "s_tamano"), "alerta y monto": top(q, "s_alerta_monto"), "azar": top(az, "_a")}


def guardar(r, nombre, titulo):
    (SALIDA_L / f"{nombre}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (SALIDA_L / f"{nombre}.md").write_text(md(r, titulo), encoding="utf-8")
    print(md(r, titulo), flush=True)


def scripts():
    return {f: huella(Path(__file__).parent / f)[:16] for f in ("lista_unica.py", "casos.py", "evaluacion.py", "m6_proveedor.py", "etiquetas.py", "d17_configuracion.py", "p1_tablas.py")}


def exploracion():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    E = eventos()
    Pa = pd.read_parquet(DIR / "proveedor_anio.parquet").astype({"ruc": str, "anio": int})
    cprov, n = comunes()
    partes, comprobado, fichas = {}, {}, []
    for a, anios in EXPLORACION.items():
        d = armar_anio(a, anios, H, E, Pa, cprov, n, HOY_EXPLORACION)
        g = pd.read_parquet(SALIDA / f"sonda_pasado_{a}.parquet", columns=K + ["s_S_limpio", "s_P_limpio"]).astype(TIPOS)
        m = d[K + ["s_S", "s_P"]].merge(g, on=K, how="left")
        comprobado[a] = {"dif_S": float((m["s_S"] - m["s_S_limpio"]).abs().max()), "dif_P": float((m["s_P"] - m["s_P_limpio"]).abs().max())}
        paso(f"reproduce los puntajes de la sonda, {a}: {comprobado[a]}")
        assert max(comprobado[a].values()) < 1e-6, "los modelos no reproducen la sonda"
        q = preparar_lista(d, a)
        L = lista(q)
        F = ficha(L.copy(), a, n)
        F.to_parquet(SALIDA_L / f"lista_{a}.parquet", index=False)
        fichas.append(F)
        partes[a] = partes_de(q, L)
        paso(f"ETAPA {a}: lista lista; {len(L):,} filas; proveedores con señal {medida(L, 'y')['proveedores']}; motivos {F['motivo'].value_counts().to_dict()}")
        del d
    r = tabla(partes)
    r["comprobado"], r["motivos"] = comprobado, motivos(fichas)
    total = r["por_verdad"]["señal adversa"]["lista única"]["proveedores"]
    assert total == 646, f"debía repetir 646 y dio {total}"
    guardar(r, "tabla", "Lista única de LLIKA: 2020, 2021 y 2023 (exploración; cifras optimistas)")
    (SALIDA_L / "manifiesto.json").write_text(json.dumps({"anios": list(EXPLORACION), "huellas": scripts(), "fecha_etiquetas": str(HOY_EXPLORACION.date())}, ensure_ascii=False, indent=1), encoding="utf-8")


def segunda():
    assert not (SALIDA_L / "segunda_medicion.json").exists(), "la segunda medición se corre una sola vez"
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F_ = Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    cprov, n = comunes()
    partes, comprobado, fichas = {}, {}, []
    for a, anios in SEGUNDA.items():
        Pa = P1.tabla_proveedor(H, F_, anios + [a]).astype({"ruc": str, "anio": int})
        paso(f"ETAPA {a}: tabla de proveedores lista")
        d = armar_anio(a, anios, H, E, Pa, cprov, n, FECHA_SEGUNDA)
        ex = pd.read_parquet(SALIDA / f"examen_puntajes_{a}.parquet", columns=K + ETIQUETAS + ["alerta", "s_monto"]).astype(TIPOS)
        m = d[K + ETIQUETAS + ["alerta", "c_monto"]].merge(ex, on=K, how="outer", suffixes=("", "_ex"), indicator=True)
        comprobado[a] = {"llaves_distintas": int((m["_merge"] != "both").sum()), **{c: int((m[c].astype(bool) != m[f"{c}_ex"].astype(bool)).sum()) for c in ETIQUETAS + ["alerta"]}, "dif_monto": float((m["c_monto"].astype(float) - m["s_monto"].astype(float)).abs().max())}
        paso(f"frente al examen, {a}: {comprobado[a]}")
        assert comprobado[a]["llaves_distintas"] == 0 and sum(comprobado[a][c] for c in ETIQUETAS + ["alerta"]) == 0 and comprobado[a]["dif_monto"] < 1, "no coincide con el examen: se detiene antes de medir"
        q = preparar_lista(d, a)
        L = lista(q)
        F = ficha(L.copy(), a, n)
        F.to_parquet(SALIDA_L / f"lista_{a}.parquet", index=False)
        fichas.append(F)
        partes[a] = partes_de(q, L)
        paso(f"ETAPA {a}: lista lista; {len(L):,} filas")
        del d, Pa
    r = tabla(partes)
    r["comprobado"], r["motivos"] = comprobado, motivos(fichas)
    r["rotulo"] = "segunda medición, no ciega"
    r["huellas"] = scripts()
    guardar(r, "segunda_medicion", "Lista única de LLIKA: 2022 y 2024 (segunda medición, no ciega)")


def congelar():
    man = SALIDA_L / "lista_2025_manifiesto.json"
    sal = SALIDA_L / "lista_2025_unica.parquet"
    assert not man.exists(), "la lista de 2025 se congela una sola vez"
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F_ = Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    cprov, n = comunes()
    anterior = huella(SALIDA / "lista_2025.parquet")
    Pa = P1.tabla_proveedor(H, F_, AJUSTE_LISTA + [ANIO_LISTA]).astype({"ruc": str, "anio": int})
    paso("ETAPA 2025: tabla de proveedores lista")
    d = armar_anio(ANIO_LISTA, AJUSTE_LISTA, H, E, Pa, cprov, n, CORTE_LISTA, con_etiquetas=False)
    assert not any(c.startswith(("y", "fuentes", "conocido", "peso")) for c in d.columns), "la lista no guarda etiquetas"
    q = preparar_lista(d, ANIO_LISTA)
    F = ficha(lista(q), ANIO_LISTA, n, con_etiquetas=False)
    assert not any(c.startswith(("y", "fuentes", "conocido", "peso")) for c in F.columns), "la lista no guarda etiquetas"
    F.to_parquet(sal, index=False)
    assert huella(SALIDA / "lista_2025.parquet") == anterior, "la lista anterior cambió"
    m = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "anio": ANIO_LISTA, "ajuste": AJUSTE_LISTA, "corte": CORTE_LISTA, "filas": len(F), "entidades": int(F["ruc_entidad"].nunique()), "proveedores": int(F["ruc"].nunique()), "motivos": F["motivo"].value_counts().to_dict(), "sha256_lista": huella(sal), "sha256_lista_anterior_sin_tocar": anterior, "sha256_eventos": huella(DIR / "eventos_examen.parquet"), "huellas": scripts()}
    man.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    paso(f"CONGELADA: {m['filas']:,} filas; {m['entidades']:,} entidades; {m['motivos']}; huella {m['sha256_lista'][:16]}; la anterior sigue en {anterior[:16]}")


def main():
    SALIDA_L.mkdir(parents=True, exist_ok=True)
    modo = sys.argv[1] if len(sys.argv) > 1 else "exploracion"
    {"exploracion": exploracion, "segunda": segunda, "congelar": congelar}[modo]()
    paso(f"LISTA {modo}")


if __name__ == "__main__":
    main()
