"""Corrida por caso de LLIKA (HALLAZGOS 160, secciones 47, 48 y 52). Todo fijado antes de abrir 2022 y 2024. La versión con D1 está en _anteriores/.

Ensayo, con un año de exploración (debe reproducir lo medido):
    python -I -X utf8 corrida_sellada.py --anio 2023
Prueba del juicio, con los años de exploración:
    python -I -X utf8 corrida_sellada.py --prueba-juicio
Exámenes, una sola vez cada uno y solo con la orden de Pedro; primero 2022:
    python -I -X utf8 corrida_sellada.py --anio 2022 --abrir
    python -I -X utf8 corrida_sellada.py --anio 2024 --abrir
    python -I -X utf8 corrida_sellada.py --juicio
Salida: resultados/por_caso/ensayo_<año>.json o examen_<año>.json y .md, con sus puntajes.
"""
import argparse
import hashlib
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import auditor_corregido as AC  # noqa: E402
import p1_tablas as P1  # noqa: E402
import u2_ronda_final as U2  # noqa: E402
from alerta_impedimentos import impedimento_real  # noqa: E402
from casos import DIR, OUT, PRUEBA, SALIDA, TIPOS, columnas_adicionales, conteo, dos_columnas, etiquetas_caso, ic_prov_anio, top  # noqa: E402
from d15_variables import NUEVAS, lista_121, nivel  # noqa: E402
from d16_hallazgos_por_hora import azar as azar_v2, razon as razon_v2, simular  # noqa: E402
from d17_configuracion import alerta  # noqa: E402
from etiquetas import BASE, HOY_EXPLORACION, K, SELLADO_ANIO, SELLADO_DESDE, Hechos  # noqa: E402
from evaluacion import armar_corte  # noqa: E402
from m1_d1_configuracion import ajustar  # noqa: E402
from m2_objetivo import etiqueta_ajuste  # noqa: E402
from m6_proveedor import NUEVAS_P, Fuentes, modelo_proveedor  # noqa: E402
from medidas import alerta_local, fmt, medir  # noqa: E402
sys.argv = _a

AJUSTE = {2020: [2018], 2021: [2018, 2019], 2022: [2018, 2019, 2020], 2023: [2018, 2019, 2020, 2021], 2024: [2018, 2019, 2020, 2021, 2022]}
FECHA_EXAMEN = "2026-10-31"
ARBOLES_D1, ARBOLES_P, DESEMPATES = 50, 100, 20
PISO_ACIERTOS, PISO_ANIO, P0_CASO_1 = 30, 20, 2 / 3     # sección 52
NV = ["h_prov_ampl", "h_prov_ampl_parte"]
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def comprobar_portal():
    """La descarga que se usa es la del manifiesto: si el OECE actualiza sus archivos, el examen no cambia."""
    raiz = BASE.parents[2] / "crudos" / "oece_portal"
    m = pd.read_csv(raiz / "manifiesto.csv")
    malos = 0
    for r in m.itertuples():
        pref = U2_TABLAS[r.tabla]
        malos += hashlib.sha256((raiz / r.tabla / f"{pref}{r.anio}.xlsx").read_bytes()).hexdigest() != r.sha256
    assert malos == 0, f"{malos} archivos del portal no coinciden con el manifiesto"
    paso(f"portal: {len(m)} archivos coinciden con el manifiesto")


def comprobar_huellas():
    """Los exámenes corren con lo fijado en el pre-registro: si un archivo cambió, se para antes de abrir."""
    raiz = BASE.parents[3]
    pre = (raiz / "PREREGISTRO_BENCHMARK.md").read_text(encoding="utf-8")
    assert "**Estado:** fijado" in pre, "el pre-registro no está fijado"
    filas = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{16})` \|", pre)
    assert len(filas) >= 20, f"huellas leídas: {len(filas)}"
    malos = [a for a, h in filas if hashlib.sha256((raiz / a).read_bytes()).hexdigest()[:16] != h]
    assert not malos, f"cambiaron desde que se fijó: {malos}"
    paso(f"huellas: {len(filas)} archivos coinciden con el pre-registro")


U2_TABLAS = {"arbitraje": "ARBITRAJE", "ampliaciones": "AMPLIACIONES", "consorcios": "CONSORCIO", "miembroscomite": "MIEMBROCOMITE", "contratos": "CONTRATOS", "convocatorias": "CONVOCATORIAS"}


def tiene_A(E, d, fecha):
    e = E[(E["fuente"] == "A") & (E["fecha_conocido"] <= pd.Timestamp(fecha)) & E["ruc_entidad"].notna()]
    return pd.MultiIndex.from_frame(d[K].astype(TIPOS)).isin(pd.MultiIndex.from_arrays([e["ruc_entidad"].astype(str), e["ruc"].astype(str), e["anio_hechos"].astype(int)]))


def vista2(d, a, fecha, abrir, n=DESEMPATES):
    """Caso 5, con 20 órdenes por entidad-año. El resultado es el promedio de n desempates entre órdenes del mismo monto (160, sección 48)."""
    inf = pd.read_parquet(BASE / "informes.parquet", columns=["informe", "fecha_emision", "fecha_publicacion"]).set_index("informe")
    f_inf = inf["fecha_publicacion"].fillna(inf["fecha_emision"])
    ho = pd.read_parquet(BASE / "hechos_orden.parquet")
    A = ho[(ho["fuente"] == "A") & ho["entidad_coincide"].astype(bool)].copy()
    A["fecha_conocido"] = A["documento"].map(f_inf)
    if not abrir:
        A = A[(A["anio"] != SELLADO_ANIO) & (A["fecha_conocido"] < SELLADO_DESDE)]
    ids = set(A.loc[A["fecha_conocido"] <= pd.Timestamp(fecha), "id"])
    t = d[K + ["grupo", "y_rev", "c_monto", "tiene_A"]].copy()
    t = t[t.groupby("grupo")["y_rev"].transform("any")]
    o0 = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto"], filters=[("anio", "==", a)])
    o0 = o0.astype({"ruc_entidad": str, "ruc": str}).assign(anio=a).merge(t, on=K, how="inner")
    partes = []
    for s in range(n):
        rng = np.random.default_rng(s)
        o = o0.assign(j=rng.random(len(o0))).sort_values(["grupo", "monto", "j"], ascending=[True, False, True]).reset_index(drop=True)
        o["r_caso"] = o.groupby(K).cumcount() + 1
        o["citada"] = o["id"].isin(ids)
        o["delata"] = (o["y_rev"] & o["tiene_A"] & o["citada"]) | (o["y_rev"] & ~o["tiene_A"] & (o["r_caso"] == 1))
        o["caso"] = o.groupby(K, sort=False).ngroup()
        r = simular(o, 20, "delata")
        r["azar"] = azar_v2(o, 20, "delata").reindex(r.index).fillna(0)
        partes.append(r.assign(s=s))
    p = pd.concat(partes)
    cols = ["casos enteros por monto", "vista 2", "regla del tope", "azar", "B"]
    r = p.groupby(level=0)[cols].mean()
    por_tirada = p.groupby("s")[cols].sum()
    return r, {"entidades_anio": len(r), "horas": float(r["B"].sum()), "hallazgos": {c: float(r[c].sum()) for c in cols[:4]}, "rango_desempates": {c: [float(por_tirada[c].min()), float(por_tirada[c].max())] for c in cols[:3]},
               "vista2_frente_a_la_regla": razon_v2(r, "vista 2", "regla del tope"), "vista2_frente_a_casos_enteros": razon_v2(r, "vista 2", "casos enteros por monto"), "vista2_frente_al_azar": razon_v2(r, "vista 2", "azar")}


def control_anio(E, a, fecha):
    """Entidades-año con un informe que cubre el año y sin hallazgo cruzado (160, secciones 40 y 42)."""
    i = pd.read_parquet(BASE / "informes.parquet", columns=["informe", "ruc_entidad", "alcance_desde", "alcance_hasta", "fecha_emision", "fecha_publicacion"])
    j = i[i["fecha_publicacion"].fillna(i["fecha_emision"]) <= pd.Timestamp(fecha)].dropna(subset=["ruc_entidad", "alcance_desde", "alcance_hasta"]).copy()
    j["d"], j["h"] = pd.to_datetime(j["alcance_desde"], errors="coerce").dt.year, pd.to_datetime(j["alcance_hasta"], errors="coerce").dt.year
    j = j.dropna(subset=["d", "h"])
    con = set(j.loc[(j["d"] <= a) & (j["h"] >= a), "ruc_entidad"].astype(str))
    e = E[E["fuente"].isin(["A", "B"]) & (E["anio_hechos"] == a) & (E["fecha_conocido"] <= pd.Timestamp(fecha))].dropna(subset=["ruc_entidad"])
    return con, con - set(e["ruc_entidad"].astype(str))


def auditor(E, d, a, fecha, sin_hallazgo):
    """El auditor sin inflar (160, sección 42): solo se reporta; no decide nada."""
    AC.HOY_EXPLORACION = pd.Timestamp(fecha)
    LISTAS = {"alerta y monto": "s_alerta_monto", "modelo S": "s_S_prov", "modelo P con ampliaciones": "s_P_ampl", "tamaño": "s_tamano_proveedor", "regla por monto": "s_monto"}
    d = d.copy()
    d["f_lista"] = pd.Timestamp(f"{a}-12-31")
    d["f_auditor"] = AC.fecha_auditor(E, d)
    for y, f in AC.fechas_evento(E, d).items():
        d[f"f_{y}"] = f
    aud = d[d["y_rev"]]
    res = {"auditor": {v: AC.cuenta(aud, y, "f_auditor") for v, y in AC.VERDADES.items()}, "auditor_hallazgos": len(aud)}
    obs = aud.groupby("grupo").size()
    sin = {f"{e}|{a}" for e in sin_hallazgo} & set(d["grupo"])
    n_g = pd.concat([np.ceil(obs / AC.TASA).astype(int), pd.Series(AC.REV_SIN_HALLAZGO, index=sorted(sin - set(obs.index)))]).rename_axis("grupo")
    res["entidades_anio_con_control"], res["revisiones_que_paga"] = len(n_g), int(n_g.sum())
    t = d[d["grupo"].isin(n_g.index)]
    n_a = pd.Series({a: int(n_g.sum())}).rename_axis("anio")
    for ambito, base, n_por in (("donde_hubo_control", t, n_g), ("todo_el_pais", d, n_a)):
        res[ambito] = {}
        for lista, col in LISTAS.items():
            L = AC.elegir(base, col, n_por)
            res[ambito][lista] = {v: AC.cuenta(L, y, "f_lista") for v, y in AC.VERDADES.items()}
            res[ambito][lista]["hallazgos"] = int(L["y_rev"].sum())      # esa verdad la pone el auditor: se reporta con esa nota
    return res


def azar_columnas(d, y, n=DESEMPATES):
    t = d[d.groupby("grupo")[y].transform("any")]
    filas = []
    for s in range(n):
        q = t.assign(_a=np.random.default_rng(s).random(len(t)))
        filas.append(conteo(top(q, "_a", semilla=s), y))
    return {k: float(np.mean([f[k] for f in filas])) for k in ("revisiones", "aciertos", "prov_anio_revisados", "prov_anio_acierto")}


def fila_md(n, r):
    return f"| {n} | {r['aciertos']:,} de {r['revisiones']:,} ({r['precision_caso']:.3f}) | {r['prov_anio_acierto']:,} de {r['prov_anio_revisados']:,} ({r['precision_prov_anio']:.3f}) |"


def binomial_discordantes(q, y, col, vara="s_tamano_proveedor"):
    """Proveedores-año hallados solo por una lista. p0: la parte de los revisados solo por la lista, porque las dos revisan distinto número (160, sección 52)."""
    t_ = q[q.groupby("grupo")[y].transform("any")]
    A, B = top(t_, col), top(t_, vara)
    pa = lambda L: set(zip(L["ruc"], L["anio"]))  # noqa: E731
    ra, rb, ha, hb = pa(A), pa(B), pa(A[A[y]]), pa(B[B[y]])
    n1, n0, e1, e0 = len(ha - hb), len(hb - ha), len(ra - rb), len(rb - ra)
    p0 = e1 / max(e1 + e0, 1)
    return {"solo_lista": n1, "solo_vara": n0, "p0": p0, "p": float(binomtest(n1, n1 + n0, p0, alternative="greater").pvalue) if n1 + n0 else None}


def veredicto(razon, ic, modelo, vara, anios, binomial=None, sin_binomial=False):
    """La regla de la sección 52. anios: {año: {razon, ic90, vara}}."""
    if max(modelo, vara) < PISO_ACIERTOS:
        return f"candidata: ni la solución ni su vara llegan a {PISO_ACIERTOS} aciertos"
    if vara < PISO_ACIERTOS and sin_binomial:
        return f"candidata: la vara tiene menos de {PISO_ACIERTOS} aciertos y aquí no hay binomial"
    por_anio = all((v["razon"] is not None and v["razon"] >= 1) if v["vara"] >= PISO_ANIO else (v["ic90"] is None or v["ic90"][1] >= 1.05) for v in anios.values())
    pasa = razon is not None and ic is not None and razon >= 1.05 and ic[0] > 1 and por_anio
    if vara < PISO_ACIERTOS:
        pasa = pasa and binomial is not None and binomial["p"] is not None and binomial["p"] < 0.05
    return "pasa" if pasa else "no pasa"


def medir_casos(d, anios, con_azar=True):
    """Las medidas de un conjunto de años. d trae puntajes y etiquetas."""
    cs = {a: g for a, g in d.groupby("anio")}
    res = {}
    M1 = {"alerta y monto": "s_alerta_monto", "regla por monto": "s_monto", "tamaño": "s_tamano_proveedor"}
    r = medir(cs, "y_imp_real", {"alerta": "s_alerta_monto", "monto": "s_monto"}, ["monto"])["metodos"]["alerta"]["frente_a"]["monto"]
    c = dos_columnas(d, "y_imp_real", M1)
    res["caso_1"] = {"columnas": c, "adicionales": columnas_adicionales(d, "y_imp_real", M1), "razon": r["razon"], "ic90_entidad": r["ic90_entidad"], "por_anio": {k: v["razon"] for k, v in r["por_anio"].items()}, "prov_anio": ic_prov_anio(d, "y_imp_real", "s_alerta_monto", "s_monto"), "vara": c["regla por monto"]["aciertos"]}
    for nom, y, col, extra in (("caso_3", "y_S", "s_S_prov", {}), ("caso_4", "y_P", "s_P_ampl", {"modelo sin ampliaciones": "s_P_prov"})):
        M = {"solución": col, **extra, "tamaño": "s_tamano_proveedor", "regla por monto": "s_monto"}
        c = dos_columnas(d, y, M)
        g = ic_prov_anio(d, y, col, "s_tamano_proveedor")
        pa = {int(a): ic_prov_anio(d[d["anio"] == a], y, col, "s_tamano_proveedor", n=1)["razon"] for a in cs}
        res[nom] = {"columnas": c, "adicionales": columnas_adicionales(d, y, M), "frente_tamano": g, "por_anio": pa, "frente_monto": ic_prov_anio(d, y, col, "s_monto"), "vara": c["tamaño"]["prov_anio_acierto"],
                    "por_caso": fmt(medir(cs, y, {"m": col, "t": "s_tamano_proveedor"}, ["t"])["metodos"]["m"]["frente_a"]["t"])}
        if extra:
            res[nom]["frente_modelo_sin_ampliaciones"] = ic_prov_anio(d, y, col, "s_P_prov")
    if con_azar:
        for nom, y in (("caso_1", "y_imp_real"), ("caso_3", "y_S"), ("caso_4", "y_P")):
            res[nom]["azar"] = azar_columnas(d, y)
    MD = {"regla por monto": "s_monto", "tamaño": "s_tamano_proveedor", "alerta y monto": "s_alerta_monto", "modelo S": "s_S_prov", "D1": "s_D1"}
    for nom, y in (("caso_2", "y_C_otra"), ("P_misma_entidad", "y_P_ent"), ("S_misma_entidad", "y_S_ent")):
        res[nom] = {"columnas": dos_columnas(d, y, MD), "positivos": int(d[y].sum())}
    res["vista_general"] = {"columnas": dos_columnas(d, "y", {"D1": "s_D1", "tamaño": "s_tamano_proveedor", "regla por monto": "s_monto"}), "D1_frente_tamano": ic_prov_anio(d, "y", "s_D1", "s_tamano_proveedor"), "D1_frente_monto": ic_prov_anio(d, "y", "s_D1", "s_monto")}
    return res


def correr_anio(a, abrir):
    fecha = FECHA_EXAMEN if abrir else str(HOY_EXPLORACION.date())
    corte = f"{a}-12-31"
    pref = "examen" if abrir else "ensayo"
    fres = SALIDA / f"{pref}_{a}.json"
    if abrir:
        assert a in (2022, 2024), "solo 2022 y 2024 se abren"
        assert not fres.exists(), "cada examen se corre una sola vez"
        comprobar_huellas()
        if a == 2024:
            assert (SALIDA / "examen_2022.json").exists(), "primero 2022"
    else:
        assert a not in (2022, 2024), "2022 y 2024 solo con --abrir y con la orden de Pedro"
    paso(f"año {a}; {'EXAMEN' if abrir else 'ensayo'}; ajuste {AJUSTE[a]}; fecha de prueba {fecha}")
    comprobar_portal()
    H = Hechos(conocimiento="publicacion", abrir_sellado=abrir)
    H.etiqueta = H.etiqueta_amplia
    F = Fuentes(abrir=abrir)
    E = P1.tabla_eventos(H, F)
    res = {"anio": a, "examen": abrir, "fecha_prueba": fecha, "ajuste": AJUSTE[a], "arboles_modelo": ARBOLES_P, "arboles_D1": ARBOLES_D1, "semillas": 10, "desempates_caso_5": DESEMPATES}
    comp = {}
    if not abrir:
        g = pd.read_parquet(DIR / "eventos.parquet")
        cc = ["fuente", "documento", "ruc", "anio_hechos", "fecha_conocido", "impedimento_real", "actor", "fecha_denuncia"]
        h = lambda z: int(pd.util.hash_pandas_object(z[cc].astype(str), index=False).sum() % (2 ** 62))  # noqa: E731
        comp["eventos_iguales"] = bool(len(g) == len(E) and h(g) == h(E))
        paso(f"Eventos: {len(E):,} filas; iguales a la tabla guardada {comp['eventos_iguales']}")
        assert comp["eventos_iguales"], "Eventos no reproduce"
    else:
        E.to_parquet(DIR / "eventos_examen.parquet", index=False)
    P = P1.tabla_proveedor(H, F, list(AJUSTE[a]) + [a])
    P = P.astype({"ruc": str, "anio": int})
    guardado = pd.read_parquet(DIR / "proveedor_anio.parquet").astype({"ruc": str, "anio": int})
    a_comp = [y for y in AJUSTE[a] + ([a] if not abrir else []) if y != SELLADO_ANIO]
    m = P[P["anio"].isin(a_comp)].merge(guardado[guardado["anio"].isin(a_comp)], on=["ruc", "anio"], how="outer", suffixes=("", "_g"), indicator=True)
    comp["proveedor_anio_dif_max"] = float(max((m[c].astype(float) - m[f"{c}_g"].astype(float)).abs().max() for c in NUEVAS_P))
    comp["proveedor_anio_llaves_iguales"] = bool((m["_merge"] == "both").all())
    paso(f"proveedor-año frente a lo guardado: {comp}")
    assert comp["proveedor_anio_llaves_iguales"] and comp["proveedor_anio_dif_max"] < 1e-6, "proveedor-año no reproduce"
    del guardado, m

    x = U2.ampliaciones(U2.contrato_a_ruc(), abrir)

    def preparar(t):
        t = t.astype({"ruc": str, "ruc_entidad": str, "anio": int}).merge(P[["ruc", "anio"] + NUEVAS_P], on=["ruc", "anio"], how="left")
        t["h_prov_ampl"] = 0.0
        for b, g in t.groupby("anio"):
            s = x[x["f"] <= pd.Timestamp(f"{b}-12-31")].groupby("ruc")["contrato"].nunique()
            t.loc[g.index, "h_prov_ampl"] = g["ruc"].map(s).fillna(0).to_numpy(dtype=float)
        t["h_prov_ampl_parte"] = t["h_prov_ampl"] / t["p_contratos_hasta_t"].astype(float).clip(lower=1)
        return t

    c116 = [c for c in lista_121() if c not in NUEVAS]
    assert len(c116) == 116, f"variables: {len(c116)}"
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    tr, te = armar_corte(H, AJUSTE[a], a, fecha_prueba=fecha)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    y_d1, peso = etiqueta_ajuste(H, tr, "T2", corte)
    tr, te = preparar(tr), preparar(te)
    v = etiquetas_caso(E, tr, corte, ventana=None)
    yS, yP = v["y_S"].to_numpy(), v["y_P"].to_numpy()
    fu = te["fuentes_estrictas"].fillna("").astype(str)
    d = te[K + ["grupo", "c_monto"]].copy()
    d = d.join(etiquetas_caso(E, d, fecha))
    d["alerta"] = alerta(te)
    d["s_monto"] = te["c_monto"].to_numpy(dtype=float)
    d["s_tamano_proveedor"] = np.log1p(te[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1)).to_numpy()
    d["s_alerta_monto"] = 100 * d["alerta"].to_numpy() + np.log1p(d["c_monto"].astype(float).to_numpy())
    d["tiene_A"] = tiene_A(E, d, fecha)
    # las etiquetas desde Eventos deben ser las de armar_corte
    dif = {"y": int((te["y"].astype(bool).to_numpy() != d["y"].to_numpy()).sum()), "y_estricta": int((te["y_estricta"].astype(bool).to_numpy() != d["y_estricta"].to_numpy()).sum()),
           "y_rev": int((fu.str.contains("A|B").to_numpy() != d["y_rev"].to_numpy()).sum()), "tiene_A": int((fu.str.contains("A").to_numpy() != d["tiene_A"].to_numpy()).sum()),
           "y_imp_real": int(((fu.str.contains("C").to_numpy() & impedimento_real(H, te)) != d["y_imp_real"].to_numpy()).sum())}
    comp["etiquetas_diferencias"] = dif
    paso(f"casos de ajuste {len(tr):,} (S {int(yS.sum()):,}; P {int(yP.sum()):,}); de prueba {len(te):,}; diferencias de etiquetas {dif}")
    assert sum(dif.values()) == 0, "las etiquetas desde Eventos no coinciden con armar_corte"
    res["positivos"] = {c: int(d[c].sum()) for c in ("y", "y_estricta", "y_rev", "y_imp_real", "y_C_otra", "y_S", "y_P")}
    res["casos_prueba"] = len(d)

    d["s_S_prov"] = modelo_proveedor(tr, yS, te, cprov, (ARBOLES_P,))[ARBOLES_P][0]
    paso("modelo S listo")
    d["s_P_prov"] = modelo_proveedor(tr, yP, te, cprov, (ARBOLES_P,))[ARBOLES_P][0]
    paso("modelo P sin ampliaciones listo")
    d["s_P_ampl"] = modelo_proveedor(tr, yP, te, cprov + NV, (ARBOLES_P,))[ARBOLES_P][0]
    paso("modelo P con ampliaciones listo")
    d["s_D1"] = ajustar(tr.assign(y=y_d1, peso=peso), te, c116, (ARBOLES_D1,), True)[ARBOLES_D1].mean(axis=0)
    paso("D1 listo")
    d.to_parquet(SALIDA / f"{pref}_puntajes_{a}.parquet", index=False)

    if not abrir:
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet", columns=K).astype(TIPOS)
        p2 = pd.read_parquet(SALIDA / f"puntajes_p2_{a}.parquet", columns=K + ["s_S_prov", "s_P_prov"]).astype(TIPOS)
        u = pd.read_parquet(SALIDA / f"u2_c_puntajes_{a}.parquet").astype(TIPOS)
        m2 = pd.read_parquet(OUT / f"puntajes_m2_{a}.parquet", columns=K + ["s_m2"]).astype(TIPOS)
        q = d[K + ["s_S_prov", "s_P_prov", "s_P_ampl", "s_D1"]].merge(p2, on=K, suffixes=("", "_g")).merge(u, on=K, suffixes=("", "_g")).merge(m2, on=K)
        comp["casos_iguales"] = bool(len(b) == len(d) == len(q))
        comp["puntajes_dif_max"] = {"S": float((q["s_S_prov"] - q["s_S_prov_g"]).abs().max()), "P": float((q["s_P_prov"] - q["s_P_prov_g"]).abs().max()), "P con ampliaciones": float((q["s_P_ampl"] - q["s_P_ampl_g"]).abs().max()), "D1": float((q["s_D1"] - q["s_m2"]).abs().max())}
        paso(f"reproducción de puntajes: {comp}")
    res["medidas"] = medir_casos(d, [a])
    r, v2 = vista2(d, a, fecha, abrir)
    r.to_parquet(SALIDA / f"{pref}_vista2_{a}.parquet")
    con, sin = control_anio(E, a, fecha)
    v2["control_sin_hallazgo"] = len({f"{e}|{a}" for e in sin} & set(d["grupo"]))
    v2["horas_con_control"] = v2["horas"] + 20 * v2["control_sin_hallazgo"]
    res["caso_5"] = v2
    paso(f"caso 5: {v2['hallazgos']}; rango {v2['rango_desempates']}; {v2['vista2_frente_a_la_regla']}; control sin hallazgo {v2['control_sin_hallazgo']}")
    res["alerta_local"] = alerta_local(d, d["alerta"].to_numpy(), d["y_imp_real"].to_numpy(), d["s_monto"].to_numpy(dtype=float))
    res["auditor"] = auditor(E, d, a, fecha, sin)
    if not abrir:
        t = pd.read_parquet(DIR / "entidad_anio.parquet")
        g_sin = {str(e) for e in t.loc[(t["anio"] == a) & t["control_sin_hallazgo"], "ruc_entidad"]}
        comp["control_sin_hallazgo_iguales"] = g_sin == sin
        ref = json.loads((SALIDA / "p2_resultado.json").read_text(encoding="utf-8"))
        ref_c = json.loads((SALIDA / "u2_c_penalidades.json").read_text(encoding="utf-8"))["confirmar"]["columnas"]
        conteos = {}
        for nom, lista_ref, y, lista in (("caso_1", ref["caso_1"]["columnas"]["alerta y monto"], "caso_1", "alerta y monto"), ("caso_1 monto", ref["caso_1"]["columnas"]["monto"], "caso_1", "regla por monto"), ("caso_1 tamaño", ref["caso_1"]["columnas"]["tamaño"], "caso_1", "tamaño"),
                                         ("caso_3", ref["caso_3"]["columnas"]["prov"], "caso_3", "solución"), ("caso_3 tamaño", ref["caso_3"]["columnas"]["tamaño"], "caso_3", "tamaño"), ("caso_3 monto", ref["caso_3"]["columnas"]["monto"], "caso_3", "regla por monto"),
                                         ("caso_4", ref_c["con ampliaciones"], "caso_4", "solución"), ("caso_4 sin ampliaciones", ref_c["modelo por proveedor"], "caso_4", "modelo sin ampliaciones"), ("caso_4 tamaño", ref_c["tamaño"], "caso_4", "tamaño"), ("caso_4 monto", ref_c["monto"], "caso_4", "regla por monto")):
            x0, x1 = lista_ref["por_anio"][str(a)], res["medidas"][y]["columnas"][lista]
            conteos[nom] = {"antes": [x0["aciertos"], x0["prov_anio_acierto"], x0["prov_anio_revisados"]], "ahora": [x1["aciertos"], x1["prov_anio_acierto"], x1["prov_anio_revisados"]]}
        comp["conteos"] = conteos
        comp["conteos_iguales"] = all(v["antes"] == v["ahora"] for v in conteos.values())
        paso(f"conteos iguales {comp['conteos_iguales']}: {conteos}")
    res["comprobaciones"] = comp
    fres.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    me = res["medidas"]
    L = [f"# {'Examen' if abrir else 'Ensayo'} por caso: {a} (fecha de prueba {fecha}; ajuste {AJUSTE[a]})", "", f"Casos: {len(d):,}. Positivos: {res['positivos']}", ""]
    for nom, tit in (("caso_1", "Caso 1. Impedimento"), ("caso_3", "Caso 3. Proveedor sancionado"), ("caso_4", "Caso 4. Proveedor con penalidad")):
        L += [f"## {tit}", "", "| Lista | Por caso | Por proveedor-año |", "|---|---|---|"] + [fila_md(n, x_) for n, x_ in me[nom]["columnas"].items()] + ["", f"Azar: {me[nom]['azar']}. Adicionales: {me[nom]['adicionales']}", ""]
    L += ["## Caso 5. Hallazgos", "", f"{v2['hallazgos']}; rango {v2['rango_desempates']}. Frente a las órdenes por monto: {v2['vista2_frente_a_la_regla']}.", "", "Un año solo no se juzga: el juicio es con los dos exámenes juntos.", "", "## Comprobaciones", "", json.dumps(comp, ensure_ascii=False, default=float)]
    (SALIDA / f"{pref}_{a}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    paso(f"sin juicio, un año solo. Caso 1: {me['caso_1']['razon']} {me['caso_1']['ic90_entidad']}; caso 3: {me['caso_3']['frente_tamano']}; caso 4: {me['caso_4']['frente_tamano']}; caso 5: {v2['vista2_frente_a_la_regla']}")
    paso("LISTO")


def juzgar(d, vistas):
    """El juicio con varios años juntos (160, sección 52). Caso 4: un solo candidato, el modelo con ampliaciones."""
    out = {}
    cs = {int(a): g.reset_index(drop=True) for a, g in d.groupby("anio")}
    r = medir(cs, "y_imp_real", {"alerta": "s_alerta_monto", "monto": "s_monto"}, ["monto"])["metodos"]["alerta"]["frente_a"]["monto"]
    loc = {a: alerta_local(g, g["alerta"].to_numpy(), g["y_imp_real"].to_numpy(), g["s_monto"].to_numpy(dtype=float)) for a, g in cs.items()}
    n1, n0 = sum(int(v["solo_alerta"]) for v in loc.values()), sum(int(v["solo_monto"]) for v in loc.values())
    p = float(binomtest(n1, n1 + n0, P0_CASO_1, alternative="greater").pvalue) if n1 + n0 else None
    cada = all(int(v["solo_alerta"]) > int(v["solo_monto"]) for v in loc.values())
    out["caso_1"] = {"razon_descriptiva": r["razon"], "ic90_descriptivo": r["ic90_entidad"], "por_anio": {a: (int(v["solo_alerta"]), int(v["solo_monto"])) for a, v in loc.items()}, "conjunto": (n1, n0), "p_dos_tercios": p, "mas_alerta_en_cada_anio": cada, "veredicto": "pasa" if p is not None and p < 0.05 and cada else "no pasa"}
    for nom, y, col in (("caso_3", "y_S", "s_S_prov"), ("caso_4", "y_P", "s_P_ampl")):
        c = dos_columnas(d, y, {"m": col, "v": "s_tamano_proveedor"})
        g = ic_prov_anio(d, y, col, "s_tamano_proveedor")
        an = {}
        for a in cs:
            q = d[d["anio"] == a]
            ga = ic_prov_anio(q, y, col, "s_tamano_proveedor")
            an[a] = {"razon": ga["razon"], "ic90": ga["ic90"], "vara": dos_columnas(q, y, {"v": "s_tamano_proveedor"})["v"]["prov_anio_acierto"]}
        modelo, vara = c["m"]["prov_anio_acierto"], c["v"]["prov_anio_acierto"]
        b = binomial_discordantes(d, y, col)
        out[nom] = {"razon": g["razon"], "ic90": g["ic90"], "aciertos": modelo, "vara": vara, "por_anio": an, "binomial": b, "veredicto": veredicto(g["razon"], g["ic90"], modelo, vara, an, b)}
    r5 = pd.concat(vistas.values())
    v = razon_v2(r5, "vista 2", "regla del tope")
    an = {}
    for a, x in vistas.items():
        z = razon_v2(x, "vista 2", "regla del tope")
        an[a] = {"razon": z["razon"], "ic90": z["ic90"], "vara": float(x["regla del tope"].sum())}
    modelo, vara = float(r5["vista 2"].sum()), float(r5["regla del tope"].sum())
    out["caso_5"] = {"razon": v["razon"], "ic90": v["ic90"], "aciertos": modelo, "vara": vara, "por_anio": an, "veredicto": veredicto(v["razon"], v["ic90"], modelo, vara, an, sin_binomial=True)}
    return out


def prueba_juicio():
    """Con los años de exploración debe repetir 7,50, 1,47, 1,10 y 1,26, y con pares de años, lo de revision_argon4 (caso 3, 3 de 3; caso 4, 2 de 3)."""
    from etiquetas import HOY_EXPLORACION as HOY
    E = U2.eventos()
    d = U2.casos(PRUEBA, E)
    d = d.merge(pd.concat([pd.read_parquet(SALIDA / f"u2_c_puntajes_{a}.parquet") for a in PRUEBA]).astype(TIPOS), on=K)
    d["tiene_A"] = tiene_A(E, d, HOY)
    vistas = {}
    for a in PRUEBA:
        vistas[a], f = vista2(d[d["anio"] == a].reset_index(drop=True), a, str(HOY.date()), False)
        paso(f"{a}: {f['hallazgos']}; rango {f['rango_desempates']}")
    res = {"tres_cortes": juzgar(d, vistas)}
    for n, v in res["tres_cortes"].items():
        paso(f"tres cortes, {n}: {v}")
    for x, z in itertools.combinations(PRUEBA, 2):
        res[f"{x}_{z}"] = juzgar(d[d["anio"].isin([x, z])], {x: vistas[x], z: vistas[z]})
        paso(f"par {x} y {z}: " + "; ".join(f"{n} {v['veredicto']}" for n, v in res[f"{x}_{z}"].items()))
    (SALIDA / "ensayo_juicio.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    paso("LISTO")


def juicio():
    assert (SALIDA / "examen_2022.json").exists() and (SALIDA / "examen_2024.json").exists(), "faltan exámenes"
    assert not (SALIDA / "examen_juicio.json").exists(), "el juicio se hace una sola vez"
    d = pd.concat([pd.read_parquet(SALIDA / f"examen_puntajes_{a}.parquet") for a in (2022, 2024)], ignore_index=True)
    out = juzgar(d, {a: pd.read_parquet(SALIDA / f"examen_vista2_{a}.parquet") for a in (2022, 2024)})
    me = medir_casos(d, [2022, 2024], con_azar=False)
    (SALIDA / "examen_juicio.json").write_text(json.dumps({"juicio": out, "medidas": me}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for n, v in out.items():
        paso(f"{n}: {v}")
    paso("LISTO")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anio", type=int)
    ap.add_argument("--abrir", action="store_true")
    ap.add_argument("--prueba-juicio", action="store_true")
    ap.add_argument("--juicio", action="store_true")
    args = ap.parse_args()
    if args.prueba_juicio:
        prueba_juicio()
    elif args.juicio:
        juicio()
    else:
        correr_anio(args.anio, args.abrir)
