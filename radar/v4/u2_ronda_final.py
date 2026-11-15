"""Última ronda, pruebas B, C y D (HALLAZGOS 160, sección 46). Fijado antes de medir. Sin 2022 como año de prueba ni como variable, sin lo conocido desde 2026 y sin 2024. Solo imprime agregados, nunca nombres.

    python -I -X utf8 u2_ronda_final.py b      # sancionados: arbitrajes a la fecha, prueba de techo
    python -I -X utf8 u2_ronda_final.py d      # impedimentos: miembros de comité como funcionarios
    python -I -X utf8 u2_ronda_final.py c      # penalidades: ampliaciones a la fecha, en el modelo por proveedor
    python -I -X utf8 u2_ronda_final.py extra  # columnas adicionales de los vigentes
Salida: resultados/por_caso/u2_<prueba>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
from casos import DIR, OUT, PRUEBA, SALIDA, TIPOS, columnas_adicionales, dos_columnas, etiquetas_caso, eventos, ic_prov_anio  # noqa: E402
from comun import PROC, norm_serie  # noqa: E402
from d15_variables import NUEVAS, lista_121  # noqa: E402
from d17_configuracion import alerta  # noqa: E402,F401
from etiquetas import BASE, HOY_EXPLORACION, K, Hechos  # noqa: E402
from evaluacion import armar_corte  # noqa: E402
from m6_proveedor import NUEVAS_P, modelo_proveedor  # noqa: E402
from medidas import medir  # noqa: E402
sys.argv = _a

PORTAL = BASE.parent / "portal"
CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
ELEGIR = (2020, 2021)
FIN = pd.Timestamp("2026-01-01")
SELLO = 2022
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def fec(s):
    """Las tablas del portal traen dd/mm/aaaa o aaaa-mm-dd según el archivo."""
    x = s.astype(str).str.strip()
    a = pd.to_datetime(x.where(x.str.match(r"\d{2}/\d{2}/\d{4}")).str[:10], format="%d/%m/%Y", errors="coerce")
    b = pd.to_datetime(x.where(x.str.match(r"\d{4}-\d{2}-\d{2}")).str[:10], format="%Y-%m-%d", errors="coerce")
    return a.fillna(b)


def valida(f, abrir=False):
    """Con fecha propia, sin 2022 y conocida antes de 2026. Con abrir (examen), solo con fecha propia."""
    return f.notna() if abrir else f.notna() & (f.dt.year != SELLO) & (f < FIN)


def codigo(s):
    return s.astype(str).str.strip().str.split(".").str[0]


def contrato_a_ruc():
    """Contrato a RUC: OCDS (adjudicación) y Contratos del portal; si el contratista es consorcio, también sus miembros. Solo RUC de 11 dígitos."""
    ad = pd.read_parquet(PROC / "ocds" / "adjudicaciones.parquet", columns=["ocid", "adjudicacion_id", "ruc"]).dropna(subset=["ruc"]).drop_duplicates()
    co = pd.read_parquet(PROC / "ocds" / "contratos.parquet", columns=["ocid", "adjudicacion_id", "contrato_id"])
    a = co.merge(ad, on=["ocid", "adjudicacion_id"])[["contrato_id", "ruc"]].rename(columns={"contrato_id": "contrato"})
    p = pd.read_parquet(PORTAL / "contratos.parquet", columns=["N_COD_CONTRATO", "RUC_CONTRATISTA"]).rename(columns={"N_COD_CONTRATO": "contrato", "RUC_CONTRATISTA": "ruc"})
    m = pd.concat([a, p], ignore_index=True).astype(str).drop_duplicates()
    m["contrato"] = codigo(m["contrato"])
    z = pd.read_parquet(PORTAL / "consorcios.parquet", columns=["RUC_CONSORCIO", "RUC_MIEMBRO"]).astype(str).drop_duplicates()
    mi = m.merge(z, left_on="ruc", right_on="RUC_CONSORCIO")[["contrato", "RUC_MIEMBRO"]].rename(columns={"RUC_MIEMBRO": "ruc"})
    r = pd.concat([m, mi], ignore_index=True).drop_duplicates()
    r = r[r["ruc"].str.fullmatch(r"\d{11}")]
    paso(f"contrato a RUC: {r['contrato'].nunique():,} contratos; {len(r):,} pares; por miembros de consorcio {len(mi):,}")
    return r


def casos(anios, E, extra=()):
    partes = []
    for a in anios:
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet", columns=K + ["grupo", "c_monto", "alerta", "s_monto", "s_tamano_proveedor", "s_alerta_monto"]).astype(TIPOS)
        b = b.merge(pd.read_parquet(SALIDA / f"puntajes_p2_{a}.parquet", columns=K + ["s_S_prov", "s_P_prov"]).astype(TIPOS), on=K)
        partes.append(b.join(etiquetas_caso(E, b, HOY_EXPLORACION)))
    d = pd.concat(partes, ignore_index=True)
    d["alerta"] = d["alerta"].astype(bool)
    return d


def guardar(n, r):
    (SALIDA / f"u2_{n}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ B
def razon_en_deciles(d, c_por_anio, y="y_S"):
    """c_por_anio: {año: Series de C por ruc}. Dentro de los 3 deciles superiores de tamaño de cada año."""
    p = d.groupby(["ruc", "anio"], sort=False).agg(t=("s_tamano_proveedor", "max"), y=(y, "max")).reset_index()
    p["alto"] = p.groupby("anio")["t"].rank(pct=True) > 0.7
    p["c"] = 0.0
    for a, s in c_por_anio.items():
        k = p["anio"] == a
        p.loc[k, "c"] = p.loc[k, "ruc"].map(s).fillna(0).to_numpy()
    q = p[p["alto"]]
    uno, cero = q[q["c"] >= 1], q[q["c"] < 1]
    tu, tc = uno["y"].mean() if len(uno) else 0.0, cero["y"].mean()
    r = {"proveedores_anio_altos": len(q), "con_c": len(uno), "positivos_con_c": int(uno["y"].sum()), "tasa_con_c": float(tu), "tasa_sin_c": float(tc), "razon": float(tu / tc) if tc else None,
         "con_c_todo_tamano": int((p["c"] >= 1).sum()), "positivos_con_c_todo_tamano": int((p["y"] & (p["c"] >= 1)).sum())}
    r["por_anio"] = {}
    for a, g in q.groupby("anio"):
        u, c = g[g["c"] >= 1], g[g["c"] < 1]
        r["por_anio"][int(a)] = {"con_c": len(u), "positivos_con_c": int(u["y"].sum()), "razon": float(u["y"].mean() / c["y"].mean()) if len(u) and c["y"].mean() else None}
    return r


def prueba_b():
    E = eventos()
    d = casos(ELEGIR, E)
    M = contrato_a_ruc()
    ar = pd.read_parquet(PORTAL / "arbitraje.parquet", columns=["N_COD_CONTRATO", "FECHA_INSTALACION", "FECHA_SUSCRIPCION_CONTRATO"])
    ar["contrato"], ar["f"], ar["firma"] = codigo(ar["N_COD_CONTRATO"]), fec(ar["FECHA_INSTALACION"]), fec(ar["FECHA_SUSCRIPCION_CONTRATO"])
    x = ar.merge(M, on="contrato")
    paso(f"arbitrajes: {len(ar):,}; con RUC {x['contrato'].nunique():,}; instalados hasta 2021: {int((ar['f'].dt.year <= 2021).sum())}; hasta 2020: {int((ar['f'].dt.year <= 2020).sum())}")
    res = {"arbitrajes": len(ar), "instalados_hasta_2020": int((ar["f"].dt.year <= 2020).sum()), "instalados_hasta_2021": int((ar["f"].dt.year <= 2021).sum())}
    c1 = {a: x[valida(x["f"]) & (x["f"] <= pd.Timestamp(f"{a}-12-31"))].groupby("ruc")["contrato"].nunique() for a in ELEGIR}
    res["a_la_fecha"] = razon_en_deciles(d, c1)
    pasa = bool((res["a_la_fecha"]["razon"] or 0) >= 2.5 and res["a_la_fecha"]["positivos_con_c"] >= 25)
    res["pasa_techo"] = pasa
    paso(f"A LA FECHA: {res['a_la_fecha']}; PASA EL TECHO {pasa}")
    # solo como techo con fuga: no decide
    c2 = {a: x[x["firma"] <= pd.Timestamp(f"{a}-12-31")].groupby("ruc")["contrato"].nunique() for a in ELEGIR}
    res["con_fuga_arbitraje_sin_fecha"] = razon_en_deciles(d, c2)
    paso(f"CON FUGA, arbitraje de contrato ya firmado: {res['con_fuga_arbitraje_sin_fecha']}")
    t = pd.read_parquet(PORTAL / "contratos.parquet", columns=["N_COD_CONTRATO", "TIENERESOLUCION", "FECHA_SUSCRIPCION_CONTRATO"])
    t = t[t["TIENERESOLUCION"] == "SI"].drop_duplicates("N_COD_CONTRATO")
    t["contrato"], t["firma"] = codigo(t["N_COD_CONTRATO"]), fec(t["FECHA_SUSCRIPCION_CONTRATO"])
    x2 = t.merge(M, on="contrato")
    c3 = {a: x2[x2["firma"] <= pd.Timestamp(f"{a}-12-31")].groupby("ruc")["contrato"].nunique() for a in ELEGIR}
    res["con_fuga_tieneresolucion"] = razon_en_deciles(d, c3)
    paso(f"CON FUGA, TIENERESOLUCION de contrato ya firmado: {res['con_fuga_tieneresolucion']}")
    guardar("b_sancionados", res)
    if pasa:
        paso("EL TECHO PASA: falta el modelo (fijado en 46), no se corre en esta llamada")


# ------------------------------------------------------------------------------------------------ D
def clave(s):
    """El mismo nombre completo, sin importar el orden de las palabras. Al menos 3 palabras."""
    u = pd.Series(s.unique())
    v = norm_serie(u.fillna("").astype(str)).map(lambda z: " ".join(sorted(z.split())) if len(z.split()) >= 3 else None)
    return s.map(dict(zip(u, v)))


def prueba_d():
    assert not (SALIDA / "u2_d_impedimentos.json").exists(), "se corre una sola vez"
    E = eventos()
    d = casos(PRUEBA, E)
    k = pd.read_parquet(PORTAL / "miembroscomite.parquet", columns=["CODIGOCONVOCATORIA", "MIEMBROCOMITE"]).astype(str).drop_duplicates()
    v = pd.read_parquet(PORTAL / "convocatorias.parquet", columns=["CODIGOCONVOCATORIA", "ENTIDAD_RUC", "FECHA_CONVOCATORIA"]).astype(str).drop_duplicates("CODIGOCONVOCATORIA")
    v["f"] = fec(v["FECHA_CONVOCATORIA"])
    k = k.merge(v, on="CODIGOCONVOCATORIA")
    k = k[valida(k["f"])]
    k["nom"] = clave(k["MIEMBROCOMITE"])
    k = k.dropna(subset=["nom"])
    com = k.groupby(["ENTIDAD_RUC", "nom"])["f"].min().rename("primera").reset_index().rename(columns={"ENTIDAD_RUC": "ruc_entidad"})
    com["anio_primera"] = com["primera"].dt.year
    paso(f"comité: {len(k):,} filas; {com['nom'].nunique():,} personas; {len(com):,} pares persona-entidad; entidades {com['ruc_entidad'].nunique():,}")
    cortes = sorted((PROC / "acc3_conformacion").glob("conformacion_corte*.parquet"))
    fechas = {c: pd.Timestamp(c.stem[-8:]) for c in cortes}
    marca = np.zeros(len(d), dtype=bool)
    idx = pd.MultiIndex.from_frame(d[K])
    for a in PRUEBA:
        fin = pd.Timestamp(f"{a}-12-31")
        vis = com[com["anio_primera"] <= a]
        # personas que venden
        o = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc", "nombre"], filters=[("anio", "==", a)]).astype({"ruc": str})
        o = o[o["ruc"].str.startswith("10")].drop_duplicates("ruc")
        o["nom"] = clave(o["nombre"])
        cs = d[(d["anio"] == a) & d["ruc"].str.startswith("10")][K].merge(o[["ruc", "nom"]], on="ruc").merge(vis[["ruc_entidad", "nom"]], on=["ruc_entidad", "nom"])
        # empresas con un integrante de ese nombre: el mismo corte que la alerta vigente
        validos = [c for c in cortes if fechas[c] <= fin]
        c = validos[-1] if validos else cortes[0]
        s = pd.read_parquet(c, columns=["ruc_proveedor", "nombre_integrante", "dni_integrante", "usable_para_cruce"])
        s = s[s["usable_para_cruce"].fillna(False).astype(bool) & s["dni_integrante"].str.fullmatch(r"\d{8}", na=False)].rename(columns={"ruc_proveedor": "ruc"}).drop_duplicates(["ruc", "dni_integrante"])
        s["nom"] = clave(s["nombre_integrante"])
        ce = d[(d["anio"] == a) & ~d["ruc"].str.startswith("10")][K].merge(s[["ruc", "nom"]], on="ruc").merge(vis[["ruc_entidad", "nom"]], on=["ruc_entidad", "nom"])
        m = pd.concat([cs, ce])[K].drop_duplicates()
        marca |= idx.isin(pd.MultiIndex.from_frame(m))
        paso(f"{a}: casos marcados por comité, personas {cs[K].drop_duplicates().shape[0]:,}; empresas {ce[K].drop_duplicates().shape[0]:,}")
    d["comite"] = marca
    d["a2"] = d["alerta"] | d["comite"]
    d["s_a2"] = 100 * d["a2"].astype(float) + np.log1p(d["c_monto"].astype(float))
    d["s_a1"] = 100 * d["alerta"].astype(float) + np.log1p(d["c_monto"].astype(float))
    res = {"casos_comite": int(d["comite"].sum()), "fuera_de_la_alerta": int((d["comite"] & ~d["alerta"]).sum()), "impedimentos_comite": int((d["comite"] & d["y_imp_real"]).sum()),
           "impedimentos_nuevos_en_total": int((d["comite"] & ~d["alerta"] & d["y_imp_real"]).sum()), "por_anio": {}}
    for a, g in d.groupby("anio"):
        n = g["comite"] & ~g["alerta"]
        res["por_anio"][int(a)] = {"casos_comite": int(g["comite"].sum()), "fuera_de_la_alerta": int(n.sum()), "impedimentos_nuevos": int((n & g["y_imp_real"]).sum()), "proveedores_de_nuevos": int(g.loc[n & g["y_imp_real"], "ruc"].nunique()), "impedimentos": int(g["y_imp_real"].sum()), "de_la_alerta": int((g["alerta"] & g["y_imp_real"]).sum())}
    paso(f"resumen: {res}")
    cs = {a: g for a, g in d.groupby("anio")}
    r = medir({a: cs[a] for a in ELEGIR}, "y_imp_real", {"nueva": "s_a2", "vigente": "s_a1"}, ["vigente"], n_boot=2000)["metodos"]
    n, v = r["nueva"], r["vigente"]
    ra = n["aciertos"] / max(v["aciertos"], 1)
    res["elegir"] = {"aciertos_nueva": n["aciertos"], "aciertos_vigente": v["aciertos"], "por_anio_nueva": n["aciertos_por_anio"], "por_anio_vigente": v["aciertos_por_anio"], "razon_aciertos": ra, "precision_nueva": n["precision"], "precision_vigente": v["precision"], "razon": n["frente_a"]["vigente"]["razon"], "ic90": n["frente_a"]["vigente"]["ic90_entidad"]}
    pasa = bool(ra >= 1.05)
    res["pasa_elegir"] = pasa
    paso(f"ELEGIR 2020 y 2021: {res['elegir']}; PASA {pasa}")
    if pasa:
        r = medir(cs, "y_imp_real", {"nueva": "s_a2", "vigente": "s_a1"}, ["vigente"], n_boot=2000)["metodos"]
        g = r["nueva"]["frente_a"]["vigente"]
        pa = {k: x["razon"] for k, x in g["por_anio"].items()}
        ok = bool(g["razon"] >= 1.05 and sum(1 for x in pa.values() if x and x >= 1.05) >= 2 and g["ic90_entidad"][0] > 1)
        res["confirmar"] = {"razon": g["razon"], "ic90": g["ic90_entidad"], "por_anio": pa, "aciertos_nueva": r["nueva"]["aciertos"], "aciertos_vigente": r["vigente"]["aciertos"], "pasa": ok, "columnas": dos_columnas(d, "y_imp_real", {"nueva": "s_a2", "vigente": "s_a1"})}
        paso(f"CONFIRMAR: {res['confirmar']}")
    else:
        paso("no pasa a confirmar; 2023 no se mira con esta alerta")
    guardar("d_impedimentos", res)
    if pasa:
        d.loc[d["a2"], K + ["comite"]].to_parquet(SALIDA / "u2_d_alerta_nueva.parquet", index=False)


# ------------------------------------------------------------------------------------------------ C
def ampliaciones(M, abrir=False):
    m = pd.read_parquet(PORTAL / "ampliaciones.parquet", columns=["CODIGOCONTRATO", "FECHA_PUBLICACION_AMPLIACION"])
    m["contrato"], m["f"] = codigo(m["CODIGOCONTRATO"]), fec(m["FECHA_PUBLICACION_AMPLIACION"])
    m = m[valida(m["f"], abrir)]
    x = m.merge(M, on="contrato")
    t = m.assign(cruza=m["contrato"].isin(set(M["contrato"])))
    paso(f"ampliaciones usables: {len(m):,}; cruzan con contrato: {t['cruza'].mean():.3f}" + ("" if abrir else "; por año " + str(t.groupby(t["f"].dt.year)["cruza"].mean().round(3).to_dict())))
    return x


def puntajes_c(anios, x, E):
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    P = pd.read_parquet(DIR / "proveedor_anio.parquet").astype({"ruc": str, "anio": int})
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    NV = ["h_prov_ampl", "h_prov_ampl_parte"]
    nuevas = P[["ruc", "anio"] + NUEVAS_P]

    def con_ampl(t):
        t = t.astype({"ruc": str, "ruc_entidad": str, "anio": int}).merge(nuevas, on=["ruc", "anio"], how="left")
        t["h_prov_ampl"] = 0.0
        for a, g in t.groupby("anio"):
            s = x[x["f"] <= pd.Timestamp(f"{a}-12-31")].groupby("ruc")["contrato"].nunique()
            t.loc[g.index, "h_prov_ampl"] = g["ruc"].map(s).fillna(0).to_numpy(dtype=float)
        t["h_prov_ampl_parte"] = t["h_prov_ampl"] / t["p_contratos_hasta_t"].astype(float).clip(lower=1)
        return t

    for a in anios:
        tr, te = armar_corte(H, CORTES[a], a)
        tr, te = con_ampl(tr.reset_index(drop=True)), con_ampl(te.reset_index(drop=True))
        y = etiquetas_caso(E, tr, f"{a}-12-31", ventana=None)["y_P"].to_numpy()
        paso(f"corte {a}: ajuste {len(tr):,} casos; proveedores-año con ampliación en la prueba {te.loc[te['h_prov_ampl'] > 0, ['ruc', 'anio']].drop_duplicates().shape[0]:,}; positivos de ajuste {int(y.sum()):,}")
        s = modelo_proveedor(tr, y, te, cprov + NV, (100,))[100][0]
        te[K].assign(s_P_ampl=s).to_parquet(SALIDA / f"u2_c_puntajes_{a}.parquet", index=False)
        paso(f"corte {a}: guardado")
        del tr, te


def medir_c(anios, E):
    d = casos(anios, E)
    n0 = len(d)
    d = d.merge(pd.concat([pd.read_parquet(SALIDA / f"u2_c_puntajes_{a}.parquet") for a in anios]).astype(TIPOS), on=K)
    assert len(d) == n0, "la prueba no son los mismos casos"
    M = {"con ampliaciones": "s_P_ampl", "modelo por proveedor": "s_P_prov", "tamaño": "s_tamano_proveedor", "monto": "s_monto"}
    t = dos_columnas(d, "y_P", M)
    g = ic_prov_anio(d, "y_P", "s_P_ampl", "s_tamano_proveedor")
    return d, t, g, ic_prov_anio(d, "y_P", "s_P_ampl", "s_P_prov")


def prueba_c():
    assert not (SALIDA / "u2_c_penalidades.json").exists(), "se corre una sola vez"
    E = eventos()
    x = ampliaciones(contrato_a_ruc())
    puntajes_c(ELEGIR, x, E)
    d, t, g, gp = medir_c(ELEGIR, E)
    pasa = bool(g["razon"] >= 1.05 and g["ic90"][0] > 1)
    res = {"elegir": {"columnas": t, "frente_tamano": g, "frente_modelo": gp, "pasa": pasa}}
    paso(f"ELEGIR 2020 y 2021: frente al tamaño {g}; frente al modelo por proveedor {gp}; PASA {pasa}")
    for n, v in t.items():
        paso(f"   {n}: {v['prov_anio_acierto']} de {v['prov_anio_revisados']} ({v['precision_prov_anio']:.4f}); por caso {v['aciertos']} de {v['revisiones']}")
    if pasa:
        puntajes_c((2023,), x, E)
        d, t, g, gp = medir_c(PRUEBA, E)
        pa = {a: ic_prov_anio(d[d["anio"] == a], "y_P", "s_P_ampl", "s_tamano_proveedor", n=1)["razon"] for a in PRUEBA}
        ok = bool(g["razon"] >= 1.05 and sum(1 for v in pa.values() if v and v >= 1.05) >= 2 and g["ic90"][0] > 1)
        res["confirmar"] = {"columnas": t, "frente_tamano": g, "frente_modelo": gp, "por_anio": pa, "pasa": ok}
        paso(f"CONFIRMAR: frente al tamaño {g}; por año {pa}; frente al modelo {gp}; PASA {ok}")
    else:
        paso("no pasa a confirmar; queda el tamaño")
    guardar("c_penalidades", res)


# ------------------------------------------------------------------------------------------------ columnas adicionales
def extra():
    E = eventos()
    d = casos(PRUEBA, E)
    res = {}
    for nom, y, M in (("caso_1", "y_imp_real", {"alerta y monto": "s_alerta_monto", "monto": "s_monto", "tamaño": "s_tamano_proveedor"}),
                      ("caso_3", "y_S", {"modelo por proveedor": "s_S_prov", "monto": "s_monto", "tamaño": "s_tamano_proveedor"}),
                      ("caso_4", "y_P", {"tamaño": "s_tamano_proveedor", "modelo por proveedor": "s_P_prov", "monto": "s_monto"}),
                      ("caso_5", "y_rev", {"monto del caso": "s_monto", "tamaño": "s_tamano_proveedor"})):
        res[nom] = columnas_adicionales(d, y, M)
        for n, v in res[nom].items():
            paso(f"{nom} {n}: hallados {v['prov_anio_hallados']} de {v['prov_anio_positivos']} ({v['recall_prov_anio']:.3f}); revisiones por proveedor-año {v['repeticion']:.2f}")
    guardar("extra_columnas", res)


if __name__ == "__main__":
    {"b": prueba_b, "c": prueba_c, "d": prueba_d, "extra": extra}[sys.argv[1]]()
    paso("LISTO " + sys.argv[1])
