"""Qué pesa más en el «parecido» R de la lista de 2025 (HALLAZGOS 160, sección 91; Pedro, 13/11: «calcúlalo»). Fijado antes de correr.

Solo lectura: sin API y sin etiquetas de 2025. No modifica lista_unica.py ni lista_unica_limpia.py; los usa tal cual y solo
reemplaza, en memoria, la función que ajusta los modelos por otra idéntica que además guarda los modelos ajustados.

    1. Rehace los dos modelos de la lista congelada de 2025 (ajuste 2018 a 2023, corte 31/12/2025) y arma la lista.
       Se detiene si no repite la congelada anterior sin la limpieza y la vigente con ella.
    2. Medida principal: baraja entre proveedores todos los datos de una familia a la vez, recalcula R con los mismos
       modelos y mide cuánto cambia el R de cada proveedor. Cinco repeticiones (semillas 0 a 4).
    3. De apoyo, con la misma baraja: qué parte del 2 % más alto deja de estarlo y cuántas filas de la lista cambian.
    4. Contraste: aporte de cada dato al puntaje de cada proveedor (contribuciones de los árboles y del margen de exposición).
    5. Perfil: cómo son los proveedores del 2 % más alto frente al resto, en datos que cualquiera entiende.

    python -I -X utf8 importancia_parecido.py
Salida, fuera de git: resultados/lista_unica_limpia/importancia_parecido.json y .md. No guarda RUC ni nombres.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import lista_unica as LU  # noqa: E402
import lista_unica_limpia as LL  # noqa: E402
import m6_proveedor as M6  # noqa: E402
import p1_tablas as P1  # noqa: E402
from casos import DIR  # noqa: E402
from d1_linea_base import exposicion, monotonia  # noqa: E402
from etiquetas import Hechos  # noqa: E402
from evaluacion import Transformador  # noqa: E402

sys.argv = _a

A = LU.ANIO_LISTA
REPETICIONES = 5
ARBOLES = 100
ALTO = LU.MOCHILA_ALTA
EXPO = ["c_monto", "c_ordenes"]          # la exposición: entra al margen, no a los árboles

PENALIDADES = ["h_prov_penalidades", "h_prov_pen_mora", "h_prov_pen_otra", "h_prov_pen_monto", "h_prov_pen_entidades", "h_prov_pen_reciente", "h_prov_cred_pen"]
SANCIONES = ["h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones", "h_prov_inhabilitado_al_cierre", "h_prov_impedimento", "h_prov_san_falsa", "h_prov_san_resolver",
             "h_prov_san_impedido", "h_prov_san_definitiva", "h_prov_san_meses", "h_prov_jud", "h_prov_jud_activa", "h_prov_cred_san"]
HECHOS_TOTAL = ["h_prov_hechos", "h_prov_entidades"]
FAMILIAS = {
    "Antecedentes": PENALIDADES + SANCIONES + HECHOS_TOTAL,
    "Cuánto vende sin concurso": EXPO + ["p_ordenes", "p_monto", "p_entidades", "p_entidades_hasta_t", "p_ordenes_hasta_t"],
    "Concursos y contratos": ["p_exp_ocds_procesos", "p_exp_ocds_entidades", "p_exp_ocds_ganados", "p_exp_ocds_monto", "p_ejec_contratos", "p_ejec_monto", "p_contratos_hasta_t"],
    "Cómo le va en los concursos": ["p_ocds_n_pares_persistentes", "p_ocds_n_pares_rotan", "p_ocds_postor_compania", "p_ocds_postula_inhabilitado", "p_ocds_gana_inhabilitado", "p_ocds_monto_gana_inhabilitado",
                                    "p_ocds_entidades_perdio_y_orden", "p_ocds_ordenes_tras_perder", "p_ocds_monto_tras_perder", "p_ocds_ganados_postor_unico"],
    "Antigüedad y crecimiento": ["p_anios_previos_activo", "p_nuevo", "p_historia_desconocida", "p_crecimiento_monto", "p_sin_anio_previo"],
    "Compras por catálogo": ["p_cat_ordenes", "p_cat_monto", "p_cat_entidades"],
    "Qué vende": ["p_parte_servicios", "p_n_rubros"],
}
DENTRO_DE_ANTECEDENTES = {"Penalidades": PENALIDADES, "Sanciones y prohibiciones": SANCIONES, "Hechos previos en total": HECHOS_TOTAL}
AJUSTES = []


def instrumentado(tr, y, te, cols, arboles, semillas=range(10)):
    """Lo mismo que m6_proveedor.modelo_proveedor, línea por línea, pero guarda lo ajustado."""
    assert tuple(arboles) == (ARBOLES,)
    t = tr.assign(_y=y)
    ptr = M6.por_proveedor(t, cols).merge(t.groupby(["ruc", "anio"], sort=False)["_y"].max().reset_index(), on=["ruc", "anio"])
    pte = M6.por_proveedor(te, cols)
    yy = ptr["_y"].to_numpy(dtype=bool)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(exposicion(ptr), yy)
    m_tr, m_te = lr.decision_function(exposicion(ptr)), lr.decision_function(exposicion(pte))
    T = Transformador(ptr, cols)
    Xtr, Xte = T(ptr), T(pte)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                  monotone_constraints=monotonia(Xtr.columns), monotone_constraints_method="advanced", verbose=-1, num_threads=12)
    ds = lgb.Dataset(Xtr, yy.astype(int), init_score=m_tr, free_raw_data=False)
    arbs = [lgb.train({**params, "seed": s}, ds, num_boost_round=ARBOLES) for s in semillas]
    exc = np.mean([b.predict(Xte, raw_score=True, num_iteration=ARBOLES) for b in arbs], axis=0)
    AJUSTES.append({"lr": lr, "T": T, "arboles": arbs, "pte": pte, "media_expo": exposicion(ptr).mean(axis=0), "positivos": int(yy.sum()), "filas_ajuste": len(ptr), "columnas": list(Xtr.columns)})
    llave = pd.MultiIndex.from_frame(pte[["ruc", "anio"]])
    pos = llave.get_indexer(pd.MultiIndex.from_frame(te[["ruc", "anio"]]))
    return {ARBOLES: ((m_te + exc)[pos], exc[pos])}


def puntaje(aj, p):
    X = aj["T"](p)
    return aj["lr"].decision_function(exposicion(p)) + np.mean([b.predict(X, raw_score=True, num_iteration=ARBOLES) for b in aj["arboles"]], axis=0)


def parecido(p):
    s, q = LU.sig(puntaje(AJUSTES[0], p)), LU.sig(puntaje(AJUSTES[1], p))
    return 1 - (1 - s) * (1 - q)


def lista_con(ql, rucs, R):
    """La lista limpia de 2025 con otro R: misma regla, mismas funciones."""
    r = ql["ruc"].map(pd.Series(R, index=rucs)).to_numpy()
    z = ql.assign(R=r)
    z["puntaje"] = z["R"] * z["logM"] * z["c_local"]
    z["s_lista"] = 1e4 * (z["grado"] >= 2) + z["puntaje"]
    L = LU.lista(z)
    return set(zip(L["ruc_entidad"], L["ruc"]))


def barajar(p, cols, m, rep):
    """Las columnas de la familia cambian de proveedor, todas con el mismo orden; solo entre los proveedores de la lista."""
    orden = np.random.default_rng(rep).permutation(int(m.sum()))
    z = p.copy()
    filas = np.flatnonzero(m)
    for c in cols:
        v = p[c].to_numpy()
        w = v.copy()
        w[filas] = v[filas][orden]
        z[c] = w
    return z


def medir(p, m, R0, pct0, ql, rucs, base, cols, con_lista=True):
    filas = []
    for rep in range(REPETICIONES):
        R = parecido(barajar(p, cols, m, rep))
        pct = pd.Series(R[m]).rank(pct=True).to_numpy()
        f = {"cambio_R": float(np.abs(R[m] - R0[m]).mean()), "sale_del_2pct": float((pct[pct0 >= ALTO] < ALTO).mean())}
        if con_lista:
            f["filas_que_cambian"] = len(base - lista_con(ql, rucs, R))
        filas.append(f)
    r = {k: float(np.mean([f[k] for f in filas])) for k in filas[0]}
    r["rango_cambio_R"] = [min(f["cambio_R"] for f in filas), max(f["cambio_R"] for f in filas)]
    return r


def contribuciones(aj, p, m):
    X = aj["T"](p)
    c = np.zeros((len(X), X.shape[1] + 1))
    for b in aj["arboles"]:
        c += b.predict(X, pred_contrib=True, num_iteration=ARBOLES)
    c = c[:, :-1] / len(aj["arboles"])
    e = (exposicion(p) - aj["media_expo"]) * aj["lr"].coef_[0]
    d = pd.DataFrame(np.column_stack([c, e]), columns=list(X.columns) + EXPO)
    return d[m]


def perfil(p, m, pct0):
    z = p[m].reset_index(drop=True)
    alto = pct0 >= ALTO
    ant = z[PENALIDADES + SANCIONES + HECHOS_TOTAL].fillna(0)
    datos = {
        "con algún antecedente (%)": 100 * (ant[["h_prov_hechos", "h_prov_penalidades", "h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones"]].sum(axis=1) > 0),
        "con alguna penalidad (%)": 100 * (ant["h_prov_penalidades"] > 0),
        "con alguna sanción, multa o prohibición (%)": 100 * (ant[["h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones"]].sum(axis=1) > 0),
        "participó en algún concurso (%)": 100 * (z["p_exp_ocds_procesos"].fillna(0) > 0),
        "ganó algún concurso sin competidores (%)": 100 * (z["p_ocds_ganados_postor_unico"].fillna(0) > 0),
        "es su primer año (%)": 100 * (z["p_nuevo"].fillna(0) > 0),
        "le vende a 3 entidades o más (%)": 100 * (z["p_entidades"].fillna(0) >= 3),
    }
    r = {k: {"alto": round(float(v[alto].mean()), 1), "resto": round(float(v[~alto].mean()), 1)} for k, v in datos.items()}
    r["vendió sin concurso en el año, mediana (S/)"] = {"alto": round(float(z.loc[alto, "c_monto"].median())), "resto": round(float(z.loc[~alto, "c_monto"].median()))}
    r["compras sin concurso en el año, mediana"] = {"alto": float(z.loc[alto, "c_ordenes"].median()), "resto": float(z.loc[~alto, "c_ordenes"].median())}
    r["proveedores"] = {"alto": int(alto.sum()), "resto": int((~alto).sum())}
    return r


def main():
    sal = LL.SAL / "importancia_parecido.json"
    LU.modelo_proveedor = instrumentado
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F_ = M6.Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    cprov, n = LU.comunes()
    assert sorted(c for f in FAMILIAS.values() for c in f) == sorted(cprov + EXPO), "las familias no cubren los 54 datos y la exposición, una vez cada uno"
    Pa = P1.tabla_proveedor(H, F_, LU.AJUSTE_LISTA + [A]).astype({"ruc": str, "anio": int})
    LU.paso("ETAPA 1: tabla de proveedores lista")
    d = LU.armar_anio(A, LU.AJUSTE_LISTA, H, E, Pa, cprov, n, LU.CORTE_LISTA, con_etiquetas=False)
    assert not any(c.startswith(("y", "fuentes", "conocido", "peso")) for c in d.columns), "no se leen etiquetas de 2025"
    assert len(AJUSTES) == 2 and all(len(a["columnas"]) == 54 for a in AJUSTES)
    q0 = LU.preparar_lista(d, A)
    del d, Pa
    s0 = LL.igual_a_la_guardada(LU.ficha(LU.lista(LL.sin_cambios(q0)).copy(), A, n, con_etiquetas=False), LL.VIEJA / "lista_2025_unica.parquet")
    ql = LL.limpio(q0)
    L = LU.lista(ql)
    Fi = LU.ficha(L.copy(), A, n, con_etiquetas=False)
    s1 = LL.igual_a_la_guardada(Fi, LL.SAL / "lista_2025_unica.parquet")
    G = pd.read_parquet(LL.SAL / "lista_2025_unica.parquet", columns=["ruc_entidad", "ruc", "percentil_historial"]).astype({"ruc_entidad": str, "ruc": str})
    dif_pct = float((Fi[["ruc_entidad", "ruc", "percentil_historial"]].merge(G, on=["ruc_entidad", "ruc"], suffixes=("", "_g")).eval("percentil_historial - percentil_historial_g").abs().max()))
    LU.paso(f"ETAPA 2: sin la limpieza {s0}; con ella {s1}; diferencia máxima de percentil {dif_pct:.2e}")
    for s in (s0, s1):
        assert s["llaves_distintas"] == 0 and s["motivos_distintos"] == 0 and s["ordenes_distintas"] == 0, "no repite la lista congelada: se detiene antes de medir"
    assert dif_pct < 1e-9, "los percentiles no son los de la lista congelada"

    p = AJUSTES[0]["pte"]
    assert p[["ruc", "anio"]].equals(AJUSTES[1]["pte"][["ruc", "anio"]]) and p["ruc"].is_unique
    rucs = p["ruc"].to_numpy()
    m = p["ruc"].isin(set(q0["ruc"])).to_numpy()
    R0 = parecido(p)
    u = q0.drop_duplicates("ruc").set_index("ruc")["R"]
    dif_R = float(np.abs(pd.Series(R0[m], index=rucs[m]) - u.reindex(rucs[m])).max())
    assert dif_R < 1e-9, f"el R recalculado no es el de la lista: {dif_R}"
    pct0 = pd.Series(R0[m]).rank(pct=True).to_numpy()
    base = set(zip(L["ruc_entidad"], L["ruc"]))
    assert lista_con(ql, rucs, R0) == base, "la lista rehecha con el mismo R no es la lista"
    LU.paso(f"ETAPA 3: {int(m.sum()):,} proveedores de 2025 en la lista; R repite con diferencia {dif_R:.1e}; {int((pct0 >= ALTO).sum()):,} en el 2 % más alto; lista de {len(base):,} filas")

    fam = {}
    for k, cols in FAMILIAS.items():
        fam[k] = {"datos": len(cols), **medir(p, m, R0, pct0, ql, rucs, base, cols)}
        LU.paso(f"ETAPA 4, familia «{k}»: {fam[k]}")
    total = sum(f["cambio_R"] for f in fam.values())
    for f in fam.values():
        f["peso_pct"] = round(100 * f["cambio_R"] / total, 1)
        f["filas_que_cambian_pct"] = round(100 * f["filas_que_cambian"] / len(base), 1)
    dentro = {}
    for k, cols in DENTRO_DE_ANTECEDENTES.items():
        dentro[k] = {"datos": len(cols), **medir(p, m, R0, pct0, ql, rucs, base, cols, con_lista=False)}
        LU.paso(f"ETAPA 5, dentro de Antecedentes, «{k}»: {dentro[k]}")
    tot_d = sum(f["cambio_R"] for f in dentro.values())
    for f in dentro.values():
        f["parte_pct"] = round(100 * f["cambio_R"] / tot_d, 1)

    contraste, sueltos = {}, {}
    for nombre, aj in (("sanción", AJUSTES[0]), ("penalidad", AJUSTES[1])):
        c = contribuciones(aj, p, m)
        porf = {k: float(c[cols].sum(axis=1).abs().mean()) for k, cols in FAMILIAS.items()}
        t = sum(porf.values())
        contraste[nombre] = {k: round(100 * v / t, 1) for k, v in porf.items()}
        sueltos[nombre] = {k: round(float(v), 4) for k, v in c.abs().mean().sort_values(ascending=False).head(12).items()}
        LU.paso(f"ETAPA 6, contribuciones del modelo de {nombre}: {contraste[nombre]}")
        del c
    orden = sorted(fam, key=lambda k: -fam[k]["cambio_R"])
    coincide = {nombre: sorted(v, key=lambda k: -v[k])[:3] == orden[:3] for nombre, v in contraste.items()}
    r = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "anio": A, "ajuste": LU.AJUSTE_LISTA, "corte": LU.CORTE_LISTA, "repeticiones": REPETICIONES,
         "comprobado": {"sin_la_limpieza": s0, "con_la_limpieza": s1, "diferencia_maxima_de_percentil": dif_pct, "diferencia_maxima_de_R": dif_R},
         "modelos": {nombre: {"proveedores_año_de_ajuste": aj["filas_ajuste"], "positivos": aj["positivos"], "coeficientes_de_exposicion": [float(x) for x in aj["lr"].coef_[0]]} for nombre, aj in (("sanción", AJUSTES[0]), ("penalidad", AJUSTES[1]))},
         "proveedores": int(m.sum()), "en_el_2pct_mas_alto": int((pct0 >= ALTO).sum()), "filas_de_la_lista": len(base), "R_mediano": float(np.median(R0[m])), "R_en_el_corte_del_2pct": float(np.quantile(R0[m], ALTO)),
         "orden": orden, "familias": fam, "dentro_de_antecedentes": dentro, "contraste_por_contribuciones_pct": contraste, "coinciden_las_tres_primeras": coincide,
         "datos_sueltos_que_mas_aportan": sueltos, "perfil_del_2pct_mas_alto_frente_al_resto": perfil(p, m, pct0), "variables_por_familia": FAMILIAS,
         "huellas": {**LL.huellas(), "importancia_parecido.py": LU.huella(__file__)[:16]}}
    sal.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    md = ["# Qué pesa más en el parecido R (lista de 2025; sin etiquetas)", "", f"{r['proveedores']:,} proveedores; {r['en_el_2pct_mas_alto']:,} en el 2 % más alto; lista de {len(base):,} filas. Cinco barajas por familia.", "",
          "| Familia | Datos | Peso (%) | Sale del 2 % más alto (%) | Filas de la lista que cambian (%) | Contribuciones, sanción (%) | Contribuciones, penalidad (%) |", "|---|---|---|---|---|---|---|"]
    for k in orden:
        f = fam[k]
        md.append(f"| {k} | {f['datos']} | {f['peso_pct']} | {100 * f['sale_del_2pct']:.1f} | {f['filas_que_cambian_pct']} | {contraste['sanción'][k]} | {contraste['penalidad'][k]} |")
    md += ["", "Dentro de Antecedentes: " + "; ".join(f"{k} {v['parte_pct']} %" for k, v in dentro.items()) + ".", "", "Perfil del 2 % más alto frente al resto:", ""]
    md += [f"- {k}: {v['alto']} frente a {v['resto']}" for k, v in r["perfil_del_2pct_mas_alto_frente_al_resto"].items()]
    (LL.SAL / "importancia_parecido.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    print(json.dumps({k: r[k] for k in ("coinciden_las_tres_primeras", "datos_sueltos_que_mas_aportan", "modelos")}, ensure_ascii=False, indent=1, default=float), flush=True)
    LU.paso("FIN importancia")


if __name__ == "__main__":
    main()
