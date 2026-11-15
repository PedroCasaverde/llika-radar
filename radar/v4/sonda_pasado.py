"""Las dos dudas de Pedro (HALLAZGOS 160, sección 73): sacar a servicios básicos y telefonía también del ajuste, y el pasado con distinta importancia. Solo lectura, sin API y solo años de exploración.

    python -I -X utf8 sonda_pasado.py
Salida: resultados/por_caso/sonda_pasado_<año>.parquet (puntajes) y sonda_pasado.json. Solo agregados en el json.
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
import sonda_prompt_final as S  # noqa: E402
import sonda_prompt_final_2 as S2  # noqa: E402
import sonda_prompt_final_8 as S8  # noqa: E402
import sonda_tres_puntos as T  # noqa: E402
import sonda_tres_puntos_2 as T2  # noqa: E402
from casos import DIR, SALIDA, TIPOS, etiquetas_caso, eventos, top  # noqa: E402
from d1_linea_base import exposicion, monotonia  # noqa: E402
from d15_variables import NUEVAS, lista_121  # noqa: E402
from etiquetas import BASE, K, Hechos  # noqa: E402
from evaluacion import Transformador, armar_corte  # noqa: E402
from m6_proveedor import NUEVAS_P, modelo_proveedor, por_proveedor  # noqa: E402
sys.argv = _a

CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
GAMMAS_C = (0.0, 0.25, 0.5, 0.75, 1.0)
GAMMAS_H = (0.25, 0.5, 0.75, 1.0)
T0 = time.time()


def nombres():
    n = pd.read_parquet(BASE / "proveedores.parquet", columns=["ruc", "nombre"]).astype({"ruc": str}).drop_duplicates("ruc")
    return n.set_index("ruc")["nombre"]


def ganancias(tr, y, cols):
    """Un ajuste igual al del modelo por proveedor, para ver de dónde sale su ganancia."""
    t = tr.assign(_y=y)
    ptr = por_proveedor(t, cols).merge(t.groupby(["ruc", "anio"], sort=False)["_y"].max().reset_index(), on=["ruc", "anio"])
    yy = ptr["_y"].to_numpy(dtype=bool)
    m = LogisticRegression(C=1.0, max_iter=500).fit(exposicion(ptr), yy).decision_function(exposicion(ptr))
    X = Transformador(ptr, cols)(ptr)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, monotone_constraints=monotonia(X.columns), monotone_constraints_method="advanced", verbose=-1, num_threads=12, seed=0)
    b = lgb.train(params, lgb.Dataset(X, yy.astype(int), init_score=m, free_raw_data=False), num_boost_round=100)
    g = pd.Series(b.feature_importance("gain"), index=X.columns, dtype=float)
    hechos = g[g.index.str.startswith("h_prov_")].sum()
    actividad = g[g.index.isin(["p_anios_previos_activo", "p_nuevo", "p_historia_desconocida", "p_sin_anio_previo", "p_entidades_hasta_t", "p_contratos_hasta_t", "p_ordenes_hasta_t", "p_crecimiento_monto"])].sum()
    tot = g.sum()
    return {"hechos_pasados": float(hechos / tot), "actividad_pasada": float(actividad / tot), "actividad_del_anio": float((tot - hechos - actividad) / tot), "mayores": g.sort_values(ascending=False).head(10).round(0).to_dict()}


def parte_a():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    E = eventos()
    P = pd.read_parquet(DIR / "proveedor_anio.parquet").astype({"ruc": str, "anio": int})
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    nuevas = P[["ruc", "anio"] + NUEVAS_P]
    nom = nombres()
    info = {}
    for a, anios in CORTES.items():
        f_sal = SALIDA / f"sonda_pasado_{a}.parquet"
        f_inf = SALIDA / f"sonda_pasado_{a}.json"
        if f_sal.exists() and f_inf.exists():
            info[a] = json.loads(f_inf.read_text(encoding="utf-8"))
            S.paso(f"ETAPA corte {a} ya estaba")
            continue
        corte = f"{a}-12-31"
        tr, te = armar_corte(H, anios, a)
        tr, te = tr.reset_index(drop=True).astype({"ruc": str, "ruc_entidad": str, "anio": int}), te.reset_index(drop=True).astype({"ruc": str, "ruc_entidad": str, "anio": int})
        tr, te = tr.merge(nuevas, on=["ruc", "anio"], how="left"), te.merge(nuevas, on=["ruc", "anio"], how="left")
        v = etiquetas_caso(E, tr, corte, ventana=None)
        y = {"S": v["y_S"].to_numpy(), "P": v["y_P"].to_numpy()}
        f = (tr["u_servicio_basico"].astype(bool) | tr["u_proveedor_publico"].astype(bool) | tr["ruc"].map(nom).fillna("").str.upper().str.contains(T2.TELEFONIA, regex=True)).to_numpy()
        pa = tr[["ruc", "anio"]].assign(f=f, S=y["S"], P=y["P"]).groupby(["ruc", "anio"]).max()
        inf = {"proveedores_anio_ajuste": len(pa), "proveedores_anio_fuera": int(pa["f"].sum()), "positivos_S": int(pa["S"].sum()), "positivos_S_fuera": int((pa["S"] & pa["f"]).sum()), "positivos_P": int(pa["P"].sum()), "positivos_P_fuera": int((pa["P"] & pa["f"]).sum()), "casos_ajuste": len(tr), "casos_fuera": int(f.sum())}
        S.paso(f"ETAPA corte {a} armado: {inf}; {(time.time() - T0) / 60:.1f} min")
        d = te[K].copy()
        for tipo in ("S", "P"):
            d[f"s_{tipo}_base"] = modelo_proveedor(tr, y[tipo], te, cprov, (100,))[100][0]
            d[f"s_{tipo}_limpio"] = modelo_proveedor(tr[~f].reset_index(drop=True), y[tipo][~f], te, cprov, (100,))[100][0]
            inf[f"ganancia_{tipo}"] = ganancias(tr, y[tipo], cprov)
            S.paso(f"ETAPA corte {a}, {tipo} listo; {(time.time() - T0) / 60:.1f} min; {inf[f'ganancia_{tipo}']}")
        d.to_parquet(f_sal, index=False)
        f_inf.write_text(json.dumps(inf, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        info[a] = inf
        del tr, te, d
    return info


def pasado(q, gamma):
    """C_local con pasado, para las filas de q. Los montos del año son los de los casos; los de años anteriores, los de las órdenes."""
    if gamma == 0:
        return q["c_local"].to_numpy()
    o = ORDENES
    out = np.zeros(len(q))
    for t, g in q.groupby("anio"):
        x = o[o["anio"] < t].assign(w=lambda z: gamma ** (t - z["anio"]))
        pe = (x["monto"] * x["w"]).groupby([x["ruc_entidad"], x["ruc"]]).sum().rename("pe").reset_index()
        pp = (x["monto"] * x["w"]).groupby(x["ruc"]).sum().rename("pp").reset_index()
        m = g[["ruc_entidad", "ruc", "c_monto", "M_p"]].merge(pe, on=["ruc_entidad", "ruc"], how="left").merge(pp, on="ruc", how="left").fillna({"pe": 0.0, "pp": 0.0})
        out[q.index.get_indexer(g.index)] = ((m["c_monto"] + m["pe"]) / (m["M_p"] + m["pp"]).clip(lower=1e-9)).to_numpy()
    return out


def medir(q, col, nombre, res, G, cupo=True):
    L = pd.concat([(S8.con_cupo(g, col) if cupo else top(g, col)) for _, g in q.groupby("anio")])
    G[nombre] = L
    por = [S2.resumen(L[L["anio"] == a])["proveedores_anio_acierto"] for a in S.PRUEBA]
    res[nombre] = {"tres_anios": T.fila(L), "2023": T.fila(L[L["anio"] == 2023]), "por_anio": por, "ajuste_2020_2021": por[0] + por[1]}
    S.paso(f"{nombre}: por año {por}; " + T.texto("", res[nombre]["tres_anios"]))


def parte_b(info):
    global ORDENES
    d = T.preparar()
    nom = nombres()
    d["fuera"] = d["quitar"] | d["ruc"].map(nom).fillna("").str.upper().str.contains(T2.TELEFONIA, regex=True)
    sc = pd.concat([pd.read_parquet(SALIDA / f"sonda_pasado_{a}.parquet") for a in S.PRUEBA]).astype(TIPOS)
    d = d.merge(sc, on=K, how="left")
    res = {"info": info, "reproduce": {}, "listas": {}, "razones": {}}
    for tipo, col in (("S", "s_S_prov"), ("P", "s_P_prov")):
        dif = (d[f"s_{tipo}_base"] - d[col]).abs()
        res["reproduce"][tipo] = {"dif_max": float(dif.max()), "dif_p999": float(dif.quantile(0.999)), "faltan": int(d[f"s_{tipo}_base"].isna().sum())}
    S.paso(f"reproduce los puntajes guardados: {res['reproduce']}")
    q = d[~d["fuera"]].copy()
    q["M_p"] = d.groupby(["ruc", "anio"])["c_monto"].transform("sum").reindex(q.index)
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc_entidad", "ruc", "anio", "monto"]).astype({"ruc_entidad": str, "ruc": str, "anio": int})
    ORDENES = o[o["anio"] <= 2023].groupby(["ruc_entidad", "ruc", "anio"], as_index=False)["monto"].sum()
    q23 = q[q["anio"] == 2023][["ruc_entidad", "ruc", "c_monto"]].merge(ORDENES[ORDENES["anio"] == 2023], on=["ruc_entidad", "ruc"], how="left")
    S.paso(f"c_monto frente a la suma de órdenes en 2023: mediana {(q23['monto'] / q23['c_monto']).median():.4f}; iguales {np.isclose(q23['monto'], q23['c_monto'], rtol=1e-3).mean():.3f}; sin órdenes {q23['monto'].isna().mean():.4f}")
    q["R0"] = T.junta(T.sig(q["s_S_prov"]), T.sig(q["s_P_prov"]))
    q["R_limpio"] = T.junta(T.sig(q["s_S_limpio"]), T.sig(q["s_P_limpio"]))
    V = q["V"]
    G = {}
    L = res["listas"]
    q["s_base"] = 1e4 * V + q["R0"] * q["c_local"]
    medir(q, "s_base", "base", L, G)
    medir(q, "s_base", "base_sin_cupo", L, G, cupo=False)
    # Prueba 1
    q["s_limpio"] = 1e4 * V + q["R_limpio"] * q["c_local"]
    medir(q, "s_limpio", "sacados_del_ajuste", L, G)
    medir(q, "s_limpio", "sacados_del_ajuste_sin_cupo", L, G, cupo=False)
    S.paso(f"ETAPA prueba 1 lista; {(time.time() - T0) / 60:.1f} min")
    # Prueba 2: C_local con pasado
    for g_ in GAMMAS_C[1:]:
        q[f"c_{g_}"] = pasado(q, g_)
        q[f"s_c_{g_}"] = 1e4 * V + q["R0"] * q[f"c_{g_}"]
        medir(q, f"s_c_{g_}", f"c_local_pasado_{g_}", L, G)
    S.paso(f"ETAPA C_local con pasado lista; {(time.time() - T0) / 60:.1f} min")
    # Prueba 2: hechos con descuento
    E = eventos()
    e = E[E["fuente"].isin(["S", "P"])][["ruc", "anio_hechos", "fecha_conocido"]].astype({"ruc": str})
    q["R_tramo"] = np.select([q["hechos"] >= 1, q["p_ejec_contratos"] >= 1], [1.0, 0.1], 0.01)
    q["s_hechos"] = 1e4 * V + q["R_tramo"] * q["c_local"]
    medir(q, "s_hechos", "hechos", L, G)
    for g_ in GAMMAS_H:
        D = np.zeros(len(q))
        for t, g in q.groupby("anio"):
            x = e[(e["fecha_conocido"] <= pd.Timestamp(f"{t}-12-31")) & (e["anio_hechos"] <= t)]
            s = (g_ ** (t - x["anio_hechos"])).groupby(x["ruc"]).sum()
            D[q.index.get_indexer(g.index)] = g["ruc"].map(s).fillna(0.0).to_numpy()
        q["R_d"] = np.where(D > 0, 1 + D, q["R_tramo"])
        q[f"s_h_{g_}"] = 1e4 * V + q["R_d"] * q["c_local"]
        medir(q, f"s_h_{g_}", f"hechos_descuento_{g_}", L, G)
    S.paso(f"ETAPA hechos con descuento lista; {(time.time() - T0) / 60:.1f} min")
    B = G["base"]
    for n in [k for k in G if k != "base"]:
        res["razones"][n] = {"tres_anios": S8.ic_proveedores(G[n], B), "ajuste": S8.ic_proveedores(G[n][G[n]["anio"].isin(S.AJUSTE)], B[B["anio"].isin(S.AJUSTE)]), "2023": S8.ic_proveedores(G[n][G[n]["anio"] == 2023], B[B["anio"] == 2023])}
        S.paso(f"{n} frente a la base: {res['razones'][n]}")
    (SALIDA / "sonda_pasado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def main():
    info = parte_a()
    parte_b(info)
    S.paso(f"LISTA en {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
