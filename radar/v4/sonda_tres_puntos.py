"""Los puntos 2 y 3 de Pedro (HALLAZGOS 160, sección 72). Solo lectura, sin API y solo años de exploración: pesos y modelos que unen todo con 2020 y 2021; medida en 2023.

    python -I -X utf8 sonda_tres_puntos.py
Salida: resultados/por_caso/sonda_tres_puntos.json. Solo agregados; los nombres de empresas van únicamente al registro, fuera de git.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sonda_prompt_final as S  # noqa: E402
import sonda_prompt_final_2 as S2  # noqa: E402
import sonda_prompt_final_5 as S5  # noqa: E402
import sonda_prompt_final_8 as S8  # noqa: E402
from casos import DIR, SALIDA, TIPOS, top  # noqa: E402
from etiquetas import BASE, K  # noqa: E402
from evaluacion import X_PATH, columnas  # noqa: E402

OUT = BASE.parent / "resultados"
PA = ["h_prov_sanciones_otras", "h_prov_penalidades", "h_prov_multas", "h_prov_inhabilitaciones", "p_ejec_contratos"]
P2 = ["s_S_tasa", "s_P_tasa", "s_S_logit", "s_P_logit", "s_S_d1", "s_P_d1"]


def sig(s):
    return 1 / (1 + np.exp(-np.clip(np.asarray(s, dtype=float), -30, 30)))


def junta(a, b):
    return 1 - (1 - np.clip(a, 0, 1)) * (1 - np.clip(b, 0, 1))


def preparar():
    d = S.preparar().reset_index(drop=True)
    x = pd.read_parquet(X_PATH, columns=K + ["u_servicio_basico", "u_proveedor_publico"], filters=[("anio", "in", list(S.PRUEBA))]).astype(TIPOS)
    d = d.merge(x, on=K, how="left")
    p2 = pd.concat([pd.read_parquet(SALIDA / f"puntajes_p2_{a}.parquet", columns=K + P2) for a in S.PRUEBA]).astype(TIPOS)
    d = d.merge(p2, on=K, how="left")
    d18 = pd.concat([pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet", columns=K + ["s_D1_binario"]) for a in S.PRUEBA]).astype(TIPOS)
    d = d.merge(d18, on=K, how="left")
    pa = pd.read_parquet(DIR / "proveedor_anio.parquet", columns=["ruc", "anio"] + PA).astype({"ruc": str, "anio": int})
    pa = pa[pa["anio"].isin(S.PRUEBA)].drop_duplicates(["ruc", "anio"])
    d = d.merge(pa, on=["ruc", "anio"], how="left")
    d[PA] = d[PA].fillna(0)
    e = pd.read_parquet(BASE / "entidades.parquet", columns=["ruc_entidad", "tipo"]).astype({"ruc_entidad": str}).drop_duplicates("ruc_entidad")
    d = d.merge(e, on="ruc_entidad", how="left")
    d["tipo_cod"] = d["tipo"].fillna("otra").astype("category").cat.codes
    d["n_ent"] = d.groupby(["ruc", "anio"])["ruc_entidad"].transform("nunique")
    d["log_n_ent"] = np.log1p(d["n_ent"])
    d["quitar"] = d["u_servicio_basico"].fillna(False).astype(bool) | d["u_proveedor_publico"].fillna(False).astype(bool)
    d["quitar_100"] = d["quitar"] | (d["n_ent"] >= 100)
    d["hechos"] = d[PA[:4]].sum(axis=1)
    return d


def concentracion(L):
    h = L[L["y"]].groupby("ruc").size().sort_values(ascending=False)
    v = L.groupby("ruc").size().sort_values(ascending=False)
    return {"aciertos_de_la_que_mas_aporta": int(h.iloc[0]) if len(h) else 0, "aciertos_de_las_5": int(h.head(5).sum()), "filas_de_los_20_mas_repetidos": int(v.head(20).sum())}


def fila(L):
    r = S2.resumen(L)
    r.update(concentracion(L))
    return r


def texto(n, r):
    return (f"{n}: proveedores con señal {r['proveedores_anio_acierto']} de {r['mostrados']} filas ({r['proveedores_anio_acierto'] / r['mostrados']:.3%}); mostrados {r['proveedores_anio_mostrados']} ({r['precision_prov_anio']:.2%}); "
            f"por fila {r['aciertos_amplia']} ({r['precision_caso']:.2%}), la que más aporta {r['aciertos_de_la_que_mas_aporta']}, las 5 {r['aciertos_de_las_5']}; imp {r['prov_anio_impedimento']} S {r['prov_anio_sancion']} P {r['prov_anio_penalidad']} hall {r['prov_anio_hallazgo']}; del caso {r['aciertos_del_caso']}; ent-año con señal {r['entidades_anio_con_acierto']}")


def main():
    t0 = time.time()
    d = preparar()
    S.paso(f"ETAPA datos listos en {time.time() - t0:.0f} s. Casos por quitar {d['quitar'].mean():.4f}; proveedores-año {d[d['quitar']].groupby(['ruc', 'anio']).ngroups}; con 100 entidades o más {d['quitar_100'].mean():.4f}, proveedores-año {d[d['quitar_100']].groupby(['ruc', 'anio']).ngroups}")
    logM, cl, V = d["logM"], d["c_local"], d["V"]
    d["s_R_sin_c_local"] = 1e4 * V + logM * d["R"]
    d["s_final"] = 1e4 * V + logM * d["R"] * cl
    res = {"punto_2": {}, "punto_3": {}, "tiempos": {}, "razones": {}}

    # Punto 2: con y sin esas empresas
    listas2 = {"monto": "s_monto", "tamano": "s_tamano_proveedor", "alerta_y_monto": "s_alerta_monto", "mochila_sin_c_local": "s_R_sin_c_local", "final": "s_final"}
    guard = {}
    for esc, m in (("todas", pd.Series(True, index=d.index)), ("sin_basicos_ni_publicas", ~d["quitar"]), ("tambien_sin_100_entidades", ~d["quitar_100"])):
        res["punto_2"][esc] = {}
        q0 = d[m]
        for n, c in listas2.items():
            L3 = pd.concat([top(g, c) for _, g in q0.groupby("anio")])
            guard[(esc, n)] = L3
            res["punto_2"][esc][n] = {"tres_anios": fila(L3), "2023": fila(L3[L3["anio"] == 2023])}
            S.paso(f"[{esc}] tres años, " + texto(n, res["punto_2"][esc][n]["tres_anios"]))
            S.paso(f"[{esc}] 2023, " + texto(n, res["punto_2"][esc][n]["2023"]))
    nom = pd.read_parquet(BASE / "proveedores.parquet", columns=["ruc", "nombre"]).astype({"ruc": str}).drop_duplicates("ruc").set_index("ruc")["nombre"].str.slice(0, 44)
    for n in ("final", "tamano"):
        L = guard[("sin_basicos_ni_publicas", n)]
        L = L[L["anio"] == 2023]
        v = L.groupby("ruc").agg(filas=("y", "size"), aciertos=("y", "sum")).sort_values("filas", ascending=False).head(22)
        v["nombre"] = v.index.map(nom)
        S.paso(f"más repetidos sin básicos ni públicas, {n}, 2023 (solo registro):\n" + v.to_string())
    S.paso(f"ETAPA punto 2 listo en {time.time() - t0:.0f} s")

    # Punto 3: la escalera, con esas empresas fuera, vínculo primero y × log(1 + M) × C_local
    q = d[~d["quitar"]].copy()
    R = {
        "a_sin_modelo": np.ones(len(q)),
        "b_tamano": q.groupby("anio")["s_tamano_proveedor"].rank(pct=True).to_numpy(),
        "c_hechos": np.select([q["hechos"] >= 1, q["p_ejec_contratos"] >= 1], [1.0, 0.1], 0.01),
        "d_tasa": junta(q["s_S_tasa"], q["s_P_tasa"]),
        "e_logistica": junta(sig(q["s_S_logit"]), sig(q["s_P_logit"])),
        "f_modelo_proveedor": q["R"].to_numpy(),
        "g_D1_caso": sig(q["s_D1_binario"]),
    }
    for n, r in R.items():
        q[f"s_{n}"] = 1e4 * q["V"] + q["logM"] * r * q["c_local"]
    G = {}
    for n in R:
        L3 = pd.concat([top(g, f"s_{n}") for _, g in q.groupby("anio")])
        G[n] = L3
        res["punto_3"][n] = {"tres_anios": fila(L3), "2023": fila(L3[L3["anio"] == 2023]), "por_anio": [S2.resumen(L3[L3["anio"] == a])["proveedores_anio_acierto"] for a in S.PRUEBA]}
        S.paso(f"escalera, tres años, {res['punto_3'][n]['por_anio']}, " + texto(n, res["punto_3"][n]["tres_anios"]))
    for n in ("c_hechos", "f_modelo_proveedor"):
        L3 = pd.concat([S8.con_cupo(g, f"s_{n}") for _, g in q.groupby("anio")])
        G[f"{n}_con_cupo"] = L3
        res["punto_3"][f"{n}_con_cupo"] = {"tres_anios": fila(L3), "2023": fila(L3[L3["anio"] == 2023]), "por_anio": [S2.resumen(L3[L3["anio"] == a])["proveedores_anio_acierto"] for a in S.PRUEBA]}
        S.paso(f"escalera, tres años, {res['punto_3'][f'{n}_con_cupo']['por_anio']}, " + texto(f"{n}_con_cupo", res["punto_3"][f"{n}_con_cupo"]["tres_anios"]))
    S.paso(f"ETAPA escalera lista en {time.time() - t0:.0f} s")

    # Modelos que unen todo: ajuste con 2020 y 2021, medida en 2023
    tr = q[q["anio"].isin(S.AJUSTE)].copy()
    te = q[q["anio"] == 2023].copy()
    tr["w"] = 1 / tr.groupby(["ruc", "anio"])["y"].transform("size")
    pocas = ["V", "s_S_prov", "s_P_ampl", "logM", "c_local", "log_n_ent", "f30", "f_cerca", "tipo_cod"]
    base = dict(learning_rate=0.05, num_leaves=15, min_child_samples=200, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, verbose=-1)

    t1 = time.time()
    pred = np.zeros(len(te))
    for s in (0, 1, 2):
        ds = lgb.Dataset(tr[pocas].to_numpy(dtype=float), tr["y"].astype(int).to_numpy(), weight=tr["w"].to_numpy(), feature_name=pocas, free_raw_data=False)
        b = lgb.train({**base, "objective": "binary", "seed": s}, ds, num_boost_round=300)
        pred += b.predict(te[pocas].to_numpy(dtype=float)) / 3
    te["s_h_lgbm_pocas"] = pred
    res["tiempos"]["h_lgbm_pocas_s"] = time.time() - t1
    S.paso(f"ETAPA h lista en {time.time() - t1:.0f} s; ganancia {dict(zip(pocas, b.feature_importance('gain').round(0).tolist()))}")

    t1 = time.time()
    tg = tr[tr.groupby("grupo")["y"].transform("any")]
    # LightGBM admite hasta 10 000 filas por grupo: en los más grandes quedan todos los positivos y una muestra de negativos
    tg = tg.assign(_r=np.random.default_rng(0).random(len(tg))).sort_values(["grupo", "y", "_r"], ascending=[True, False, True])
    tg = tg[tg.groupby("grupo").cumcount() < 9000].sort_values("grupo", kind="stable")
    pred = np.zeros(len(te))
    for s in (0, 1, 2):
        ds = lgb.Dataset(tg[pocas].to_numpy(dtype=float), tg["y"].astype(int).to_numpy(), group=tg.groupby("grupo", sort=False).size().to_numpy(), feature_name=pocas, free_raw_data=False)
        b = lgb.train({**base, "objective": "lambdarank", "lambdarank_truncation_level": 5, "label_gain": [0, 1], "seed": s}, ds, num_boost_round=300)
        pred += b.predict(te[pocas].to_numpy(dtype=float)) / 3
    te["s_i_lambdarank"] = pred
    res["tiempos"]["i_lambdarank_s"] = time.time() - t1
    S.paso(f"ETAPA i lista en {time.time() - t1:.0f} s")

    t1 = time.time()
    cols = [c for c in columnas() if c not in K]
    import pyarrow.parquet as pq
    hay = set(pq.read_schema(X_PATH).names)
    cols = [c for c in cols if c in hay]
    X = pd.read_parquet(X_PATH, columns=K + cols, filters=[("anio", "in", list(S.PRUEBA))]).astype(TIPOS)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    X = X[K + num]
    X[num] = X[num].astype("float32")
    S.paso(f"variables del caso: {len(num)}; {X.memory_usage().sum() / 1e9:.1f} GB")
    extra = ["V", "s_S_prov", "s_P_ampl", "logM", "c_local", "log_n_ent", "f_cerca"]
    tx = tr[K + ["y", "w"] + extra].merge(X, on=K, how="left")
    ex = te[K + extra].merge(X, on=K, how="left")
    todas = extra + [c for c in num if c not in extra]
    pred = np.zeros(len(te))
    for s in (0, 1, 2):
        ds = lgb.Dataset(tx[todas].to_numpy(dtype="float32"), tx["y"].astype(int).to_numpy(), weight=tx["w"].to_numpy(), feature_name=todas, free_raw_data=False)
        b = lgb.train({**base, "objective": "binary", "seed": s}, ds, num_boost_round=300)
        pred += b.predict(ex[todas].to_numpy(dtype="float32")) / 3
    te["s_j_lgbm_todas"] = pred
    res["tiempos"]["j_lgbm_todas_s"] = time.time() - t1
    res["variables_j"] = len(todas)
    g = pd.Series(b.feature_importance("gain"), index=todas).sort_values(ascending=False).head(12)
    S.paso(f"ETAPA j lista en {time.time() - t1:.0f} s; {len(todas)} variables; mayor ganancia {g.round(0).to_dict()}")

    for n in ("h_lgbm_pocas", "i_lambdarank", "j_lgbm_todas"):
        te[f"s_{n}_alerta"] = 1e4 * te["V"] + te[f"s_{n}"]
        for c, et in ((f"s_{n}", n), (f"s_{n}_alerta", f"{n}_alerta_primero")):
            L = top(te, c)
            G[et] = L
            res["punto_3"][et] = {"2023": fila(L)}
            S.paso("une todo, 2023, " + texto(et, res["punto_3"][et]["2023"]))
    for n in ("a_sin_modelo", "b_tamano", "c_hechos", "d_tasa", "e_logistica", "f_modelo_proveedor", "g_D1_caso", "c_hechos_con_cupo", "f_modelo_proveedor_con_cupo"):
        S.paso("escalera, 2023, " + texto(n, res["punto_3"][n]["2023"]))

    # razones en 2023 y en tres años
    def de(n, a=2023):
        return G[n][G[n]["anio"] == a]
    M = {n: guard[("sin_basicos_ni_publicas", n)] for n in ("monto", "tamano", "alerta_y_monto")}
    for n in [k for k in G if k not in ("h_lgbm_pocas", "i_lambdarank", "j_lgbm_todas")]:
        A = G[n][G[n]["anio"] == 2023]
        fila_r = {}
        for v, B in M.items():
            fila_r[v] = S8.ic_proveedores(A, B[B["anio"] == 2023])
            if G[n]["anio"].nunique() == 3:
                fila_r[f"{v}_tres"] = S8.ic_proveedores(G[n], B)
        for v in ("c_hechos", "f_modelo_proveedor"):
            if v != n:
                fila_r[v] = S8.ic_proveedores(A, de(v))
        res["razones"][n] = fila_r
        S.paso(f"razones, {n}: " + json.dumps(fila_r, ensure_ascii=False))
    (SALIDA / "sonda_tres_puntos.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    S.paso(f"LISTA en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
