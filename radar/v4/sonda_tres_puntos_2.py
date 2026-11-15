"""Cierre del punto 3 de Pedro (HALLAZGOS 160, sección 72): ¿se puede simplificar más la lista sin perder? ¿Y lo complejo gana si entra en la misma fórmula? Solo lectura, sin API y solo años de exploración.

    python -I -X utf8 sonda_tres_puntos_2.py
Salida: resultados/por_caso/sonda_tres_puntos_2.json. Solo agregados.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sonda_prompt_final as S  # noqa: E402
import sonda_prompt_final_2 as S2  # noqa: E402
import sonda_prompt_final_8 as S8  # noqa: E402
import sonda_tres_puntos as T  # noqa: E402
from casos import SALIDA, TIPOS, top  # noqa: E402
from etiquetas import BASE, K  # noqa: E402
from evaluacion import X_PATH, columnas  # noqa: E402

TELEFONIA = r"WIN EMPRESAS|GTD PER|AMERICATEL|OPTICAL NETWORKS|VIETTEL|BITEL|AMERICA MOVIL|ENTEL PERU|TELEFONICA"


def main():
    d = T.preparar()
    n = pd.read_parquet(BASE / "proveedores.parquet", columns=["ruc", "nombre"]).astype({"ruc": str}).drop_duplicates("ruc")
    d = d.merge(n, on="ruc", how="left")
    d["telefonia"] = d["nombre"].fillna("").str.upper().str.contains(TELEFONIA, regex=True)
    S.paso(f"filas de telefonía e internet que la bandera no quita: {int((d['telefonia'] & ~d['quitar']).sum())}, de {d[d['telefonia'] & ~d['quitar']]['ruc'].nunique()} proveedores")
    q = d[~d["quitar"]].copy()
    q2 = d[~(d["quitar"] | d["telefonia"])].copy()
    pS, pP, pP0 = T.sig(q["s_S_prov"]), T.sig(q["s_P_ampl"]), T.sig(q["s_P_prov"])
    V, lm, cl = q["V"], q["logM"], q["c_local"]
    q["s_final"] = 1e4 * V + lm * T.junta(pS, pP) * cl
    q["s_sin_ampliaciones"] = 1e4 * V + lm * T.junta(pS, pP0) * cl
    q["s_solo_S"] = 1e4 * V + lm * pS * cl
    q["s_solo_P"] = 1e4 * V + lm * pP * cl
    q["s_sin_log"] = 1e4 * V + T.junta(pS, pP) * cl
    q["s_rangos"] = 1e4 * V + q.groupby("anio")["R"].rank(pct=True) * cl
    q2["s_final"] = 1e4 * q2["V"] + q2["logM"] * q2["R"] * q2["c_local"]
    res = {"tres_anios": {}, "2023": {}, "razones": {}}
    G = {}
    for n_, c in (("final", "s_final"), ("sin_ampliaciones", "s_sin_ampliaciones"), ("solo_S", "s_solo_S"), ("solo_P", "s_solo_P"), ("sin_log", "s_sin_log"), ("rangos", "s_rangos")):
        L = pd.concat([top(g, c) for _, g in q.groupby("anio")])
        G[n_] = L
        res["tres_anios"][n_] = T.fila(L)
        res["2023"][n_] = T.fila(L[L["anio"] == 2023])
        S.paso("tres años, " + T.texto(n_, res["tres_anios"][n_]))
    L = pd.concat([top(g, "s_final") for _, g in q2.groupby("anio")])
    G["final_sin_telefonia"] = L
    res["tres_anios"]["final_sin_telefonia"] = T.fila(L)
    S.paso("tres años, " + T.texto("final_sin_telefonia", res["tres_anios"]["final_sin_telefonia"]))
    # lo complejo dentro de la misma fórmula: ajuste con 2020 y 2021, medida en 2023
    tr, te = q[q["anio"].isin(S.AJUSTE)].copy(), q[q["anio"] == 2023].copy()
    tr["w"] = 1 / tr.groupby(["ruc", "anio"])["y"].transform("size")
    pocas = ["s_S_prov", "s_P_ampl", "logM", "log_n_ent", "f30", "f_cerca", "tipo_cod"]
    base = dict(learning_rate=0.05, num_leaves=15, min_child_samples=200, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, verbose=-1)
    pred = np.zeros(len(te))
    for s in (0, 1, 2):
        ds = lgb.Dataset(tr[pocas].to_numpy(dtype=float), tr["y"].astype(int).to_numpy(), weight=tr["w"].to_numpy(), feature_name=pocas, free_raw_data=False)
        pred += lgb.train({**base, "objective": "binary", "seed": s}, ds, num_boost_round=300).predict(te[pocas].to_numpy(dtype=float)) / 3
    te["s_h_formula"] = 1e4 * te["V"] + te["logM"] * pred * te["c_local"]
    cols = [c for c in columnas() if c not in K and c in set(pq.read_schema(X_PATH).names)]
    X = pd.read_parquet(X_PATH, columns=K + cols, filters=[("anio", "in", list(S.PRUEBA))]).astype(TIPOS)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    X = X[K + num]
    X[num] = X[num].astype("float32")
    extra = ["s_S_prov", "s_P_ampl", "logM", "log_n_ent", "f_cerca"]
    tx, ex = tr[K + ["y", "w"] + extra].merge(X, on=K, how="left"), te[K + extra].merge(X, on=K, how="left")
    todas = extra + [c for c in num if c not in extra]
    pred = np.zeros(len(te))
    for s in (0, 1, 2):
        ds = lgb.Dataset(tx[todas].to_numpy(dtype="float32"), tx["y"].astype(int).to_numpy(), weight=tx["w"].to_numpy(), feature_name=todas, free_raw_data=False)
        pred += lgb.train({**base, "objective": "binary", "seed": s}, ds, num_boost_round=300).predict(ex[todas].to_numpy(dtype="float32")) / 3
    te["s_j_formula"] = 1e4 * te["V"] + te["logM"] * pred * te["c_local"]
    F = G["final"][G["final"]["anio"] == 2023]
    for n_, c in (("h_en_la_formula", "s_h_formula"), ("j_en_la_formula", "s_j_formula")):
        L = top(te, c)
        res["2023"][n_] = T.fila(L)
        res["razones"][n_] = S8.ic_proveedores(L, F)
        S.paso("2023, " + T.texto(n_, res["2023"][n_]) + f"; frente a la final {res['razones'][n_]}")
    for n_ in ("sin_ampliaciones", "solo_S", "solo_P", "sin_log", "rangos", "final_sin_telefonia"):
        res["razones"][n_] = {"tres_anios": S8.ic_proveedores(G[n_], G["final"]), "2023": S8.ic_proveedores(G[n_][G[n_]["anio"] == 2023], F)}
        S.paso(f"{n_} frente a la final: {res['razones'][n_]}")
    S.paso("2023, " + T.texto("final", res["2023"]["final"]))
    (SALIDA / "sonda_tres_puntos_2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    S.paso("LISTA")


if __name__ == "__main__":
    main()
