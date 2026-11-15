"""Última parte de la sonda de la sección 71: la precisión en 5 contando una vez a cada proveedor por año, con IC. Solo lectura, sin API y solo años de exploración.

Medida: proveedores-año distintos con señal adversa / filas mostradas en el Top 5, sobre todas las entidades-año. Las filas son el esfuerzo de quien lee; repetir a un proveedor no suma, y mostrar menos proveedores distintos no reduce el denominador.

    python -I -X utf8 sonda_prompt_final_5.py
Salida: resultados/por_caso/sonda_prompt_final_5.json. Solo agregados.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sonda_prompt_final as S  # noqa: E402
import sonda_prompt_final_2 as S2  # noqa: E402
from casos import SALIDA, top  # noqa: E402


def medida(L, y="y"):
    return {"filas": len(L), "proveedores_con_senal": int(L[L[y]].groupby(["ruc", "anio"]).ngroups), "aciertos_por_fila": int(L[y].sum())}


def ic(A, B, y="y", n=1000, semilla=0):
    """Razón de la medida, A frente a B, con pesos de Poisson de entidades y proveedores. Una fila pesa según su entidad; un proveedor-año cuenta si alguna de sus filas sigue activa."""
    e, ne = pd.factorize(pd.concat([A["ruc_entidad"], B["ruc_entidad"]]))
    p, np_ = pd.factorize(pd.concat([A["ruc"], B["ruc"]]))
    py, npy = pd.factorize(pd.concat([A["ruc"] + "|" + A["anio"].astype(str), B["ruc"] + "|" + B["anio"].astype(str)]))
    c = len(A)
    prov_de = np.zeros(len(npy), dtype=int)
    prov_de[py] = p
    hit = np.r_[A[y].to_numpy(), B[y].to_numpy()].astype(bool)
    rng = np.random.default_rng(semilla)

    def med(lo, hi, we, wp):
        act = we[e[lo:hi]] > 0
        ac = np.zeros(len(npy), bool)
        ac[py[lo:hi][act & hit[lo:hi]]] = True
        return (wp[prov_de] * ac).sum() / max(we[e[lo:hi]].sum(), 1e-12)

    u_e, u_p = np.ones(len(ne), dtype=int), np.ones(len(np_), dtype=int)
    raz = med(0, c, u_e, u_p) / med(c, len(py), u_e, u_p)
    bs = []
    for _ in range(n):
        we, wp = rng.poisson(1.0, len(ne)), rng.poisson(1.0, len(np_))
        d0 = med(c, len(py), we, wp)
        bs.append(med(0, c, we, wp) / d0 if d0 > 0 else np.nan)
    return {"razon": float(raz), "ic90": [float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))]}


def main():
    d = S.preparar().reset_index(drop=True)
    d["x1"], d["x2"], d["x3"] = d["logM"] * d["V"], d["logM"] * d["R"], d["logM"] * d["compra"]
    fit = d[d["anio"].isin(S.AJUSTE)]
    _, w, _ = S.logistica(fit, ["x1", "x2", "x3"], "y")
    _, w4, _ = S.logistica(fit[fit["V"] == 0], ["logM", "x2"], "y")
    d["s_prompt"] = w["x1"] * d["x1"] + w["x2"] * d["x2"] + w["x3"] * d["x3"]
    d["s_prompt_alerta"] = 1e4 * d["V"] + w["x2"] * d["x2"] + w["x3"] * d["x3"]
    d["s_simple"] = 1e4 * d["V"] + w4["x2"] * d["x2"] + w4["logM"] * d["logM"]
    d["s_alerta_RM"] = 1e4 * d["V"] + d["x2"]
    listas = {"monto": "s_monto", "tamano": "s_tamano_proveedor", "alerta_y_monto": "s_alerta_monto", "modelo_S": "s_S_prov", "modelo_P": "s_P_ampl", "prompt": "s_prompt", "prompt_alerta_primero": "s_prompt_alerta", "simple": "s_simple", "alerta_R_por_M": "s_alerta_RM"}
    G = {}
    res = {"por_anio": {}, "tres_anios": {}, "por_verdad_tres_anios": {}, "razones": {}}
    for a, q in d.groupby("anio"):
        a = int(a)
        res["por_anio"][a] = {}
        for n, c in listas.items():
            G[(a, n)] = top(q, c)
            res["por_anio"][a][n] = medida(G[(a, n)])
        G[(a, "cupos")] = S2.cupos(q)
        res["por_anio"][a]["cupos"] = medida(G[(a, "cupos")])
        az = []
        for s in range(10):
            az.append(medida(top(q.assign(_a=np.random.default_rng(s).random(len(q))), "_a", semilla=s)))
        res["por_anio"][a]["azar"] = {k: float(np.mean([x[k] for x in az])) for k in az[0]}
    for n in [*listas, "cupos"]:
        T = pd.concat([G[(a, n)] for a in S.PRUEBA])
        res["tres_anios"][n] = medida(T)
        res["por_verdad_tres_anios"][n] = {v: int(T[T[y]].groupby(["ruc", "anio"]).ngroups) for v, y in (("impedimento", "y_imp_real"), ("sancion", "y_S"), ("penalidad", "y_P"), ("hallazgo", "y_rev"))}
        r = res["tres_anios"][n]
        S.paso(f"tres años, {n}: {r['proveedores_con_senal']} de {r['filas']} filas = {r['proveedores_con_senal'] / r['filas']:.4%}; por fila {r['aciertos_por_fila']}; por año {[res['por_anio'][a][n]['proveedores_con_senal'] for a in S.PRUEBA]}; por verdad {res['por_verdad_tres_anios'][n]}")
    S.paso(f"azar por año: {[res['por_anio'][a]['azar']['proveedores_con_senal'] for a in S.PRUEBA]}")
    for n, v in (("cupos", "monto"), ("cupos", "tamano"), ("cupos", "alerta_y_monto"), ("cupos", "prompt"), ("alerta_R_por_M", "monto"), ("alerta_R_por_M", "tamano"), ("prompt", "monto"), ("prompt", "tamano"), ("prompt_alerta_primero", "monto"), ("alerta_y_monto", "monto"), ("alerta_y_monto", "tamano"), ("modelo_P", "monto")):
        A, B = pd.concat([G[(a, n)] for a in S.PRUEBA]), pd.concat([G[(a, v)] for a in S.PRUEBA])
        res["razones"][f"{n}_frente_a_{v}"] = {"tres_anios": ic(A, B), "2023": ic(G[(2023, n)], G[(2023, v)])}
        S.paso(f"{n} frente a {v}: {res['razones'][f'{n}_frente_a_{v}']}")
    (SALIDA / "sonda_prompt_final_5.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    S.paso("LISTA")


if __name__ == "__main__":
    main()
