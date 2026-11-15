"""Segunda parte de la sonda de la sección 71: una lista por cupos, las cifras por año, por proveedor-año y a quién muestran las listas. Solo lectura, sin API y solo años de exploración.

    python -I -X utf8 sonda_prompt_final_2.py
Salida: resultados/por_caso/sonda_prompt_final_2.json. Solo agregados; los nombres de las empresas más repetidas van únicamente al registro, fuera de git.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sonda_prompt_final as S  # noqa: E402
from casos import SALIDA, top  # noqa: E402
from etiquetas import BASE  # noqa: E402


def cupos(q):
    """Entidad-año por entidad-año: primero las alertas, por monto; los puestos que quedan se alternan entre el mayor riesgo del proveedor y el mayor monto, sin repetir. Sin pesos ni regresión."""
    filas = []
    for _, g in q.groupby("grupo", sort=False):
        if len(g) <= 5:
            filas.extend(g.index.tolist())
            continue
        elegidos = g[g["V"] > 0].sort_values("c_monto", ascending=False).index[:5].tolist()
        por_r = g.sort_values("logit_R", ascending=False).index[:11].tolist()
        por_m = g.sort_values("c_monto", ascending=False).index[:11].tolist()
        i, j, turno = 0, 0, 0
        while len(elegidos) < 5:
            if turno % 2 == 0:
                while por_r[i] in elegidos:
                    i += 1
                elegidos.append(por_r[i])
            else:
                while por_m[j] in elegidos:
                    j += 1
                elegidos.append(por_m[j])
            turno += 1
        filas.extend(elegidos)
    return q.loc[filas]


def resumen(L):
    r = {"mostrados": len(L), "entidades_anio": int(L["grupo"].nunique()), "proveedores_anio_mostrados": int(L.groupby(["ruc", "anio"]).ngroups), "proveedores_anio_acierto": int(L[L["y"]].groupby(["ruc", "anio"]).ngroups), "entidades_anio_con_acierto": int(L[L["y"]]["grupo"].nunique())}
    for n, y in S.VERDADES.items():
        r[f"aciertos_{n}"] = int(L[y].sum())
    r["precision_caso"] = r["aciertos_amplia"] / max(r["mostrados"], 1)
    r["precision_prov_anio"] = r["proveedores_anio_acierto"] / max(r["proveedores_anio_mostrados"], 1)
    for n, y in (("impedimento", "y_imp_real"), ("sancion", "y_S"), ("penalidad", "y_P"), ("hallazgo", "y_rev")):
        r[f"prov_anio_{n}"] = int(L[L[y]].groupby(["ruc", "anio"]).ngroups)
    return r


def ic_prov_anio_todas(A, B, y="y", n=500, semilla=0):
    """Razón de precisión por proveedor-año sobre todas las entidades-año, con IC en dos vías, como casos.ic_prov_anio pero sin filtrar por entidades-año con algún positivo."""
    e, ne = pd.factorize(pd.concat([A["ruc_entidad"], B["ruc_entidad"]]))
    p, np_ = pd.factorize(pd.concat([A["ruc"], B["ruc"]]))
    py, npy = pd.factorize(pd.concat([A["ruc"] + "|" + A["anio"].astype(str), B["ruc"] + "|" + B["anio"].astype(str)]))
    c = len(A)
    prov_de = np.zeros(len(npy), dtype=int)
    prov_de[py] = p
    hit = np.r_[A[y].to_numpy(), B[y].to_numpy()].astype(bool)
    rng = np.random.default_rng(semilla)

    def prec(lo, hi, we, wp):
        act = we[e[lo:hi]] > 0
        rev, ac = np.zeros(len(npy), bool), np.zeros(len(npy), bool)
        rev[py[lo:hi][act]] = True
        ac[py[lo:hi][act & hit[lo:hi]]] = True
        w = wp[prov_de]
        return (w * ac).sum() / max((w * rev).sum(), 1e-12)

    u_e, u_p = np.ones(len(ne), dtype=int), np.ones(len(np_), dtype=int)
    p0 = prec(c, len(py), u_e, u_p)
    raz = prec(0, c, u_e, u_p) / p0
    bs = []
    for _ in range(n):
        we, wp = rng.poisson(1.0, len(ne)), rng.poisson(1.0, len(np_))
        d0 = prec(c, len(py), we, wp)
        bs.append(prec(0, c, we, wp) / d0 if d0 > 0 else np.nan)
    return {"razon": float(raz), "ic90": [float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))]}


def main():
    d = S.preparar().reset_index(drop=True)
    fit = d[d["anio"].isin(S.AJUSTE)].copy()
    d["x1"], d["x2"], d["x3"] = d["logM"] * d["V"], d["logM"] * d["R"], d["logM"] * d["compra"]
    fit = d[d["anio"].isin(S.AJUSTE)]
    m, w, _ = S.logistica(fit, ["x1", "x2", "x3"], "y")
    d["s_prompt"] = m.decision_function(d[["x1", "x2", "x3"]].to_numpy(dtype=float))
    d["s_alerta_y_R"] = d["logit_R"] + 1e4 * d["V"]
    S.paso(f"pesos del prompt: {w}")
    res = {"pesos_prompt": w, "por_anio": {}, "razones_prov_anio": {}, "razones_caso": {}}
    listas = {"monto": "s_monto", "tamano": "s_tamano_proveedor", "alerta_y_monto": "s_alerta_monto", "modelo_S": "s_S_prov", "modelo_P": "s_P_ampl", "prompt": "s_prompt", "alerta_y_R": "s_alerta_y_R"}
    guardadas = {}
    for a, q in d.groupby("anio"):
        res["por_anio"][int(a)] = {}
        for n, c in listas.items():
            L = top(q, c)
            res["por_anio"][int(a)][n] = resumen(L)
            guardadas[(int(a), n)] = L
        L = cupos(q)
        res["por_anio"][int(a)]["cupos"] = resumen(L)
        guardadas[(int(a), "cupos")] = L
        for n, r in res["por_anio"][int(a)].items():
            S.paso(f"{a} {n}: caso {r['aciertos_amplia']}/{r['mostrados']} = {r['precision_caso']:.4f}; prov-año {r['proveedores_anio_acierto']}/{r['proveedores_anio_mostrados']} = {r['precision_prov_anio']:.4f}; imp {r['aciertos_impedimento']} S {r['aciertos_sancion']} P {r['aciertos_penalidad']} hall {r['aciertos_hallazgo']} del caso {r['aciertos_del_caso']}; ent-año con acierto {r['entidades_anio_con_acierto']}")
    for n in ("prompt", "alerta_y_R", "cupos", "alerta_y_monto", "modelo_P"):
        A = guardadas[(2023, n)]
        res["razones_prov_anio"][n] = {v: ic_prov_anio_todas(A, guardadas[(2023, v)]) for v in ("monto", "tamano")}
        S.paso(f"2023, por proveedor-año, {n}: {res['razones_prov_anio'][n]}")
    # los tres años juntos, solo las reglas sin pesos ajustados
    for n in ("alerta_y_R", "cupos", "alerta_y_monto"):
        A = pd.concat([guardadas[(a, n)] for a in S.PRUEBA])
        res["razones_prov_anio"][f"{n}_tres_anios"] = {v: ic_prov_anio_todas(A, pd.concat([guardadas[(a, v)] for a in S.PRUEBA])) for v in ("monto", "tamano")}
        S.paso(f"tres años, por proveedor-año, {n}: {res['razones_prov_anio'][f'{n}_tres_anios']}")
    # a quién muestran las listas: los proveedores más repetidos
    nom = pd.read_parquet(BASE / "proveedores.parquet", columns=None).head(0).columns.tolist()
    S.paso(f"columnas de proveedores: {nom[:12]}")
    res["repeticion"] = {}
    q23 = d[d["anio"] == 2023]
    tam = q23.groupby("ruc")["c_monto"].agg(["sum", "size"])
    for n in ("tamano", "prompt", "alerta_y_R", "cupos", "monto"):
        L = guardadas[(2023, n)]
        v = L.groupby("ruc").size().sort_values(ascending=False)
        res["repeticion"][n] = {"filas": len(L), "proveedores": len(v), "filas_de_los_20_mas_repetidos": int(v.head(20).sum()), "parte": float(v.head(20).sum() / len(L)), "aciertos_de_los_20": int(L[L["ruc"].isin(v.head(20).index)]["y"].sum()), "aciertos": int(L["y"].sum())}
        S.paso(f"repetición {n}: {res['repeticion'][n]}")
    L = guardadas[(2023, "tamano")]
    v = L.groupby("ruc").agg(filas=("y", "size"), aciertos=("y", "sum"), S=("y_S", "max"), P=("y_P", "max")).sort_values("filas", ascending=False).head(25)
    try:
        pr = pd.read_parquet(BASE / "proveedores.parquet", columns=["ruc", "nombre"]).astype({"ruc": str}).drop_duplicates("ruc").set_index("ruc")["nombre"]
    except Exception as ex:  # noqa: BLE001
        S.paso(f"sin nombres: {ex}")
        pr = pd.Series(dtype=str)
    v["nombre"] = v.index.map(pr)
    S.paso("los más repetidos por tamaño (solo para el registro):\n" + v.to_string())
    (SALIDA / "sonda_prompt_final_2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    S.paso("LISTA")


if __name__ == "__main__":
    main()
