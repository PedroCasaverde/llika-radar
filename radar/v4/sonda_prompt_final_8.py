"""Cierre de la sonda de la sección 71: la fórmula del prompt con C_local, sin pesos y con la alerta primero; con un puesto para la compra más grande; y sin el modelo, para ver qué aporta. Solo lectura, sin API y solo años de exploración.

    python -I -X utf8 sonda_prompt_final_8.py
Salida: resultados/por_caso/sonda_prompt_final_8.json. Solo agregados.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sonda_prompt_final as S  # noqa: E402
import sonda_prompt_final_2 as S2  # noqa: E402
import sonda_prompt_final_5 as S5  # noqa: E402
from casos import SALIDA, top  # noqa: E402


def con_cupo(q, col):
    """Los 4 primeros por la columna y el quinto, la compra más grande que no entró."""
    filas = []
    for _, g in q.groupby("grupo", sort=False):
        if len(g) <= 5:
            filas.extend(g.index.tolist())
            continue
        elegidos = g.nlargest(4, col).index.tolist()
        filas.extend(elegidos + [g.drop(index=elegidos)["c_monto"].idxmax()])
    return q.loc[filas]


def ic_proveedores(A, B, y="y", n=2000, semilla=0):
    """Las entidades-año son un censo: lo incierto es qué proveedores tienen señal. Pesos de Poisson solo por proveedor, con las filas fijas."""
    def por_prov(L):
        return L[L[y]].groupby(["ruc", "anio"]).size().reset_index()["ruc"]
    a, b = por_prov(A), por_prov(B)
    p, u = pd.factorize(pd.concat([a, b]))
    ca = len(a)
    rng = np.random.default_rng(semilla)
    bs = []
    for _ in range(n):
        w = rng.poisson(1.0, len(u))
        d0 = w[p[ca:]].sum() / len(B)
        bs.append((w[p[:ca]].sum() / len(A)) / d0 if d0 > 0 else np.nan)
    return {"razon": (len(a) / len(A)) / (len(b) / len(B)), "ic90": [float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))]}


def main():
    d = S.preparar().reset_index(drop=True)
    p = d.drop_duplicates(["ruc", "anio"])[["ruc", "anio", "R"]].copy()
    p["pct"] = p.groupby("anio")["R"].rank(pct=True)
    d = d.merge(p[["ruc", "anio", "pct"]], on=["ruc", "anio"], how="left")
    d["mochila"] = d["pct"] >= 0.98
    d["s_final"] = 1e4 * d["V"] + d["logM"] * d["R"] * d["c_local"]
    d["s_sin_modelo"] = 1e4 * d["V"] + d["logM"] * d["c_local"]
    d["s_sin_alerta"] = d["logM"] * d["R"] * d["c_local"]
    listas = {"monto": "s_monto", "tamano": "s_tamano_proveedor", "alerta_y_monto": "s_alerta_monto", "final": "s_final", "sin_modelo": "s_sin_modelo", "sin_alerta": "s_sin_alerta"}
    G = {}
    res = {"por_anio": {}, "tres": {}, "razones": {}, "motivos": {}, "condicion": {}}
    for a, q in d.groupby("anio"):
        a = int(a)
        res["por_anio"][a] = {}
        for n, c in listas.items():
            G[(a, n)] = top(q, c)
        G[(a, "final_con_cupo")] = con_cupo(q, "s_final")
        for n in [*listas, "final_con_cupo"]:
            res["por_anio"][a][n] = S2.resumen(G[(a, n)])
    nombres = [*listas, "final_con_cupo"]
    for n in nombres:
        T = pd.concat([G[(a, n)] for a in S.PRUEBA])
        r = S2.resumen(T)
        res["tres"][n] = r
        hv = T[T["y"]].groupby("ruc").size().sort_values(ascending=False)
        v = T.groupby("ruc").size().sort_values(ascending=False)
        res["tres"][n].update({"persona_natural": int(T["ruc"].str.startswith("10").sum()), "aciertos_fila_5_que_mas_aportan": int(hv.head(5).sum()), "filas_20_mas_repetidos": int(v.head(20).sum())})
        S.paso(f"{n}: {r['proveedores_anio_acierto']} de {r['mostrados']} = {r['proveedores_anio_acierto'] / r['mostrados']:.3%}; por año {[res['por_anio'][a][n]['proveedores_anio_acierto'] for a in S.PRUEBA]}; por fila {r['aciertos_amplia']} ({r['precision_caso']:.3%}); proveedores mostrados {r['proveedores_anio_mostrados']} (precisión {r['precision_prov_anio']:.3%}); por verdad imp {r['prov_anio_impedimento']} S {r['prov_anio_sancion']} P {r['prov_anio_penalidad']} hall {r['prov_anio_hallazgo']}; del caso {r['aciertos_del_caso']}; ent-año con señal {r['entidades_anio_con_acierto']}; persona natural {res['tres'][n]['persona_natural']}; 20 más repetidos {res['tres'][n]['filas_20_mas_repetidos']}")
    for n, v in (("final", "monto"), ("final", "tamano"), ("final", "alerta_y_monto"), ("final", "sin_modelo"), ("final_con_cupo", "monto"), ("final_con_cupo", "tamano"), ("final_con_cupo", "alerta_y_monto"), ("final_con_cupo", "final"), ("sin_modelo", "monto")):
        A, B = pd.concat([G[(a, n)] for a in S.PRUEBA]), pd.concat([G[(a, v)] for a in S.PRUEBA])
        res["razones"][f"{n}_frente_a_{v}"] = {"tres_anios_proveedores": ic_proveedores(A, B), "tres_anios_dos_vias": S5.ic(A, B), "2023_proveedores": ic_proveedores(G[(2023, n)], G[(2023, v)])}
        S.paso(f"{n} frente a {v}: {res['razones'][f'{n}_frente_a_{v}']}")
    for n in ("final", "final_con_cupo"):
        res["motivos"][n] = {}
        T = pd.concat([G[(a, n)] for a in S.PRUEBA])
        if n == "final_con_cupo":
            base = pd.concat([top(q, "s_final", k=4) for _, q in d.groupby("anio")])
            en_cuatro = T.index.isin(base.index)
        else:
            en_cuatro = np.ones(len(T), bool)
        m = np.select([T["V"].to_numpy() > 0, T["mochila"].to_numpy() & en_cuatro], ["vinculo", "mochila"], "compra")
        q = T.assign(motivo=m)
        for k, g in q.groupby("motivo"):
            res["motivos"][n][k] = {"filas": len(g), "proveedores_con_senal": int(g[g["y"]].groupby(["ruc", "anio"]).ngroups), "proveedores": int(g.groupby(["ruc", "anio"]).ngroups), "aciertos_por_fila": int(g["y"].sum()), "persona_natural": int(g["ruc"].str.startswith("10").sum())}
        S.paso(f"motivos, {n}: {res['motivos'][n]}")
    (SALIDA / "sonda_prompt_final_8.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    S.paso("LISTA")


if __name__ == "__main__":
    main()
