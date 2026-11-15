"""Paso 1 del cierre (HALLAZGOS 157): congelar las variables de D1. Fijado antes de correr.

Se reentrenan D1 binario (etiqueta amplia) y D1 ordinal (Frank y Hall, umbrales ≥1, ≥2 y =3) con dos listas:
- las 116 con que se entrenaron los puntajes guardados de d5 (comprobación: deben reproducirlos);
- las 121 de hoy, que suman t_padre_hijo_prob, t_empresa_pariente_aut_ordenes, t_empresa_pariente_aut_corte_posterior,
  f_empresa_funcionario_ordenes y f_funcionario_propio_ordenes.
Regla: se quedan las 121 solo si no empeoran, con los tres cortes juntos, la vista 1 (D1 binario, precisión en 5 con la
etiqueta amplia) ni la vista 3 (D1 ordinal, precisión en 5 con impedimentos reales). Si alguna empeora, quedan las 116.

    python -I -X utf8 d15_variables.py
Salida: resultados/d15_variables.json y .md, y resultados/puntajes_d15_<año>.parquet
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alerta_impedimentos import impedimento_real  # noqa: E402
from d1_linea_base import CORTES, HIST, d1  # noqa: E402
from etiquetas import Hechos, K  # noqa: E402
from evaluacion import armar_corte, columnas, por_grupo_k  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
NUEVAS = ["t_padre_hijo_prob", "t_empresa_pariente_aut_ordenes", "t_empresa_pariente_aut_corte_posterior",
          "f_empresa_funcionario_ordenes", "f_funcionario_propio_ordenes"]
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def nivel(fuentes):
    f = fuentes.fillna("").astype(str)
    return np.select([f.str.contains("A|B|C"), f.str.contains("S"), f.str.contains("P")], [3, 2, 1], 0)


def lista_121():
    return columnas() + HIST + columnas(clases=("territorial a la fecha",)) + columnas(clases=("familia a la fecha de decisión",))


def entrenar(tr, te, cols, valid):
    _, s_bin, it = d1(tr, te, cols, valid_anio=valid)
    s_ord, its = np.zeros(len(te)), []
    for k in (1, 2, 3):
        _, s_k, it_k = d1(tr.assign(y=(tr["nivel"] >= k).to_numpy()), te, cols, valid_anio=valid)
        s_ord += sig(s_k)
        its.append(it_k)
    return s_bin, s_ord, [it] + its


def p5(te, s, y):
    return por_grupo_k(te[y], s, te["grupo"].to_numpy(), (5,), desempates=10)["P@5"]


def main():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    c121 = lista_121()
    c116 = [c for c in c121 if c not in NUEVAS]
    paso(f"variables: {len(c121)} de hoy y {len(c116)} de los puntajes guardados")
    assert len(c116) == 116 and len(c121) == 121, "las listas no tienen el tamaño esperado"
    res, partes = {"cortes": []}, []
    for tr_a, te_a in CORTES:
        paso(f"corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        tr["nivel"], te["nivel"] = nivel(tr["fuentes"]), nivel(te["fuentes"])
        fu = te["fuentes"].fillna("").astype(str)
        te["y_rev"], te["y_imp"] = fu.str.contains("A|B").to_numpy(), fu.str.contains("C").to_numpy()
        te["y_imp_real"] = te["y_imp"].to_numpy() & impedimento_real(H, te)
        fila = {"prueba": te_a, "arboles": {}}
        for nombre, cols in (("116", c116), ("121", c121)):
            s_bin, s_ord, its = entrenar(tr, te, cols, tr_a[-1])
            te[f"s_D1_binario_{nombre}"], te[f"s_D1_ord_{nombre}"] = s_bin, s_ord
            fila["arboles"][nombre] = its
            paso(f"   {nombre} variables: árboles {its}")
        g = pd.read_parquet(OUT / f"puntajes_d5_{te_a}.parquet", columns=K + ["s_D1_binario", "s_D1_ord"])
        g = g.astype({"ruc_entidad": str, "ruc": str, "anio": int})
        te = te.astype({"ruc_entidad": str, "ruc": str, "anio": int}).merge(g, on=K, how="left")
        fila["reproduce_guardado"] = {
            "binario_corr": float(np.corrcoef(te["s_D1_binario"], te["s_D1_binario_116"])[0, 1]),
            "binario_max_dif": float((te["s_D1_binario"] - te["s_D1_binario_116"]).abs().max()),
            "ordinal_corr": float(np.corrcoef(te["s_D1_ord"], te["s_D1_ord_116"])[0, 1]),
            "ordinal_max_dif": float((te["s_D1_ord"] - te["s_D1_ord_116"]).abs().max())}
        paso(f"   reproduce los puntajes guardados con 116: {fila['reproduce_guardado']}")
        res["cortes"].append(fila)
        cols_guardar = K + ["grupo", "y", "y_estricta", "y_rev", "y_imp", "y_imp_real", "nivel", "c_monto"] + \
            [f"s_D1_{t}_{n}" for t in ("binario", "ord") for n in ("116", "121")]
        te[cols_guardar].to_parquet(OUT / f"puntajes_d15_{te_a}.parquet", index=False)
        partes.append(te[cols_guardar])
    todo = pd.concat(partes, ignore_index=True)
    vistas = {"vista_1_D1_binario_amplia": ("s_D1_binario", "y"), "vista_3_D1_ordinal_impedimentos_reales": ("s_D1_ord", "y_imp_real"),
              "grave_D1_ordinal_estricta": ("s_D1_ord", "y_estricta"), "hallazgos_D1_ordinal": ("s_D1_ord", "y_rev")}
    res["vistas"] = {}
    for v, (s, y) in vistas.items():
        a116 = float(p5(todo, todo[f"{s}_116"].to_numpy(), y).mean())
        a121 = float(p5(todo, todo[f"{s}_121"].to_numpy(), y).mean())
        res["vistas"][v] = {"116": a116, "121": a121}
        paso(f"   {v}: 116 → {a116:.4f}; 121 → {a121:.4f}")
    principales = ("vista_1_D1_binario_amplia", "vista_3_D1_ordinal_impedimentos_reales")
    res["queda"] = "121" if all(res["vistas"][v]["121"] >= res["vistas"][v]["116"] for v in principales) else "116"
    paso(f"quedan las {res['queda']} variables")
    (OUT / "d15_variables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Congelar las variables de D1 (HALLAZGOS 157, paso 1)", "", "| Vista | 116 variables | 121 variables |", "|---|---|---|"]
    L += [f"| {v} | {d['116']:.4f} | {d['121']:.4f} |" for v, d in res["vistas"].items()]
    L += ["", f"**Quedan las {res['queda']} variables** (regla: las 121 solo si no empeoran la vista 1 ni la vista 3).", "",
          "Reproducción de los puntajes guardados con 116 variables, por corte:", ""]
    L += [f"- {c['prueba']}: {c['reproduce_guardado']}; árboles {c['arboles']}" for c in res["cortes"]]
    (OUT / "d15_variables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
