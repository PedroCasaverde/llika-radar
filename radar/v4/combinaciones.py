"""Combinaciones del monto y de D1 (HALLAZGOS 140; Pedro: «¿no se puede combinar?»). Fijadas antes de calcular.

Fusión recíproca de rangos (Cormack, Clarke y Büttcher, 2009), dentro de cada entidad-año: puntaje = Σ 1/(60 + rango),
con k = 60 y pesos iguales:
- R1: D1 binario y D1 ordinal;
- R2: monto y D1 ordinal;
- R3: monto, D1 binario y D1 ordinal.
Usa los puntajes guardados de la ronda ordinal (resultados/puntajes_d5_<año>.parquet); no reentrena. La lista nacional
(precisión en 1 000) usa los rangos nacionales con la misma fórmula.

    python -I -X utf8 combinaciones.py
Salida: resultados/combinaciones.json y .md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluacion import diferencia_conjunta, metricas, metricas_k, razon_conjunta  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
K_RRF = 60
COMBOS = {"R1_bin_ord": ["D1_binario", "D1_ord"], "R2_monto_ord": ["monto", "D1_ord"], "R3_monto_bin_ord": ["monto", "D1_binario", "D1_ord"]}
TAREAS = {"amplia": "y", "nivel_3": "y_estricta", "revision_AB": "y_rev", "impedimentos_C": "y_imp"}


def rrf(te, nombres, por_grupo=True):
    s = np.zeros(len(te))
    for n in nombres:
        r = te.groupby("grupo")[f"s_{n}"].rank(ascending=False, method="average") if por_grupo else te[f"s_{n}"].rank(ascending=False, method="average")
        s += 1.0 / (K_RRF + r.to_numpy())
    return s


def main():
    puntajes = {}
    for a in (2020, 2021, 2023):
        te = pd.read_parquet(OUT / f"puntajes_d5_{a}.parquet")
        s = {n: te[f"s_{n}"].to_numpy() for n in ("monto", "tamano_proveedor", "D1_binario", "D1_ord")}
        for c, ns in COMBOS.items():
            s[c] = rrf(te, ns)
        puntajes[a] = (te, s)
    res = {"cortes": {}, "comparaciones": {}, "razones": {}}
    L = ["# Combinaciones por fusión recíproca de rangos (140): precisión en 5 por tarea", "",
         "| Corte | Puntaje | Amplia | Nivel 3 | Revisión (A o B) | Impedimentos (C) | MAP amplia |", "|---|---|---|---|---|---|---|"]
    for a, (te, s) in puntajes.items():
        res["cortes"][a] = {}
        for n, v in s.items():
            m = metricas(te, v)
            pk = {t: metricas_k(te, v, y=c)["P@5_ent"] for t, c in TAREAS.items()}
            res["cortes"][a][n] = {"MAP_ent": round(m["MAP_ent"], 4), **{t: round(x, 4) for t, x in pk.items()}}
            L.append(f"| {a} | {n} | " + " | ".join(f"{pk[t]:.4f}" for t in TAREAS) + f" | {m['MAP_ent']:.4f} |")
            print(L[-1], flush=True)
    L += ["", "| Comparación | Diferencia de MAP (amplia) | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for c in list(COMBOS) + ["D1_binario", "D1_ord"]:
        for ref in ("tamano_proveedor", "monto"):
            d = diferencia_conjunta([(te, s[c], s[ref]) for te, s in puntajes.values()])
            res["comparaciones"][f"{c} − {ref}"] = d
            L.append(f"| {c} − {ref} | {d['dif_MAP']:+.4f} | [{d['ic90'][0]:+.4f}, {d['ic90'][1]:+.4f}] | {d['por_corte']} |")
            print(L[-1], flush=True)
    L += ["", "| Razón (vara: frente al tamaño; también frente al monto) | Candidata | Referencia | Razón | IC al 90 % |", "|---|---|---|---|---|"]
    for c in list(COMBOS) + ["D1_binario", "D1_ord"]:
        for ref in ("tamano_proveedor", "monto"):
            for t in ("amplia", "nivel_3", "revision_AB"):
                for medida, q in (("P@5", False), ("AP", True)):
                    r = razon_conjunta([(te, s[c], s[ref]) for te, s in puntajes.values()], medida=medida, quintil=q, y=TAREAS[t])
                    k = f"{c} / {ref} | {t} | {medida}{' en rangos de monto' if q else ''}"
                    res["razones"][k] = r
                    L.append(f"| {k} | {r['modelo']:.4f} | {r['monto']:.4f} | {r['razon']:.2f} | [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}] |")
    (OUT / "combinaciones.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (OUT / "combinaciones.md").write_text("\n".join(L), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
