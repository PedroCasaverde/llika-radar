"""Comparación final de las finalistas frente al método rústico del auditor (HALLAZGOS 145). Sin reentrenar.

Finalistas, con los puntajes guardados de la ronda ordinal (`resultados/puntajes_d5_<año>.parquet`): D1 binario,
D1 ordinal, R2 (monto + ordinal) y R3 (monto + binario + ordinal, fusión de rangos de 140). Métodos rústicos:
- monto: lo más caro primero;
- riesgo: 3 × [pasó el tope en 30 días] + 2 × [algún mes superó el tope con varias órdenes] + 1 × [órdenes cerca del
  tope], desempata el monto (la regla fijada en 145 con las citas de los informes);
- azar: acierto esperado = tasa base de cada entidad-año;
- auditor promedio: 0,44 riesgo + 0,375 monto + 0,19 azar (proporciones de la muestra de 150 informes).
Medidas: precisión en 5 por entidad (en las entidades-año con algún positivo), revisiones por acierto y soles por acierto
(1 hora por revisión a S/ 65), por tarea. Para los impedimentos, además, la alerta frente a la misma cantidad de casos
tomados por cada método en todo el país.

    python -I -X utf8 comparacion_final.py
Salida: resultados/comparacion_final.md y .json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combinaciones import rrf  # noqa: E402
from etiquetas import Hechos, K  # noqa: E402
from evaluacion import cargar_x, familia_a_la_decision, por_grupo_k, universo  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
SOLES_HORA = 65
PESOS_AUDITOR = {"riesgo_rustico": 0.44, "monto": 0.375, "azar": 0.19}
TAREAS = {"señal adversa (amplia)": "y", "grave (hallazgo o impedimento)": "y_estricta", "hallazgo del control (A o B)": "y_rev",
          "impedimento (C)": "y_imp"}
RIESGO = ["c_pasa_tope_30d", "c_meses_superan_tope", "c_cerca_tope"]


def riesgo(d):
    # el monto solo desempata dentro de cada nivel; con 1e-9 lo tapaba el ruido de desempate de la evaluación (160, M8)
    return 100 * (3 * (d["c_pasa_tope_30d"] > 0) + 2 * (d["c_meses_superan_tope"] > 0) + 1 * (d["c_cerca_tope"] > 0)).to_numpy(float) \
        + np.log1p(d["c_monto"].astype(float).to_numpy())


def main():
    partes = []
    for a in (2020, 2021, 2023):
        te = pd.read_parquet(OUT / f"puntajes_d5_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        x = cargar_x([a])[K + RIESGO].astype({"ruc_entidad": str, "ruc": str, "anio": int})
        te = te.merge(x, on=K, how="left")
        te[RIESGO] = te[RIESGO].fillna(0)
        te["s_R2"] = rrf(te, ["monto", "D1_ord"])
        te["s_R3"] = rrf(te, ["monto", "D1_binario", "D1_ord"])
        te["s_riesgo_rustico"] = riesgo(te)
        partes.append(te)
    te = pd.concat(partes, ignore_index=True)
    metodos = {"monto": "s_monto", "riesgo_rustico": "s_riesgo_rustico", "D1_binario": "s_D1_binario", "D1_ordinal": "s_D1_ord",
               "R2 (monto + ordinal)": "s_R2", "R3 (monto + binario + ordinal)": "s_R3"}
    res, L = {}, ["# Comparación final frente al método rústico del auditor (145)", "",
                  "Precisión en 5 por entidad-año con algún positivo (2020, 2021 y 2023 juntos); entre paréntesis, revisiones por "
                  "acierto y soles por acierto (1 hora por revisión, S/ 65).", ""]
    for tarea, y in TAREAS.items():
        g = te["grupo"].to_numpy()
        p = {n: float(por_grupo_k(te[y], te[c], g, (5,))["P@5"].mean()) for n, c in metodos.items()}
        base = te.groupby("grupo")[y].agg(["sum", "size"])
        base = base[base["sum"] > 0]
        p["azar"] = float((base["sum"] / base["size"]).mean())
        p["auditor promedio"] = sum(w * p[m] for m, w in PESOS_AUDITOR.items())
        res[tarea] = p
        L += [f"## {tarea}", "", "| Método | Precisión en 5 | Revisiones por acierto | Soles por acierto | Frente al auditor promedio |",
              "|---|---|---|---|---|"]
        for n, v in sorted(p.items(), key=lambda t: -t[1]):
            L.append(f"| {n} | {v:.4f} | {1 / v:,.1f} | S/ {SOLES_HORA / v:,.0f} | {v / p['auditor promedio']:.2f} veces |" if v > 0
                     else f"| {n} | 0 | — | — | — |")
        L.append("")
        print(tarea, {k: round(v, 4) for k, v in p.items()}, flush=True)
    # alerta de impedimentos frente a la misma cantidad de casos por cada método, en todo el país
    H = Hechos("publicacion")
    filas = []
    for a in (2020, 2021, 2023):
        x = familia_a_la_decision(cargar_x([a]), a)
        hist = H.historia(x)
        x = pd.concat([x, hist], axis=1)
        x = x[universo(x, hist)].copy()
        x = x.join(H.etiqueta_amplia(x, "2025-12-31"))
        x = x[~x["conocido_al_cierre"]]
        fu = x["fuentes"].fillna("").astype(str)
        imp, est = fu.str.contains("C").to_numpy(), fu.str.contains("A|B|C").to_numpy()
        alerta = ((x["t_autoridad_ordenes"] > 0) | (x["t_empresa_autoridad_ordenes"] > 0) | (x["t_pariente_raro_ordenes"] > 0)
                  | (x["f_familiar_prob"] >= 0.5)).to_numpy()
        n = int(alerta.sum())
        for nombre, s in (("monto", x["c_monto"].astype(float).to_numpy()), ("riesgo_rustico", riesgo(x))):
            top = np.zeros(len(x), dtype=bool)
            top[np.argsort(-s)[:n]] = True
            filas.append({"año": a, "método": nombre, "casos": n, "impedimentos": int(imp[top].sum()), "irregularidades": int(est[top].sum())})
        filas.append({"año": a, "método": "alerta de impedimentos", "casos": n, "impedimentos": int(imp[alerta].sum()),
                      "irregularidades": int(est[alerta].sum())})
        filas.append({"año": a, "método": "azar (esperado)", "casos": n, "impedimentos": round(n * imp.mean(), 1),
                      "irregularidades": round(n * est.mean(), 1)})
    t = pd.DataFrame(filas)
    tot = t.groupby("método")[["casos", "impedimentos", "irregularidades"]].sum()
    tot["revisiones por impedimento"] = (tot["casos"] / tot["impedimentos"].replace(0, np.nan)).round(1)
    tot["soles por impedimento"] = (SOLES_HORA * tot["revisiones por impedimento"]).round(0)
    tot["revisiones por irregularidad"] = (tot["casos"] / tot["irregularidades"].replace(0, np.nan)).round(1)
    res["alerta_impedimentos"] = tot.reset_index().to_dict("records")
    L += ["## Alerta de impedimentos frente a la misma cantidad de casos en todo el país (2020, 2021 y 2023 juntos)", "",
          tot.reset_index().to_markdown(index=False)]
    print(tot.to_string(), flush=True)
    (OUT / "comparacion_final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (OUT / "comparacion_final.md").write_text("\n".join(L), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
