"""Estadísticos simples para los hallazgos del control (HALLAZGOS 149). Fijada antes de correr.

1. AUC dentro de la entidad-año de cada estadístico de 149, solo en los años de entrenamiento de cada corte, con el
   logaritmo del monto y la regla rústica como referencia. El vacío va con lo neutro: la mediana del entrenamiento en
   Benford, cero en lo demás.
2. La regresión logística condicional de 147 (mismas variables, conjuntos 1:20 dentro de la entidad-año, semilla 0) y la
   misma con los estadísticos principales de 149 como variables (Benford con su indicador de «sin dato»).
3. Candidatas frente al monto, con la vara de 141, en los tres cortes:
     regresion_control   la de 147 otra vez; debe repetir sus cifras (confirma de paso que la reconstrucción de la
                         data no tocó 2018 a 2023);
     regresion_simples   147 más los estadísticos;
     R2_simples          fusión RRF de la anterior con el monto;
     monto_y_mejor       fusión RRF del monto con el mejor estadístico solo, elegido por su AUC de entrenamiento en
                         cada corte (la opción más simple).

    python -I -X utf8 d11_simples.py
Salida: resultados/d11_simples.json y .md
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combinaciones import rrf  # noqa: E402
from comparacion_final import PESOS_AUDITOR, riesgo  # noqa: E402
from d1_linea_base import CORTES  # noqa: E402
from d3_capas import auc_dentro  # noqa: E402
from d10_regresion import TAREAS, ajustar, variables  # noqa: E402
from etiquetas import Hechos, K  # noqa: E402
from evaluacion import armar_corte, diferencia_conjunta, metricas, metricas_k, razon_conjunta  # noqa: E402

warnings.filterwarnings("ignore")
V4 = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4"
OUT = V4 / "resultados"
PRINCIPALES = ["s_obj_frac_parte", "s_obj_frac_veces", "s_benford_1d", "s_benford_2d", "s_redondo_100", "s_redondo_1000",
               "s_monto_repetido", "s_precio_z", "s_precio_atipico", "s_fecha_inusual"]
SENSIBILIDAD = ["s_obj3_frac_parte", "s_obj3_frac_veces", "s_precio3_z", "s_precio3_atipico", "s_benford_1d_nac", "s_benford_2d_nac"]
BENFORD = ["s_benford_1d", "s_benford_2d", "s_benford_1d_nac", "s_benford_2d_nac"]
D10 = {2020: 0.0654, 2021: 0.0548, 2023: 0.0596}       # P@5 de hallazgos de regresion_control en 147 (d10_regresion.md)
CANDIDATAS = ("regresion_simples", "R2_simples", "monto_y_mejor", "R2_regresion")
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def lleno(d, c, ref):
    return d[c].astype(float).fillna(ref.get(c, 0.0))


def simples(d, ref):
    X = pd.DataFrame(index=d.index)
    X["frac_objeto_parte"] = lleno(d, "s_obj_frac_parte", ref)
    X["frac_objeto_veces"] = np.log1p(lleno(d, "s_obj_frac_veces", ref))
    X["benford_sin_dato"] = d["s_benford_1d"].isna().astype(float)
    for c in ("s_benford_1d", "s_benford_2d", "s_redondo_100", "s_redondo_1000", "s_monto_repetido", "s_precio_z",
              "s_precio_atipico", "s_fecha_inusual"):
        X[c[2:]] = lleno(d, c, ref)
    return X


def main():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    xs = pd.read_parquet(V4 / "x_simples.parquet")
    res, puntajes, aucs = {"cortes": []}, {}, []
    for tr_a, te_a in CORTES:
        paso(f"armando el corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        tr, te = (d.reset_index(drop=True).merge(xs.astype({"anio": d["anio"].dtype}), on=K, how="left") for d in (tr, te))
        for d in (tr, te):
            fu = d["fuentes"].fillna("").astype(str)
            d["y_rev"], d["y_imp"] = fu.str.contains("A|B").to_numpy(), fu.str.contains("C").to_numpy()
        paso(f"   casos con estadísticos: entrenamiento {tr['s_redondo_100'].notna().mean():.2%}, prueba {te['s_redondo_100'].notna().mean():.2%}")
        # 1) AUC dentro de la entidad-año, solo en los años de entrenamiento
        ref = {c: float(tr[c].median()) for c in BENFORD}
        ta = tr[tr.groupby("grupo")["y_rev"].transform("any")]
        filas = {c: lleno(ta, c, ref).to_numpy() for c in PRINCIPALES + SENSIBILIDAD}
        filas["log monto (referencia)"] = np.log1p(ta["c_monto"].astype(float).to_numpy())
        filas["regla rústica (referencia)"] = riesgo(ta)
        for c, v in filas.items():
            a, ic = auc_dentro(ta["y_rev"].to_numpy(), v, ta["grupo"].to_numpy(), ta["ruc_entidad"].to_numpy())
            tipo = "principal" if c in PRINCIPALES else "sensibilidad" if c in SENSIBILIDAD else "referencia"
            aucs.append({"corte": te_a, "estadistico": c, "tipo": tipo, "auc": round(a, 4), "ic90": [round(ic[0], 4), round(ic[1], 4)]})
        mejor = max((r for r in aucs if r["corte"] == te_a and r["tipo"] == "principal"), key=lambda r: r["auc"])["estadistico"]
        paso("   AUC de entrenamiento: " + ", ".join(f"{r['estadistico']} {r['auc']:.3f}" for r in aucs if r["corte"] == te_a)
             + f" | mejor: {mejor}")
        # 2) regresión condicional de 147, sola y con los estadísticos
        fx = lambda d: pd.concat([variables(d), simples(d, ref)], axis=1)  # noqa: E731
        b0, mu0, sd0, _, ok0 = ajustar(tr)
        b1, mu1, sd1, conjuntos, ok1 = ajustar(tr, fx)
        te = te.astype({"ruc_entidad": str, "ruc": str, "anio": int})
        s = {"monto": te["c_monto"].to_numpy(dtype=float), "riesgo_rustico": riesgo(te)}
        s["regresion_control"] = (((variables(te) - mu0) / sd0) @ b0).to_numpy()
        s["regresion_simples"] = (((fx(te) - mu1) / sd1) @ b1).to_numpy()
        te["s_monto"], te["s_regresion_control"], te["s_regresion_simples"] = s["monto"], s["regresion_control"], s["regresion_simples"]
        te["s_mejor"] = lleno(te, mejor, ref).to_numpy()
        s["R2_regresion"] = rrf(te, ["monto", "regresion_control"])
        s["R2_simples"] = rrf(te, ["monto", "regresion_simples"])
        s["monto_y_mejor"] = rrf(te, ["monto", "mejor"])
        prev = pd.read_parquet(OUT / f"puntajes_d5_{te_a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        te = te.merge(prev[K + ["s_D1_ord"]], on=K, how="left")
        s["R2"] = rrf(te, ["monto", "D1_ord"])
        odds = np.exp(b1).round(3).to_dict()
        paso(f"   {conjuntos:,} conjuntos; convergió: {ok0} y {ok1}; razón de odds por desviación estándar de los estadísticos: "
             + ", ".join(f"{k} {odds[k]}" for k in simples(te.head(1), ref).columns))
        fila = {"prueba": te_a, "mejor_estadistico": mejor, "conjuntos": conjuntos, "convergio": [ok0, ok1], "razon_odds_simples": odds}
        for n, v in s.items():
            fila[n] = {k_: round(x_, 4) if isinstance(x_, float) else x_ for k_, x_ in metricas(te, v, y="y_rev").items()}
            fila[n]["precision_k"] = {t: {k_: round(x_, 4) for k_, x_ in metricas_k(te, v, y=c).items()} for t, c in TAREAS.items()}
            pk = fila[n]["precision_k"]
            paso(f"   {n:<18} hallazgos: P@5 {pk['revision_AB']['P@5_ent']:.4f} MAP {fila[n]['MAP_ent']:.4f} | amplia {pk['amplia']['P@5_ent']:.4f} "
                 f"| grave {pk['estricta_nivel_3']['P@5_ent']:.4f} | impedimento {pk['impedimentos_C']['P@5_ent']:.4f}")
        fila["repite_147"] = abs(fila["regresion_control"]["precision_k"]["revision_AB"]["P@5_ent"] - D10[te_a]) < 5e-4
        paso(f"   la regresión de 147 repite sus cifras: {fila['repite_147']}")
        res["cortes"].append(fila)
        puntajes[te_a] = (te[["grupo", "y", "y_estricta", "y_rev", "y_imp", "c_monto"]].copy(), s)
    res["auc_entrenamiento"] = aucs
    paso("vara frente al monto en hallazgos (IC al 90 %)")
    res["comparaciones"], res["vara"] = {}, {}
    for a in CANDIDATAS:
        d_ = diferencia_conjunta([(t.assign(y=t["y_rev"]), s_[a], s_["monto"]) for t, s_ in puntajes.values()])
        res["comparaciones"][f"{a} − monto"] = d_
        print(f"   {a} − monto (MAP hallazgos): {d_}", flush=True)
        for medida, q in (("P@5", False), ("AP", True)):
            r = razon_conjunta([(t, s_[a], s_["monto"]) for t, s_ in puntajes.values()], medida=medida, quintil=q, y="y_rev")
            k_ = f"{a} / monto | hallazgos | {medida}{' en rangos de monto' if q else ''}"
            res["vara"][k_] = r
            print(f"   {k_:<62} {r['modelo']:.4f} / {r['monto']:.4f} = {r['razon']:.2f} [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}]", flush=True)
    te_all = pd.concat([t.assign(**{f"s_{n}": v for n, v in s_.items()}) for t, s_ in puntajes.values()], ignore_index=True)
    nombres = ("monto", "riesgo_rustico", "regresion_control", "R2_regresion", "R2", "regresion_simples", "R2_simples", "monto_y_mejor")
    p5 = {n: float(metricas_k(te_all, te_all[f"s_{n}"].to_numpy(), y="y_rev")["P@5_ent"]) for n in nombres}
    base = te_all.groupby("grupo")["y_rev"].agg(["sum", "size"])
    base = base[base["sum"] > 0]
    p5["azar"] = float((base["sum"] / base["size"]).mean())
    p5["auditor promedio"] = sum(w * p5[m] for m, w in PESOS_AUDITOR.items())
    res["p5_hallazgos_conjunto"] = p5
    print("   P@5 hallazgos, tres cortes juntos:", {k: round(v, 4) for k, v in p5.items()}, flush=True)
    (OUT / "d11_simples.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# Estadísticos simples para los hallazgos del control (149)", "",
         "## 1. AUC dentro de la entidad-año, solo en los años de entrenamiento de cada corte", "",
         "| Estadístico | Tipo | " + " | ".join(f"Entrenamiento del corte {c[1]}" for c in CORTES) + " |", "|---|---|" + "---|" * len(CORTES)]
    for est in dict.fromkeys(r["estadistico"] for r in aucs):
        rr = {r["corte"]: r for r in aucs if r["estadistico"] == est}
        L.append(f"| {est} | {rr[CORTES[0][1]]['tipo']} | " + " | ".join(
            f"{rr[c[1]]['auc']:.3f} [{rr[c[1]]['ic90'][0]:.3f}, {rr[c[1]]['ic90'][1]:.3f}]" for c in CORTES) + " |")
    L += ["", "Mejor estadístico por corte (elegido en entrenamiento): " + ", ".join(f"{f['prueba']}: {f['mejor_estadistico']}" for f in res["cortes"]),
          "", "## 2. Prueba", "", "| Corte | Puntaje | Hallazgos P@5 | MAP hallazgos | Amplia | Grave | Impedimento |", "|---|---|---|---|---|---|---|"]
    for f in res["cortes"]:
        for n in nombres:
            pk = f[n]["precision_k"]
            L.append(f"| {f['prueba']} | {n} | {pk['revision_AB']['P@5_ent']:.4f} | {f[n]['MAP_ent']:.4f} | {pk['amplia']['P@5_ent']:.4f} | "
                     f"{pk['estricta_nivel_3']['P@5_ent']:.4f} | {pk['impedimentos_C']['P@5_ent']:.4f} |")
    L += ["", "La regresión de 147 repite sus cifras: " + ", ".join(f"{f['prueba']} {f['repite_147']}" for f in res["cortes"]),
          "", "| Comparación (MAP hallazgos) | Diferencia | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for k_, d_ in res["comparaciones"].items():
        L.append(f"| {k_} | {d_['dif_MAP']:+.4f} | [{d_['ic90'][0]:+.4f}, {d_['ic90'][1]:+.4f}] | {d_['por_corte']} |")
    L += ["", "| Vara (frente al monto) | Candidata | Monto | Razón | IC al 90 % |", "|---|---|---|---|---|"]
    for k_, r in res["vara"].items():
        L.append(f"| {k_} | {r['modelo']:.4f} | {r['monto']:.4f} | {r['razon']:.2f} | [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}] |")
    L += ["", "| P@5 hallazgos, tres cortes juntos | Valor |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in sorted(p5.items(), key=lambda t: -t[1])]
    L += ["", "| Razón de odds por desviación estándar (regresion_simples) | " + " | ".join(str(f["prueba"]) for f in res["cortes"]) + " |",
          "|---|" + "---|" * len(res["cortes"])]
    for k in res["cortes"][-1]["razon_odds_simples"]:
        L.append(f"| {k} | " + " | ".join(str(f["razon_odds_simples"][k]) for f in res["cortes"]) + " |")
    (OUT / "d11_simples.md").write_text("\n".join(L), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
