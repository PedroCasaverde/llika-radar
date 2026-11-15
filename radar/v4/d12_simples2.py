"""Lo que el plan de octubre y la bibliografía tenían y la v4 no probó (HALLAZGOS 150; Pedro: «sí, apruebo 150», 2/11).
Fijada antes de correr.

1. Isolation Forest (Liu, Ting y Zhou, 2008), en cada año de prueba y sin etiquetas: 200 árboles, 256 casos por árbol,
   semilla 0, con variables del caso comparables dentro de la entidad (percentiles de monto y órdenes en la entidad,
   banderas del tope, servicios, diciembre, descripción y estadísticos de 149).
2. Índice simple de riesgo (CRI; Fazekas y Kocsis, 2020): promedio con pesos iguales de 14 banderas binarias con umbral
   fijo; entran las que, en los años de entrenamiento, tienen un AUC dentro de la entidad con el límite inferior de su
   IC al 90 % sobre 0,5.
3. Bayes ingenuo con tramos: cada variable de 147 y 149 en cinco tramos por quintiles del entrenamiento; el puntaje es
   la suma del logaritmo de la razón de verosimilitud de cada tramo (hallazgo frente a sin hallazgo, suavizado de 1),
   estimado en las entidades-año con algún hallazgo, como la regresión.
Cada uno solo y fusionado con el monto (RRF), con la vara de 141 frente al monto en los tres cortes.

    python -I -X utf8 d12_simples2.py
Salida: resultados/d12_simples2.json y .md
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combinaciones import rrf  # noqa: E402
from comparacion_final import PESOS_AUDITOR, riesgo  # noqa: E402
from d1_linea_base import CORTES  # noqa: E402
from d3_capas import auc_dentro  # noqa: E402
from d10_regresion import TAREAS, variables  # noqa: E402
from d11_simples import BENFORD, simples  # noqa: E402
from etiquetas import Hechos, K  # noqa: E402
from evaluacion import armar_corte, diferencia_conjunta, metricas, metricas_k, razon_conjunta  # noqa: E402

warnings.filterwarnings("ignore")
V4 = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4"
OUT = V4 / "resultados"
CANDIDATAS = ("isolation_forest", "monto_y_if", "cri", "monto_y_cri", "bayes_ingenuo", "monto_y_bayes")
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def anomalia(d):
    """Variables para Isolation Forest: comparables dentro de la entidad, para que «grande» no sea «raro»."""
    f = lambda c: d[c].astype(float).fillna(0)  # noqa: E731
    return pd.DataFrame({
        "monto_pct": f("c_monto_pct_entidad"), "ordenes_pct": f("c_ordenes_pct_entidad"), "monto_medio_pct": f("c_monto_medio_pct_entidad"),
        "cerca_tope": (f("c_cerca_tope") > 0).astype(float), "pasa_tope_30d": (f("c_pasa_tope_30d") > 0).astype(float),
        "mes_supera_tope": (f("c_meses_superan_tope") > 0).astype(float), "varias_mismo_dia": (f("c_max_mismo_dia") > 1).astype(float),
        "parte_servicios": f("c_parte_servicios"), "parecido_trabajador": f("c_pts_trabajador"), "persona_natural": f("u_persona_natural"),
        "parte_diciembre": f("c_parte_diciembre"), "desc_generica": f("c_desc_generica"), "desc_repetida": f("c_desc_repetida"),
        "frac_objeto": f("s_obj_frac_parte"), "redondo_100": f("s_redondo_100"), "monto_repetido": f("s_monto_repetido"),
        "precio_z": f("s_precio_z"), "fecha_inusual": f("s_fecha_inusual")}, index=d.index)


def banderas(d):
    """Las 14 banderas del CRI, con el umbral fijado en HALLAZGOS 150 (el vacío no enciende la bandera)."""
    f = lambda c: d[c].astype(float)  # noqa: E731
    return pd.DataFrame({
        "cerca_tope": f("c_cerca_tope") > 0, "pasa_tope_30d": f("c_pasa_tope_30d") > 0, "mes_supera_tope": f("c_meses_superan_tope") > 0,
        "varias_mismo_dia": f("c_max_mismo_dia") > 1, "frac_objeto": f("s_obj_frac_parte") >= 0.5,
        "diciembre": f("c_parte_diciembre") >= 0.5, "fecha_inusual": f("s_fecha_inusual") > 0,
        "proveedor_nuevo": f("p_nuevo") == 1, "ruc_reciente": f("c_ruc_reciente_pct") >= 0.9,
        "desc_generica": f("c_desc_generica") >= 0.5, "precio_atipico": f("s_precio_atipico") > 0,
        "redondo_1000": f("s_redondo_1000") >= 0.5, "monto_repetido": f("s_monto_repetido") >= 0.5,
        "benford_1d": f("s_benford_1d") > 0.015}, index=d.index).astype(float)


class BayesTramos:
    """Bayes ingenuo con tramos por quintiles: el puntaje es la suma de los log de la razón de verosimilitud."""

    def __init__(self, X, y):
        y = np.asarray(y, dtype=bool)
        self.bordes, self.llr = {}, {}
        for c in X.columns:
            v = X[c].to_numpy(float)
            b = np.unique(np.quantile(v, [0.2, 0.4, 0.6, 0.8]))
            t = np.searchsorted(b, v, side="left")           # «left»: en una binaria, el 0 y el 1 quedan en tramos distintos
            k = len(b) + 1
            pos, neg = np.bincount(t[y], minlength=k), np.bincount(t[~y], minlength=k)
            self.bordes[c] = b
            self.llr[c] = np.log((pos + 1) / (y.sum() + k)) - np.log((neg + 1) / ((~y).sum() + k))

    def __call__(self, X):
        s = np.zeros(len(X))
        for c, b in self.bordes.items():
            s += self.llr[c][np.searchsorted(b, X[c].to_numpy(float), side="left")]
        return s

    def tabla(self):
        return {c: {"bordes": [round(float(x), 4) for x in b], "puntos": [round(float(x), 3) for x in self.llr[c]]}
                for c, b in self.bordes.items()}


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
        ref = {c: float(tr[c].median()) for c in BENFORD}
        ta = tr[tr.groupby("grupo")["y_rev"].transform("any")]
        # 1) Isolation Forest en el año de prueba, sin etiquetas
        Xi = anomalia(te)
        iso = IsolationForest(n_estimators=200, max_samples=256, random_state=0, n_jobs=4).fit(Xi)
        s_if = -iso.score_samples(Xi)
        # 2) CRI: banderas elegidas en el entrenamiento
        bt, elegidas = banderas(ta), []
        for c in bt.columns:
            a, ic = auc_dentro(ta["y_rev"].to_numpy(), bt[c].to_numpy(), ta["grupo"].to_numpy(), ta["ruc_entidad"].to_numpy())
            aucs.append({"corte": te_a, "bandera": c, "auc": round(a, 4), "ic90": [round(ic[0], 4), round(ic[1], 4)],
                         "encendida_pct": round(100 * float(bt[c].mean()), 2), "entra": bool(ic[0] > 0.5)})
            if ic[0] > 0.5:
                elegidas.append(c)
        s_cri = banderas(te)[elegidas].mean(axis=1).to_numpy() if elegidas else np.zeros(len(te))
        paso(f"   banderas del CRI que entran: {elegidas or 'ninguna'}")
        # 3) Bayes ingenuo con tramos
        fx = lambda d: pd.concat([variables(d), simples(d, ref)], axis=1)  # noqa: E731
        nb = BayesTramos(fx(ta), ta["y_rev"].to_numpy())
        te = te.astype({"ruc_entidad": str, "ruc": str, "anio": int})
        s = {"monto": te["c_monto"].to_numpy(dtype=float), "riesgo_rustico": riesgo(te),
             "isolation_forest": s_if, "cri": s_cri, "bayes_ingenuo": nb(fx(te))}
        for n in ("monto", "isolation_forest", "cri", "bayes_ingenuo"):
            te[f"s_{n}"] = s[n]
        s["monto_y_if"] = rrf(te, ["monto", "isolation_forest"])
        s["monto_y_cri"] = rrf(te, ["monto", "cri"])
        s["monto_y_bayes"] = rrf(te, ["monto", "bayes_ingenuo"])
        prev = pd.read_parquet(OUT / f"puntajes_d5_{te_a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        te = te.merge(prev[K + ["s_D1_ord"]], on=K, how="left")
        s["R2"] = rrf(te, ["monto", "D1_ord"])
        fila = {"prueba": te_a, "cri_banderas": elegidas, "bayes_puntos": nb.tabla()}
        for n, v in s.items():
            fila[n] = {k_: round(x_, 4) if isinstance(x_, float) else x_ for k_, x_ in metricas(te, v, y="y_rev").items()}
            fila[n]["precision_k"] = {t: {k_: round(x_, 4) for k_, x_ in metricas_k(te, v, y=c).items()} for t, c in TAREAS.items()}
            pk = fila[n]["precision_k"]
            paso(f"   {n:<18} hallazgos: P@5 {pk['revision_AB']['P@5_ent']:.4f} MAP {fila[n]['MAP_ent']:.4f} | amplia {pk['amplia']['P@5_ent']:.4f} "
                 f"| grave {pk['estricta_nivel_3']['P@5_ent']:.4f} | impedimento {pk['impedimentos_C']['P@5_ent']:.4f}")
        res["cortes"].append(fila)
        puntajes[te_a] = (te[["grupo", "y", "y_estricta", "y_rev", "y_imp", "c_monto"]].copy(), s)
    res["auc_banderas_entrenamiento"] = aucs
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
    nombres = ("monto", "riesgo_rustico", "R2") + CANDIDATAS
    p5 = {n: float(metricas_k(te_all, te_all[f"s_{n}"].to_numpy(), y="y_rev")["P@5_ent"]) for n in nombres}
    base = te_all.groupby("grupo")["y_rev"].agg(["sum", "size"])
    base = base[base["sum"] > 0]
    p5["azar"] = float((base["sum"] / base["size"]).mean())
    p5["auditor promedio"] = sum(w * p5[m] for m, w in PESOS_AUDITOR.items())
    res["p5_hallazgos_conjunto"] = p5
    print("   P@5 hallazgos, tres cortes juntos:", {k: round(v, 4) for k, v in p5.items()}, flush=True)
    (OUT / "d12_simples2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# Isolation Forest, índice simple (CRI) y Bayes ingenuo para los hallazgos del control (150)", "",
         "## 1. Banderas del CRI: AUC dentro de la entidad-año en los años de entrenamiento", "",
         "| Bandera | " + " | ".join(f"Entrenamiento del corte {c[1]}" for c in CORTES) + " |", "|---|" + "---|" * len(CORTES)]
    for b in dict.fromkeys(r["bandera"] for r in aucs):
        rr = {r["corte"]: r for r in aucs if r["bandera"] == b}
        L.append(f"| {b} | " + " | ".join(f"{rr[c[1]]['auc']:.3f} [{rr[c[1]]['ic90'][0]:.3f}, {rr[c[1]]['ic90'][1]:.3f}]"
                                         f"{' (entra)' if rr[c[1]]['entra'] else ''}; {rr[c[1]]['encendida_pct']:.1f} %" for c in CORTES) + " |")
    L += ["", "## 2. Prueba", "", "| Corte | Puntaje | Hallazgos P@5 | MAP hallazgos | Amplia | Grave | Impedimento |", "|---|---|---|---|---|---|---|"]
    for f in res["cortes"]:
        for n in nombres:
            pk = f[n]["precision_k"]
            L.append(f"| {f['prueba']} | {n} | {pk['revision_AB']['P@5_ent']:.4f} | {f[n]['MAP_ent']:.4f} | {pk['amplia']['P@5_ent']:.4f} | "
                     f"{pk['estricta_nivel_3']['P@5_ent']:.4f} | {pk['impedimentos_C']['P@5_ent']:.4f} |")
    L += ["", "| Comparación (MAP hallazgos) | Diferencia | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for k_, d_ in res["comparaciones"].items():
        L.append(f"| {k_} | {d_['dif_MAP']:+.4f} | [{d_['ic90'][0]:+.4f}, {d_['ic90'][1]:+.4f}] | {d_['por_corte']} |")
    L += ["", "| Vara (frente al monto) | Candidata | Monto | Razón | IC al 90 % |", "|---|---|---|---|---|"]
    for k_, r in res["vara"].items():
        L.append(f"| {k_} | {r['modelo']:.4f} | {r['monto']:.4f} | {r['razon']:.2f} | [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}] |")
    L += ["", "| P@5 hallazgos, tres cortes juntos | Valor |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in sorted(p5.items(), key=lambda t: -t[1])]
    (OUT / "d12_simples2.md").write_text("\n".join(L), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
