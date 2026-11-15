"""Regresión logística condicional para los hallazgos del control (HALLAZGOS 147). Fijada antes de correr.

Para cada hallazgo (A o B) de una entidad-año auditada del entrenamiento se arma un conjunto emparejado: el caso con
hallazgo y 20 casos sin hallazgo de la MISMA entidad-año, sorteados (semilla 0). La logística condicional estima qué hace
que el auditor encuentre ese caso y no otro de la misma entidad:
    P(el hallado es i | conjunto) = exp(x_i·β) / Σ_j exp(x_j·β)
Se ajusta por máxima verosimilitud con una penalización L2 muy chica (1e-3), con variables estandarizadas. El puntaje de
prueba es la suma ponderada x·β; los pesos se reportan como razón de odds por desviación estándar.

Variables fijadas por criterio del dominio (HALLAZGOS 147): log monto, log órdenes, log contratación del proveedor con el
Estado, órdenes cerca del tope, pasó el tope en 30 días, varias órdenes el mismo día, algún mes superó el tope, parte de
servicios, parecido a trabajador, persona natural y los once rubros.

    python -I -X utf8 d10_regresion.py
Salida: resultados/d10_regresion.json y .md
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combinaciones import rrf  # noqa: E402
from comparacion_final import PESOS_AUDITOR, riesgo  # noqa: E402
from d1_linea_base import CORTES  # noqa: E402
from etiquetas import Hechos, K  # noqa: E402
from evaluacion import armar_corte, diferencia_conjunta, metricas, metricas_k, razon_conjunta  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
CONTROLES, L2 = 20, 1e-3
RUBROS = ["c_rub_obras", "c_rub_consultoria", "c_rub_alimentos", "c_rub_combustible", "c_rub_salud", "c_rub_limpieza_seguridad",
          "c_rub_alquiler_transporte", "c_rub_servicios_personales", "c_rub_tecnologia", "c_rub_utiles", "c_rub_publicidad_eventos"]
TAREAS = {"revision_AB": "y_rev", "amplia": "y", "estricta_nivel_3": "y_estricta", "impedimentos_C": "y_imp"}
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def variables(d):
    f = lambda c: d[c].astype(float).fillna(0)  # noqa: E731
    X = pd.DataFrame({
        "log_monto": np.log1p(f("c_monto")),
        "log_ordenes": np.log1p(f("c_ordenes")),
        "log_proveedor_estado": np.log1p(f("p_monto") + f("p_exp_ocds_monto") + f("p_cat_monto")),
        "cerca_tope": (f("c_cerca_tope") > 0).astype(float),
        "pasa_tope_30d": (f("c_pasa_tope_30d") > 0).astype(float),
        "varias_mismo_dia": (f("c_max_mismo_dia") > 1).astype(float),
        "mes_supera_tope": (f("c_meses_superan_tope") > 0).astype(float),
        "parte_servicios": f("c_parte_servicios"),
        "parecido_trabajador": f("c_pts_trabajador"),
        "persona_natural": d["u_persona_natural"].astype(float).fillna(0),
    }, index=d.index)
    for r in RUBROS:
        X[r.replace("c_rub_", "rubro_")] = f(r)
    return X


def ajustar(tr, fx=variables):
    """Conjuntos emparejados 1:20 dentro de la entidad-año y logística condicional con L2 chica. `fx` arma las
    variables (d11 le suma los estadísticos de 149)."""
    rng = np.random.default_rng(0)
    con = tr.groupby("grupo")["y_rev"].transform("any").to_numpy()
    d = tr[con]
    X = fx(d)
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).to_numpy()
    filas, conjuntos = [], 0
    for _, idx in d.groupby("grupo").indices.items():
        y = d["y_rev"].to_numpy()[idx]
        pos, neg = idx[y], idx[~y]
        if len(neg) == 0:
            continue
        for p in pos:
            ctrl = rng.choice(neg, size=CONTROLES, replace=len(neg) < CONTROLES)
            filas.append(np.concatenate([[p], ctrl]))
            conjuntos += 1
    S = np.stack(filas)                                     # conjuntos × 21 (el primero es el hallado)
    Zs = Z[S]                                               # conjuntos × 21 × variables

    def perdida(b):
        s = Zs @ b
        m = s.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
        p = np.exp(s - lse[:, None])
        val = -(s[:, 0] - lse).sum() + L2 * conjuntos * (b @ b)
        grad = -(Zs[:, 0, :] - (p[:, :, None] * Zs).sum(axis=1)).sum(axis=0) + 2 * L2 * conjuntos * b
        return val, grad

    r = minimize(perdida, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
    beta = pd.Series(r.x, index=X.columns)
    return beta, mu, sd, conjuntos, bool(r.success)


def main():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    res, puntajes = {"cortes": []}, {}
    for tr_a, te_a in CORTES:
        paso(f"armando el corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        for d in (tr, te):
            fu = d["fuentes"].fillna("").astype(str)
            d["y_rev"], d["y_imp"] = fu.str.contains("A|B").to_numpy(), fu.str.contains("C").to_numpy()
        beta, mu, sd, conjuntos, ok = ajustar(tr)
        te = te.astype({"ruc_entidad": str, "ruc": str, "anio": int})
        s = {"monto": te["c_monto"].to_numpy(dtype=float), "riesgo_rustico": riesgo(te)}
        s["regresion_control"] = (((variables(te) - mu) / sd) @ beta).to_numpy()
        te["s_monto"], te["s_regresion_control"] = s["monto"], s["regresion_control"]
        s["R2_regresion"] = rrf(te, ["monto", "regresion_control"])
        prev = pd.read_parquet(OUT / f"puntajes_d5_{te_a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        te = te.merge(prev[K + ["s_D1_ord"]], on=K, how="left")
        s["R2"] = rrf(te, ["monto", "D1_ord"])
        odds = np.exp(beta).round(3).to_dict()
        paso(f"   {conjuntos:,} conjuntos emparejados; convergió: {ok}; razón de odds por desviación estándar: "
             + ", ".join(f"{k} {v}" for k, v in sorted(odds.items(), key=lambda t: -abs(np.log(t[1])))[:8]))
        fila = {"prueba": te_a, "conjuntos": conjuntos, "convergio": ok, "razon_odds": odds}
        for n, v in s.items():
            fila[n] = {k_: round(x_, 4) if isinstance(x_, float) else x_ for k_, x_ in metricas(te, v, y="y_rev").items()}
            fila[n]["precision_k"] = {t: {k_: round(x_, 4) for k_, x_ in metricas_k(te, v, y=c).items()} for t, c in TAREAS.items()}
            pk = fila[n]["precision_k"]
            paso(f"   {n:<18} hallazgos: P@5 {pk['revision_AB']['P@5_ent']:.4f} MAP {fila[n]['MAP_ent']:.4f} | amplia {pk['amplia']['P@5_ent']:.4f} "
                 f"| grave {pk['estricta_nivel_3']['P@5_ent']:.4f} | impedimento {pk['impedimentos_C']['P@5_ent']:.4f}")
        res["cortes"].append(fila)
        puntajes[te_a] = (te[["grupo", "y", "y_estricta", "y_rev", "y_imp", "c_monto"]].copy(), s)
    paso("vara frente al monto en hallazgos (IC al 90 %)")
    res["comparaciones"], res["vara"] = {}, {}
    for a in ("regresion_control", "R2_regresion"):
        d_ = diferencia_conjunta([(t.assign(y=t["y_rev"]), s_[a], s_["monto"]) for t, s_ in puntajes.values()])
        res["comparaciones"][f"{a} − monto"] = d_
        print(f"   {a} − monto (MAP hallazgos): {d_}", flush=True)
        for medida, q in (("P@5", False), ("AP", True)):
            r = razon_conjunta([(t, s_[a], s_["monto"]) for t, s_ in puntajes.values()], medida=medida, quintil=q, y="y_rev")
            k_ = f"{a} / monto | hallazgos | {medida}{' en rangos de monto' if q else ''}"
            res["vara"][k_] = r
            print(f"   {k_:<62} {r['modelo']:.4f} / {r['monto']:.4f} = {r['razon']:.2f} [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}]", flush=True)
    te_all = pd.concat([t.assign(**{f"s_{n}": v for n, v in s_.items()}) for t, s_ in puntajes.values()], ignore_index=True)
    p5 = {n: float(metricas_k(te_all, te_all[f"s_{n}"].to_numpy(), y="y_rev")["P@5_ent"]) for n in ("monto", "riesgo_rustico", "regresion_control", "R2_regresion", "R2")}
    base = te_all.groupby("grupo")["y_rev"].agg(["sum", "size"])
    base = base[base["sum"] > 0]
    p5["azar"] = float((base["sum"] / base["size"]).mean())
    p5["auditor promedio"] = sum(w * p5[m] for m, w in PESOS_AUDITOR.items())
    res["p5_hallazgos_conjunto"] = p5
    print("   P@5 hallazgos, tres cortes juntos:", {k: round(v, 4) for k, v in p5.items()}, flush=True)
    (OUT / "d10_regresion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# Regresión logística condicional para los hallazgos del control (147)", "",
         "| Corte | Puntaje | Hallazgos P@5 | MAP hallazgos | Amplia | Grave | Impedimento |", "|---|---|---|---|---|---|---|"]
    for f in res["cortes"]:
        for n in ("monto", "riesgo_rustico", "R2", "regresion_control", "R2_regresion"):
            pk = f[n]["precision_k"]
            L.append(f"| {f['prueba']} | {n} | {pk['revision_AB']['P@5_ent']:.4f} | {f[n]['MAP_ent']:.4f} | {pk['amplia']['P@5_ent']:.4f} | "
                     f"{pk['estricta_nivel_3']['P@5_ent']:.4f} | {pk['impedimentos_C']['P@5_ent']:.4f} |")
    L += ["", "| Comparación (MAP hallazgos) | Diferencia | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for k_, d_ in res["comparaciones"].items():
        L.append(f"| {k_} | {d_['dif_MAP']:+.4f} | [{d_['ic90'][0]:+.4f}, {d_['ic90'][1]:+.4f}] | {d_['por_corte']} |")
    L += ["", "| Vara (frente al monto) | Candidata | Monto | Razón | IC al 90 % |", "|---|---|---|---|---|"]
    for k_, r in res["vara"].items():
        L.append(f"| {k_} | {r['modelo']:.4f} | {r['monto']:.4f} | {r['razon']:.2f} | [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}] |")
    L += ["", "| Razón de odds por desviación estándar (corte 2023) | Valor |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(res["cortes"][-1]["razon_odds"].items(), key=lambda t: -abs(np.log(t[1])))]
    (OUT / "d10_regresion.md").write_text("\n".join(L), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
