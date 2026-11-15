"""Diseño D1 sobre la data v4: línea base honesta (HALLAZGOS 114 a 118).

Detección (hechos del año del caso que aún no se conocían al cierre de ese año), con etiquetas a la fecha:
entrenamiento con lo conocido al cierre del año de prueba (por publicación), prueba con lo conocido al 31/12/2025.
Cortes de exploración: 2018-2019 → 2020 y 2018-2020 → 2021. Regla de evaluación de HALLAZGOS 116.

Puntajes:
- monto: lo que se hace hoy;
- exposición: logística con log(monto) y log(número de órdenes), exponente libre;
- D1: LightGBM binario que parte del margen de exposición (init_score) y aprende solo el exceso de riesgo; configuración
  congelada (tasa 0,05, 6 hojas, profundidad 3, submuestreo 0,5), monotonía creciente en alertas, fraccionamiento,
  cercanía al tope e historia, parada temprana en el último año de entrenamiento;
- D1 con variables de otra época (sensibilidad) y el boosting sin margen de antes (referencia).

    python -I -X utf8 d1_linea_base.py [--prueba]
Salida: datos/procesados/v4/resultados/d1.json y .md.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etiquetas import Hechos  # noqa: E402
from evaluacion import Transformador, armar_corte, columnas, diferencia_conjunta, metricas  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
CORTES = [([2018, 2019], 2020), ([2018, 2019, 2020], 2021), ([2018, 2019, 2020, 2021], 2023)]   # 2023 en exploración (HALLAZGOS 125)
HIST = ["h_prov_hechos", "h_prov_entidades", "h_prov_impedimento", "h_prov_sanciones_otras", "h_prov_penalidades", "h_prov_multas",
        "h_prov_inhabilitaciones", "h_prov_inhabilitado_al_cierre", "h_ent_hechos", "h_ent_proveedores", "h_rel_hechos"]
T0 = time.time()
prueba = "--prueba" in sys.argv
CONOCIMIENTO = "emision" if "--emision" in sys.argv else "publicacion"     # sensibilidad (HALLAZGOS 123)
SUFIJO = "" if CONOCIMIENTO == "publicacion" else "_emision"


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def exposicion(d):
    return np.column_stack([np.log1p(d["c_monto"].astype(float)), np.log1p(d["c_ordenes"].astype(float))])


def monotonia(cols):
    sube = ("a_", "c_pasa_tope", "c_cerca_tope", "c_meses_superan_tope", "h_prov_", "h_rel_")
    return [1 if c.startswith(sube) else 0 for c in cols]


def d1(tr, te, cols, valid_anio, rondas=None, valid_mascara=None):
    """Exposición como margen inicial y LightGBM para el exceso de riesgo. Con `rondas` (HALLAZGOS 124): número fijo de
    rondas con todo el entrenamiento, sin parada temprana en el año menos maduro. Con `valid_mascara` (HALLAZGOS 139):
    la parada temprana mira esas filas (un 20 % de entidades de todos los años) en vez del último año."""
    w = np.where(tr["y"], tr["peso"].clip(lower=0.25), 1.0)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(exposicion(tr), tr["y"], sample_weight=w)
    m_tr, m_te = lr.decision_function(exposicion(tr)), lr.decision_function(exposicion(te))
    T = Transformador(tr, cols)
    Xtr, Xte = T(tr), T(te)
    if valid_mascara is not None:
        val = np.asarray(valid_mascara, dtype=bool)
        fit = ~val
    else:
        fit, val = (tr["anio"] != valid_anio).to_numpy(), (tr["anio"] == valid_anio).to_numpy()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, bagging_fraction=0.5,
                  bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, monotone_constraints=monotonia(Xtr.columns),
                  monotone_constraints_method="advanced", verbose=-1, num_threads=12, seed=0)
    if rondas:
        b = lgb.train(params, lgb.Dataset(Xtr, tr["y"].to_numpy(), weight=w, init_score=m_tr), num_boost_round=rondas)
        return m_te, m_te + b.predict(Xte, raw_score=True), rondas
    dtr = lgb.Dataset(Xtr[fit], tr["y"].to_numpy()[fit], weight=w[fit], init_score=m_tr[fit])
    dva = lgb.Dataset(Xtr[val], tr["y"].to_numpy()[val], weight=w[val], init_score=m_tr[val], reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m_te, m_te + b.predict(Xte, raw_score=True), b.best_iteration


def sin_margen(tr, te, cols):
    T = Transformador(tr, cols)
    Xtr, Xte = T(tr), T(te)
    vac = [c for c in Xtr.columns if Xtr[c].notna().sum() < 1000 or Xtr[c].nunique() < 2]      # el binning de HGB falla con columnas vacías
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=500, l2_regularization=1.0, early_stopping=True,
                                       validation_fraction=0.15, random_state=0).fit(Xtr.drop(columns=vac), tr["y"])
    return m.predict_proba(Xte.drop(columns=vac))[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H = Hechos(conocimiento=CONOCIMIENTO)
    paso(f"fecha de conocimiento: {CONOCIMIENTO}")
    cols_contemp = columnas()
    cols_todas = columnas(clases=("contemporánea", "estática fechable (foto de 2026)", "anacrónica"))
    hist = HIST
    res, puntajes = {"cortes": []}, {}
    for tr_a, te_a in CORTES:
        paso(f"armando el corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        if prueba:
            ents = te["ruc_entidad"].drop_duplicates().sample(frac=0.1, random_state=0)
            tr, te = tr[tr["ruc_entidad"].isin(set(ents))], te[te["ruc_entidad"].isin(set(ents))]
        paso(f"   entrenamiento {len(tr):,} casos ({int(tr['y'].sum()):,} positivos a la fecha); prueba {len(te):,} ({int(te['y'].sum()):,} positivos)")
        s = {"monto": te["c_monto"].to_numpy(dtype=float)}
        m_te, s["D1"], it = d1(tr, te, cols_contemp + hist, valid_anio=tr_a[-1])
        s["exposicion"] = m_te
        _, s["D1_300"], _ = d1(tr, te, cols_contemp + hist, valid_anio=tr_a[-1], rondas=300)      # HALLAZGOS 124
        _, s["D1_con_anacronicas"], _ = d1(tr, te, cols_todas + hist, valid_anio=tr_a[-1])
        s["boosting_sin_margen"] = sin_margen(tr, te, cols_contemp + hist)
        fila = {"entrena": tr_a, "prueba": te_a, "casos_entrenamiento": len(tr), "positivos_entrenamiento": int(tr["y"].sum()),
                "casos_prueba": len(te), "positivos_prueba": int(te["y"].sum()), "iteraciones_D1": int(it)}
        for n, v in s.items():
            fila[n] = {k: round(x, 4) if isinstance(x, float) else x for k, x in metricas(te, v).items()}
            paso(f"   {n:<20} MAP {fila[n]['MAP_ent']:.4f} | P@5 {fila[n]['P@5_ent']:.4f} | acierto@5 {fila[n]['acierto@5_ent']:.3f} "
                 f"| MAP quintil {fila[n]['MAP_quintil_monto']:.4f}")
        res["cortes"].append(fila)
        puntajes[te_a] = (te[["grupo", "y", "c_monto"]].copy(), s)
    paso("diferencias conjuntas frente al monto (IC al 90 %, remuestreo de entidades entre cortes)")
    res["frente_al_monto"] = {}
    for n in ("exposicion", "D1", "D1_300", "D1_con_anacronicas", "boosting_sin_margen"):
        res["frente_al_monto"][n] = diferencia_conjunta([(te, s[n], s["monto"]) for te, s in puntajes.values()])
        print(f"   {n:<20} {res['frente_al_monto'][n]}", flush=True)
    if prueba:
        paso("prueba terminada (no se guarda)")
        return
    (OUT / f"d1{SUFIJO}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# D1 sobre la data v4 (detección, etiquetas a la fecha por publicación)", "",
         "| Corte | Puntaje | MAP por entidad | P en 5 | Acierto en 5 | MAP en quintil de monto | Aciertos por revisión |", "|---|---|---|---|---|---|---|"]
    for f in res["cortes"]:
        for n in ("monto", "exposicion", "D1", "D1_300", "D1_con_anacronicas", "boosting_sin_margen"):
            m = f[n]
            L.append(f"| {f['prueba']} | {n} | {m['MAP_ent']:.4f} | {m['P@5_ent']:.4f} | {m['acierto@5_ent']:.3f} | {m['MAP_quintil_monto']:.4f} | {m['aciertos_por_revision_todas']:.4f} |")
    L += ["", "| Puntaje | Diferencia de MAP frente al monto | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for n, d in res["frente_al_monto"].items():
        L.append(f"| {n} | {d['dif_MAP']:+.4f} | [{d['ic90'][0]:+.4f}, {d['ic90'][1]:+.4f}] | {d['por_corte']} |")
    (OUT / f"d1{SUFIJO}.md").write_text("\n".join(L), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
