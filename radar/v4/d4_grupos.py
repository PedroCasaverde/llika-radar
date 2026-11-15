"""Pesos por grupo, encogidos hacia los globales (HALLAZGOS 131; Pedro: «vamos con 131… con o sin suma ponderada, solo
boosting con o sin pesos diferenciados»). Configuración fijada antes de correr.

Etiqueta amplia («señal adversa documentada», 130) con la tarjeta territorial (128) y la familia de las declaraciones
(132) a la fecha; cortes 2020, 2021 y 2023. D1 sin la familia se corre aparte para medir lo que aporta.
Grupos por tipo de entidad: distrital, provincial, regional, nacional (ministerio, programa o proyecto, salud y
educación) y otras (otra y universidad).
- D1: boosting con pesos globales (protocolo de d1_linea_base).
- D1 por grupo: D1 global y, en cada grupo con 200 o más positivos de entrenamiento, hasta 100 rondas más que parten del
  puntaje global (tasa 0,05, profundidad 3, 6 hojas, 500 casos por hoja); los demás grupos se quedan con el global.
- A: suma ponderada con pesos globales (d3_capas.Lineal).
- A por grupo: A más las variables estandarizadas × indicador de grupo, escaladas por 0,5 (con L2, 4 veces más
  penalizadas que los pesos globales).

    python -I -X utf8 d4_grupos.py
Salida: resultados/d4_grupos.json y .md, y los puntajes de la prueba en resultados/puntajes_d4_<año>.parquet.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_linea_base import CORTES, HIST, d1, exposicion, monotonia  # noqa: E402
from d3_capas import NEG, Lineal, peso_real  # noqa: E402
from etiquetas import Hechos  # noqa: E402
from evaluacion import Transformador, armar_corte, columnas, diferencia_conjunta, metricas, metricas_k, razon_conjunta  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
GRUPOS = ["distrital", "provincial", "regional", "nacional", "otras"]
ESCALA, MIN_POS, RONDAS_GRUPO = 0.5, 200, 100
TAREAS = {"amplia": "y", "estricta": "y_estricta", "revision_AB": "y_rev", "impedimentos_C": "y_imp", "proveedor_SP": "y_sp"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=6, max_depth=3, min_child_samples=200, bagging_fraction=0.5,
              bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, monotone_constraints_method="advanced", verbose=-1,
              num_threads=12, seed=0)
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def grupo(d):
    t = d["e_tipo"].astype(str).str.lower()
    return np.select([t.str.contains("distrital"), t.str.contains("provincial"), t.str.contains("regional"),
                      t.str.contains("ministerio|programa|salud|educ")], GRUPOS[:4], "otras")


def d1_y_grupos(tr, te, cols, valid_anio, g_tr, g_te):
    """D1 (igual que d1_linea_base.d1) y su versión por grupo, que parte del puntaje global de cada caso."""
    w = np.where(tr["y"], tr["peso"].clip(lower=0.25), 1.0)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(exposicion(tr), tr["y"], sample_weight=w)
    m_tr, m_te = lr.decision_function(exposicion(tr)), lr.decision_function(exposicion(te))
    T = Transformador(tr, cols)
    Xtr, Xte = T(tr), T(te)
    p = dict(PARAMS, monotone_constraints=monotonia(Xtr.columns))
    y = tr["y"].to_numpy()
    fit, val = (tr["anio"] != valid_anio).to_numpy(), (tr["anio"] == valid_anio).to_numpy()
    dtr = lgb.Dataset(Xtr[fit], y[fit], weight=w[fit], init_score=m_tr[fit])
    dva = lgb.Dataset(Xtr[val], y[val], weight=w[val], init_score=m_tr[val], reference=dtr)
    b = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    s_tr, s_te = m_tr + b.predict(Xtr, raw_score=True), m_te + b.predict(Xte, raw_score=True)
    s_gr, usados = s_te.copy(), {}
    pg = dict(p, min_child_samples=500)
    for k in GRUPOS:
        mt, me = g_tr == k, g_te == k
        npos = int(y[mt].sum())
        usados[k] = npos
        if npos < MIN_POS or not me.any():
            continue
        bk = lgb.train(pg, lgb.Dataset(Xtr[mt], y[mt], weight=w[mt], init_score=s_tr[mt]), num_boost_round=RONDAS_GRUPO)
        s_gr[me] = s_te[me] + bk.predict(Xte[me], raw_score=True)
    return s_te, s_gr, b.best_iteration, usados


class LinealGrupos(Lineal):
    """Suma ponderada con pesos globales más desviaciones por grupo (variables estandarizadas × indicador × 0,5)."""

    def __init__(self, tr, cols, y, peso, g, semilla=0):
        from evaluacion import CATEGORICAS
        num = [c for c in cols if c not in CATEGORICAS and c in tr.columns]
        self.T = Transformador(tr, num)
        self.niveles = {c: [v for v, k in tr[c].astype(str).value_counts().items() if k >= 1000] for c in CATEGORICAS if c in cols}
        rng = np.random.default_rng(semilla)
        y = np.asarray(y, bool)
        m = y | (rng.random(len(tr)) < NEG)
        w = np.where(y, peso, 1.0 / NEG)[m]
        Z = self.T(tr[m])
        self.med = Z.median()
        falt = Z.isna().mean()
        self.ind = falt[falt > 0.01].index.tolist()
        M = self._crudo(tr[m], Z)
        self.mu, sd = M.mean(axis=0), M.std(axis=0)
        self.ok = sd > 0
        self.sd = np.where(self.ok, sd, 1.0)
        X = self._con_grupos(((M - self.mu) / self.sd)[:, self.ok].astype(np.float32), g[m])
        self.lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y[m], sample_weight=w)
        self.n_iter = int(self.lr.n_iter_[0])

    @staticmethod
    def _con_grupos(X, g):
        return np.hstack([X] + [X * ((g == k)[:, None] * ESCALA).astype(np.float32) for k in GRUPOS])

    def puntaje(self, d, g, bloque=200_000):
        w, b = self.lr.coef_[0], self.lr.intercept_[0]
        out = []
        for i in range(0, len(d), bloque):
            X = (((self._crudo(d.iloc[i:i + bloque]) - self.mu) / self.sd)[:, self.ok]).astype(np.float32)
            out.append(self._con_grupos(X, g[i:i + bloque]) @ w + b)
        return np.concatenate(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    familia = columnas(clases=("familia a la fecha de decisión",))           # HALLAZGOS 132
    cols = columnas() + HIST + columnas(clases=("territorial a la fecha",)) + familia
    paso(f"etiqueta amplia, {len(cols)} variables (con la tarjeta territorial y la familia a la fecha: {familia})")
    res, puntajes = {"cortes": []}, {}
    for tr_a, te_a in CORTES:
        paso(f"armando el corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        fu = te["fuentes"].fillna("").astype(str)
        te["y_rev"], te["y_imp"], te["y_sp"] = (fu.str.contains(p).to_numpy() for p in ("A|B", "C", "S|P"))
        te["con_antecedentes"] = (te[[c for c in HIST if c.startswith("h_prov_")]] > 0).any(axis=1).to_numpy()
        g_tr, g_te = grupo(tr), grupo(te)
        y, w = tr["y"].to_numpy(bool), peso_real(tr)
        paso(f"   entrenamiento {len(tr):,} ({int(y.sum()):,} positivos amplios); prueba {len(te):,} ({int(te['y'].sum()):,}); "
             f"positivos de entrenamiento por grupo: {pd.Series(y).groupby(g_tr).sum().astype(int).to_dict()}")
        s = {"monto": te["c_monto"].to_numpy(dtype=float)}
        s["D1"], s["D1_por_grupo"], it, usados = d1_y_grupos(tr, te, cols, tr_a[-1], g_tr, g_te)
        _, s["D1_sin_familia"], _ = d1(tr, te, [c for c in cols if c not in familia], valid_anio=tr_a[-1])
        paso(f"   casos de prueba con familiar visible: {int((te['f_familiar_ordenes'] > 0).sum()):,}")
        paso(f"   D1 ({it} árboles) y D1 por grupo (grupos con {MIN_POS}+ positivos: {[k for k, n in usados.items() if n >= MIN_POS]})")
        A = Lineal(tr, cols, y, w)
        s["A_suma_ponderada"] = A.puntaje(te)
        AG = LinealGrupos(tr, cols, y, w, g_tr)
        s["A_por_grupo"] = AG.puntaje(te, g_te)
        paso(f"   A ({A.n_iter} iteraciones) y A por grupo ({AG.n_iter} iteraciones)")
        fila = {"entrena": tr_a, "prueba": te_a, "positivos_entrenamiento": int(y.sum()), "positivos_prueba": int(te["y"].sum()),
                "positivos_entrenamiento_por_grupo": usados}
        for n, v in s.items():
            fila[n] = {k: round(x, 4) if isinstance(x, float) else x for k, x in metricas(te, v).items()}
            fila[n]["precision_k"] = {t: {k: round(x, 4) for k, x in metricas_k(te, v, y=c).items()} for t, c in TAREAS.items()}
            top = te.assign(_s=v).sort_values(["grupo", "_s"], ascending=[True, False]).groupby("grupo").head(5)
            ac = top[top["y"]]
            fila[n]["aciertos_top5_sin_antecedentes"] = round(float((~ac["con_antecedentes"]).mean()), 4) if len(ac) else None
            paso(f"   {n:<18} MAP {fila[n]['MAP_ent']:.4f} | P@5 {fila[n]['P@5_ent']:.4f} | MAP quintil {fila[n]['MAP_quintil_monto']:.4f} "
                 f"| P@1000 {fila[n]['P@1000_nacional']:.3f} | aciertos sin antecedentes {fila[n]['aciertos_top5_sin_antecedentes']}")
        res["cortes"].append(fila)
        puntajes[te_a] = (te[["grupo", "y", "y_estricta", "y_rev", "y_imp", "y_sp", "c_monto"]].copy(), s)
        te[["grupo", "ruc_entidad", "ruc", "anio", "y", "y_estricta", "y_rev", "y_imp", "y_sp", "con_antecedentes", "c_monto"]] \
            .assign(grupo_tipo=g_te, **{f"s_{n}": v for n, v in s.items()}).to_parquet(OUT / f"puntajes_d4_{te_a}.parquet", index=False)
    paso("diferencias conjuntas y razones frente al monto (IC al 90 %)")
    nombres = ["D1", "D1_por_grupo", "A_suma_ponderada", "A_por_grupo"]
    res["comparaciones"] = {f"{a} − {b}": diferencia_conjunta([(t, s_[a], s_[b]) for t, s_ in puntajes.values()])
                            for a, b in [(n, "monto") for n in nombres] + [("D1_por_grupo", "D1"), ("A_por_grupo", "A_suma_ponderada"),
                                                                            ("D1", "D1_sin_familia")]}
    for k, d in res["comparaciones"].items():
        print(f"   {k}: {d}", flush=True)
    res["razon_frente_al_monto"] = {}
    for n in nombres:
        for t, c in TAREAS.items():
            for medida, q in (("AP", False), ("P@5", False), ("AP", True), ("P@5", True)):
                clave = f"{n} | {t} | {medida}{' en rangos de monto' if q else ''}"
                r = razon_conjunta([(t_, s_[n], s_["monto"]) for t_, s_ in puntajes.values()], medida=medida, quintil=q, y=c)
                res["razon_frente_al_monto"][clave] = r
                print(f"   {clave:<55} {r['modelo']:.4f} / {r['monto']:.4f} = {r['razon']:.2f} [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}]", flush=True)
    (OUT / "d4_grupos.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# Pesos por grupo (etiqueta amplia, con la tarjeta territorial)", "",
         "| Corte | Puntaje | MAP por entidad | P en 5 | MAP en quintil | P en 1 000 | Aciertos sin antecedentes |", "|---|---|---|---|---|---|---|"]
    for f in res["cortes"]:
        for n in ["monto"] + nombres:
            m = f[n]
            L.append(f"| {f['prueba']} | {n} | {m['MAP_ent']:.4f} | {m['P@5_ent']:.4f} | {m['MAP_quintil_monto']:.4f} | "
                     f"{m['P@1000_nacional']:.3f} | {m['aciertos_top5_sin_antecedentes']} |")
    L += ["", "| Comparación | Diferencia de MAP | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for k, d in res["comparaciones"].items():
        L.append(f"| {k} | {d['dif_MAP']:+.4f} | [{d['ic90'][0]:+.4f}, {d['ic90'][1]:+.4f}] | {d['por_corte']} |")
    L += ["", "| Candidata, tarea y medida | Candidata | Monto | Razón | IC al 90 % |", "|---|---|---|---|---|"]
    for k, r in res["razon_frente_al_monto"].items():
        L.append(f"| {k} | {r['modelo']:.4f} | {r['monto']:.4f} | {r['razon']:.2f} | [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}] |")
    (OUT / "d4_grupos.md").write_text("\n".join(L), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
