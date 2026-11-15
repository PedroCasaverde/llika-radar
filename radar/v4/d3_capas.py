"""Capas A → B → C (HALLAZGOS 127; Pedro: «sí apruebo»). Configuración fijada antes de correr.

A. Suma ponderada: logística L2 (C = 1) con las variables de D1, estandarizadas; los pesos son los coeficientes.
B. A + red de PASCO: comunidades por año en la red entidad-proveedor de todo el país (sin etiquetas) y variables del
   caso; entra solo si pasa la compuerta de homofilia en el entrenamiento.
C. B + etiquetas de MaskCluster: maestro sin el bloque de monto, a lo más el 0,3 % de casos propuestos por comunidad y
   año, con medio peso; alumno con la configuración de B; puntaje final = promedio de B y del alumno.
Cortes de exploración 2020, 2021 y 2023, etiquetas por publicación, detección. 2022 no se toca.

Con --territorial (HALLAZGOS 128) todas las candidatas, D1 incluida, suman la tarjeta territorial a la fecha; además se
reporta la precisión en k (1, 3, 5 y 10) por entidad y dentro de los rangos de monto, por tarea (todas las fuentes;
revisión = A o B; impedimentos = C núcleo), la razón frente al monto con su IC (la vara del doble de Pedro) y se guardan
los puntajes de la prueba para recalcular cualquier medida sin reentrenar.

    python -I -X utf8 d3_capas.py [--territorial]
Salida: datos/procesados/v4/red/comunidades_<año>.parquet (caché) y resultados/d3_capas[_territorial].json y .md.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_linea_base import CORTES, HIST, d1  # noqa: E402
from etiquetas import Hechos, K  # noqa: E402
from evaluacion import (CATEGORICAS, Transformador, armar_corte, cargar_x, columnas, diferencia_conjunta, metricas,  # noqa: E402
                        metricas_k, razon_conjunta)
from pasco import comunidades  # noqa: E402

warnings.filterwarnings("ignore")
V4 = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4"
RED, OUT = V4 / "red", V4 / "resultados"
MONTO = ("c_monto", "c_max_suma", "c_ordenes")
RED_COLS = ["r_prov_deptos", "r_com_entidades", "r_com_deptos", "r_com_proveedores", "r_com_hist_otras"]
ENCOGER, TOPE, PESO_PSEUDO, NEG = 100, 0.003, 0.5, 0.20
TERRITORIAL = "--territorial" in sys.argv
AMPLIA = "--amplia" in sys.argv          # etiqueta «señal adversa documentada» (HALLAZGOS 130)
SUFIJO = ("_amplia" if AMPLIA else "") + ("_territorial" if TERRITORIAL else "")
TAREAS = {"todas": "y", "revision_AB": "y_rev", "impedimentos_C": "y_imp"}
if AMPLIA:
    TAREAS = {"amplia": "y", "estricta": "y_estricta", "revision_AB": "y_rev", "impedimentos_C": "y_imp", "proveedor_SP": "y_sp"}
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


# ------------------------------------------------------------------------------------------------ red de PASCO
def red_del_anio(H, a):
    """Comunidad del proveedor y variables de red de cada caso del año, con todas las órdenes del año (sin universo)."""
    f = RED / f"comunidades_{a}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    x = cargar_x([a])[K + ["c_monto", "e_depto"]]
    com = comunidades(x, R=5, rho=10)
    x["comunidad"] = ("P:" + x["ruc"].astype(str)).map(com).fillna(-1).astype(int)
    x["r_prov_deptos"] = x.groupby("ruc")["e_depto"].transform("nunique")
    g = x.groupby("comunidad")
    x["r_com_entidades"] = g["ruc_entidad"].transform("nunique")
    x["r_com_deptos"] = g["e_depto"].transform("nunique")
    x["r_com_proveedores"] = g["ruc"].transform("nunique")
    # historia de la comunidad en otras entidades: casos de la misma comunidad y año, de otra entidad y otro proveedor,
    # cuyo proveedor tiene hechos conocidos al cierre del año; encogida hacia la media del año
    x["_h"] = (H.historia(x)["h_prov_hechos"] > 0).astype(float).to_numpy()
    S, N = g["_h"].transform("sum"), g["_h"].transform("size")
    ge, gp = x.groupby(["comunidad", "ruc_entidad"])["_h"], x.groupby(["comunidad", "ruc"])["_h"]
    s_p = S - ge.transform("sum") - gp.transform("sum") + x["_h"]
    n_p = N - ge.transform("size") - gp.transform("size") + 1
    media = x["_h"].mean()
    x["r_com_hist_otras"] = (s_p + ENCOGER * media) / (n_p + ENCOGER)
    x = x[K + ["comunidad"] + RED_COLS]
    RED.mkdir(parents=True, exist_ok=True)
    x.to_parquet(f, index=False)
    paso(f"   red {a}: {x['comunidad'].nunique():,} comunidades; historia en otras entidades, media {x['r_com_hist_otras'].mean():.4f}")
    return x


def auc_dentro(y, s, grupo, entidad, n=500, semilla=0):
    """AUC dentro de cada entidad-año con positivos y negativos; media e IC al 90 % remuestreando entidades."""
    d = pd.DataFrame({"g": grupo, "e": entidad, "y": np.asarray(y, bool), "s": s})
    d["r"] = d.groupby("g")["s"].rank()
    a = d.groupby("g").agg(n1=("y", "sum"), n=("y", "size"), r1=("r", lambda r: r[d.loc[r.index, "y"]].sum()), e=("e", "first"))
    a = a[(a["n1"] > 0) & (a["n1"] < a["n"])]
    a["auc"] = (a["r1"] - a["n1"] * (a["n1"] + 1) / 2) / (a["n1"] * (a["n"] - a["n1"]))
    pe = a.groupby("e")["auc"].agg(["sum", "size"])
    rng = np.random.default_rng(semilla)
    boot = [pe.loc[rng.choice(pe.index, len(pe))].pipe(lambda m: m["sum"].sum() / m["size"].sum()) for _ in range(n)]
    return float(a["auc"].mean()), [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))]


# ------------------------------------------------------------------------------------------------ suma ponderada
class Lineal:
    """Logística L2 con C = 1 sobre variables estandarizadas; puntaje = suma ponderada (coeficientes como pesos)."""

    def __init__(self, tr, cols, y, peso, semilla=0):
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
        X = ((M - self.mu) / self.sd)[:, self.ok]
        self.lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y[m], sample_weight=w)
        self.n_iter = int(self.lr.n_iter_[0])

    def _crudo(self, d, Z=None):
        Z = self.T(d) if Z is None else Z
        partes = [Z.fillna(self.med).to_numpy(np.float32), Z[self.ind].isna().to_numpy(np.float32)]
        for c, nv in self.niveles.items():
            v = d[c].astype(str).to_numpy()
            partes.append(np.column_stack([v == x for x in nv]).astype(np.float32) if nv else np.zeros((len(d), 0), np.float32))
        return np.hstack(partes)

    def puntaje(self, d, bloque=400_000):
        w, b = self.lr.coef_[0], self.lr.intercept_[0]
        out = [(((self._crudo(d.iloc[i:i + bloque]) - self.mu) / self.sd)[:, self.ok] @ w) + b for i in range(0, len(d), bloque)]
        return np.concatenate(out)


def peso_real(tr):
    return np.where(tr["y"], tr["peso"].clip(lower=0.25), 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H = Hechos(conocimiento="publicacion")
    if AMPLIA:
        H.etiqueta = H.etiqueta_amplia          # armar_corte usa la etiqueta amplia en entrenamiento y prueba
    anios = sorted({a for tr_a, te_a in CORTES for a in tr_a + [te_a]})
    paso(f"red de PASCO por año: {anios}")
    red = pd.concat([red_del_anio(H, a) for a in anios], ignore_index=True)
    red = red.astype({"ruc_entidad": str, "ruc": str, "anio": int})
    base = columnas() + HIST + (columnas(clases=("territorial a la fecha",)) if TERRITORIAL else [])
    paso(f"variables: {len(base)}{' (con la tarjeta territorial a la fecha)' if TERRITORIAL else ''}")
    res, puntajes = {"cortes": []}, {}
    for tr_a, te_a in CORTES:
        paso(f"armando el corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        tr = tr.astype({"ruc_entidad": str, "ruc": str, "anio": int}).merge(red, on=K, how="left")
        te = te.astype({"ruc_entidad": str, "ruc": str, "anio": int}).merge(red, on=K, how="left")
        fu = te["fuentes"].fillna("").astype(str)
        te["y_rev"], te["y_imp"] = fu.str.contains("A|B").to_numpy(), fu.str.contains("C").to_numpy()
        te["y_sp"] = fu.str.contains("S|P").to_numpy()
        if "y_estricta" not in te:
            te["y_estricta"] = te["y"]
        te["con_antecedentes"] = (te[[c for c in HIST if c.startswith("h_prov_")]] > 0).any(axis=1).to_numpy()
        paso(f"   entrenamiento {len(tr):,} ({int(tr['y'].sum()):,} positivos); prueba {len(te):,} ({int(te['y'].sum()):,}); "
             f"sin comunidad: {tr['comunidad'].isna().mean():.2%} / {te['comunidad'].isna().mean():.2%}")
        y, w = tr["y"].to_numpy(bool), peso_real(tr)
        s = {"monto": te["c_monto"].to_numpy(dtype=float)}
        _, s["D1"], _ = d1(tr, te, base, valid_anio=tr_a[-1])
        A = Lineal(tr, base, y, w)
        s["A_suma_ponderada"] = A.puntaje(te)
        paso(f"   A lista ({A.n_iter} iteraciones)")
        auc, ic = auc_dentro(y, tr["r_com_hist_otras"].fillna(0).to_numpy(), tr["grupo"].to_numpy(), tr["ruc_entidad"].to_numpy())
        pasa = ic[0] > 0.5
        paso(f"   compuerta de homofilia (entrenamiento): AUC dentro de entidad {auc:.3f} [{ic[0]:.3f}, {ic[1]:.3f}] → {'pasa' if pasa else 'no pasa'}")
        cols_b = base + RED_COLS if pasa else base
        if pasa:
            B = Lineal(tr, cols_b, y, w)
            s["B_red_PASCO"] = B.puntaje(te)
            paso(f"   B lista ({B.n_iter} iteraciones)")
        else:
            B = A
        maestro = Lineal(tr, [c for c in cols_b if not c.startswith(MONTO)], y, w)
        sm = maestro.puntaje(tr)
        d = pd.DataFrame({"c": tr["comunidad"].fillna(-1).astype(int).to_numpy(), "a": tr["anio"].to_numpy(), "s": sm, "y": y})
        pseudo = np.zeros(len(tr), dtype=bool)
        for _, g in d.groupby(["c", "a"]):
            tope = max(1, int(TOPE * len(g)))
            pseudo[g[~g["y"]].nlargest(tope, "s").index] = True
        y2 = y | pseudo
        w2 = np.where(y, w, np.where(pseudo, PESO_PSEUDO, 1.0))
        alumno = Lineal(tr, cols_b, y2, w2)
        s["C_MaskCluster"] = 0.5 * B.puntaje(te) + 0.5 * alumno.puntaje(te)
        paso(f"   C lista: {int(pseudo.sum()):,} etiquetas propuestas frente a {int(y.sum()):,} reales")
        fila = {"entrena": tr_a, "prueba": te_a, "homofilia": {"auc": round(auc, 4), "ic90": [round(v, 4) for v in ic], "pasa": pasa},
                "etiquetas_propuestas": int(pseudo.sum()), "positivos_entrenamiento": int(y.sum()), "positivos_prueba": int(te["y"].sum())}
        for n, v in s.items():
            fila[n] = {k: round(x, 4) if isinstance(x, float) else x for k, x in metricas(te, v).items()}
            fila[n]["precision_k"] = {t: {k: round(x, 4) for k, x in metricas_k(te, v, y=c).items()} for t, c in TAREAS.items()}
            paso(f"   {n:<18} MAP {fila[n]['MAP_ent']:.4f} | P@5 {fila[n]['P@5_ent']:.4f} | acierto@5 {fila[n]['acierto@5_ent']:.3f} "
                 f"| MAP quintil {fila[n]['MAP_quintil_monto']:.4f}")
        if pasa:   # pesos de la suma ponderada de B (estandarizados), para leerlos
            nombres = (B.T.cols + [f"{c}_falta" for c in B.ind] + [f"{c}={v}" for c, nv in B.niveles.items() for v in nv])
            pesos = pd.Series(B.lr.coef_[0], index=np.array(nombres)[B.ok]).sort_values(key=np.abs, ascending=False)
            fila["pesos_B_top20"] = pesos.head(20).round(3).to_dict()
        res["cortes"].append(fila)
        puntajes[te_a] = (te[["grupo", "y", "y_estricta", "y_rev", "y_imp", "y_sp", "c_monto"]].copy(), s)
        guardar = te[["grupo", "ruc_entidad", "ruc", "anio", "y", "y_estricta", "y_rev", "y_imp", "y_sp", "con_antecedentes", "c_monto"]].assign(**{f"s_{n}": v for n, v in s.items()})
        guardar.to_parquet(OUT / f"puntajes_d3{SUFIJO}_{te_a}.parquet", index=False)
    paso("diferencias conjuntas (IC al 90 %, remuestreo de entidades entre cortes)")
    comunes = [n for n in ("D1", "A_suma_ponderada", "B_red_PASCO", "C_MaskCluster") if all(n in s for _, s in puntajes.values())]
    pares = [(n, "monto") for n in comunes] + [(a, b) for a, b in (("A_suma_ponderada", "D1"), ("B_red_PASCO", "A_suma_ponderada"),
                                                                  ("C_MaskCluster", "B_red_PASCO"), ("C_MaskCluster", "A_suma_ponderada"))
                                               if a in comunes and b in comunes]
    res["comparaciones"] = {}
    for a, b in pares:
        res["comparaciones"][f"{a} − {b}"] = diferencia_conjunta([(t, s[a], s[b]) for t, s in puntajes.values()])
        print(f"   {a} − {b}: {res['comparaciones'][f'{a} − {b}']}", flush=True)
    paso("razón frente al monto (la vara del doble), por tarea y medida, con IC al 90 %")
    res["razon_frente_al_monto"] = {}
    for n in comunes:
        for t, c in TAREAS.items():
            for medida, q in (("AP", False), ("P@5", False), ("AP", True), ("P@5", True)):
                clave = f"{n} | {t} | {medida}{' en rangos de monto' if q else ''}"
                r = razon_conjunta([(te_, s_[n], s_["monto"]) for te_, s_ in puntajes.values()], medida=medida, quintil=q, y=c)
                res["razon_frente_al_monto"][clave] = r
                print(f"   {clave:<60} {r['modelo']:.4f} / {r['monto']:.4f} = {r['razon']:.2f} [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}]", flush=True)
    (OUT / f"d3_capas{SUFIJO}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# Capas A → B → C (suma ponderada, red de PASCO, MaskCluster)", "",
         "| Corte | Puntaje | MAP por entidad | P en 5 | Acierto en 5 | MAP en quintil de monto |", "|---|---|---|---|---|---|"]
    for f in res["cortes"]:
        for n in ("monto", "D1", "A_suma_ponderada", "B_red_PASCO", "C_MaskCluster"):
            if n in f:
                m = f[n]
                L.append(f"| {f['prueba']} | {n} | {m['MAP_ent']:.4f} | {m['P@5_ent']:.4f} | {m['acierto@5_ent']:.3f} | {m['MAP_quintil_monto']:.4f} |")
    L += ["", "| Comparación | Diferencia de MAP | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for n, d in res["comparaciones"].items():
        L.append(f"| {n} | {d['dif_MAP']:+.4f} | [{d['ic90'][0]:+.4f}, {d['ic90'][1]:+.4f}] | {d['por_corte']} |")
    L += ["", "| Candidata, tarea y medida | Candidata | Monto | Razón | IC al 90 % |", "|---|---|---|---|---|"]
    for clave, r in res.get("razon_frente_al_monto", {}).items():
        L.append(f"| {clave} | {r['modelo']:.4f} | {r['monto']:.4f} | {r['razon']:.2f} | [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}] |")
    (OUT / f"d3_capas{SUFIJO}.md").write_text("\n".join(L), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
