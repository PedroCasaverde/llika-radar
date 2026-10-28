"""Paso 8: ¿qué tipos de alerta anticipan una sanción posterior del Tribunal? Lasso (regresión logística con L1).

Entrena con las alertas de 2018 a 2021 y sanciones que empiezan desde 2022; prueba con las alertas de 2022-2023 y
sanciones que empiezan desde 2024. Se excluye a quien ya estaba sancionado antes de la ventana. Para entrenar se usan
todos los positivos y 100 000 negativos al azar (caso-control: el AUC y el orden no cambian). Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 08_lasso.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA  # noqa: E402

t0 = time.time()
CARTAS = ["empresa_espejo", "mismo_dueno", "funcionario_propio", "doble_percepcion", "familiar"]
o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto"])
al = pd.read_parquet(SALIDA / "radar_alertas.parquet", columns=["ruc", "anio", "tarjeta", "puntos"])
s = pd.read_parquet(SALIDA / "sanciones.parquet", columns=["ruc", "inicio"])
print(f"datos leídos en {time.time() - t0:.0f} s", flush=True)


def ventana(desde, hasta, etiqueta_desde):
    v = o[(o["anio"] >= desde) & (o["anio"] <= hasta)]
    X = v.groupby("ruc").agg(monto=("monto", "sum"), ordenes=("id", "size"), entidades=("ruc_entidad", "nunique"))
    X["log_monto"], X["log_ordenes"] = np.log1p(X["monto"]), np.log1p(X["ordenes"])
    X["persona"] = X.index.str.startswith("10").astype(int)
    aa = al[(al["anio"] >= desde) & (al["anio"] <= hasta)]
    piv = aa.pivot_table(index="ruc", columns="tarjeta", values="puntos", aggfunc="max")
    for t in CARTAS:
        X[t] = piv[t].reindex(X.index).fillna(0) if t in piv else 0.0
    corte = pd.Timestamp(f"{etiqueta_desde}-01-01")
    X = X[~X.index.isin(set(s.loc[s["inicio"] < corte, "ruc"]))]
    y = X.index.isin(set(s.loc[s["inicio"] >= corte, "ruc"])).astype(int)
    return X, y


Xtr, ytr = ventana(2018, 2021, 2022)
Xte, yte = ventana(2022, 2023, 2024)
print(f"ventanas armadas en {time.time() - t0:.0f} s: entrenamiento {len(Xtr):,} proveedores ({ytr.sum():,} con sanción "
      f"posterior); prueba {len(Xte):,} ({yte.sum():,})", flush=True)
rng = np.random.default_rng(7)
neg = np.flatnonzero(ytr == 0)
fil = np.sort(np.concatenate([np.flatnonzero(ytr == 1), rng.choice(neg, size=min(100_000, len(neg)), replace=False)]))
Xs, ys = Xtr.iloc[fil], ytr[fil]

TAM = ["log_monto", "log_ordenes", "entidades", "persona"]
res = {}
for nombre, cols in [("solo alertas", CARTAS), ("alertas + tamaño", CARTAS + TAM)]:
    sc = StandardScaler().fit(Xs[cols])
    g = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", max_iter=500),
                     {"C": [0.001, 0.01, 0.1, 1.0]}, cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc")
    g.fit(sc.transform(Xs[cols]), ys)
    p = g.predict_proba(sc.transform(Xte[cols]))[:, 1]
    res[nombre] = p
    coef = {c: round(float(v), 3) for c, v in zip(cols, g.best_estimator_.coef_[0])}
    print(f"{nombre}: C={g.best_params_['C']}; AUC en prueba {roc_auc_score(yte, p):.3f}; coeficientes {coef} "
          f"({time.time() - t0:.0f} s)", flush=True)
puntos = Xte[CARTAS].sum(axis=1).to_numpy()
print(f"puntaje por puntos: AUC en prueba {roc_auc_score(yte, puntos):.3f}")
sc = StandardScaler().fit(Xs[TAM])
solo_tam = LogisticRegression(max_iter=500).fit(sc.transform(Xs[TAM]), ys).predict_proba(sc.transform(Xte[TAM]))[:, 1]
print(f"solo tamaño del proveedor (sin alertas): AUC en prueba {roc_auc_score(yte, solo_tam):.3f}")
for nombre, sc_ in [("puntos", puntos), ("Lasso solo alertas", res["solo alertas"]), ("Lasso con tamaño", res["alertas + tamaño"]),
                    ("solo tamaño", solo_tam)]:
    top = np.argsort(-sc_, kind="stable")[:1000]
    print(f"  {nombre}: de los 1 000 primeros, {yte[top].sum()} con sanción posterior ({yte[top].mean():.1%}; base {yte.mean():.2%})")
