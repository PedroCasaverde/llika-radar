"""Lote 5 (HALLAZGOS 160, C15): los puntajes de la nueva línea base, con la data corregida y la configuración de HOY.
Nada se elige aquí: D1 binario y ordinal con las 116 variables y la parada de siempre, para tener la referencia sobre la que
se prueban los modelos y configuraciones.

Guarda resultados/puntajes_d18_<año>.parquet con los puntajes y las etiquetas de la prueba:
- y: etiqueta amplia con la ventana (a); nivel: 1 solo P, 2 S, 3 A, B o C;
- y_rev, y_imp, y_imp_real e y_estricta: con todo lo conocido, sin la ventana (fuentes_estrictas).

    python -I -X utf8 d18_linea_base.py [año]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alerta_impedimentos import impedimento_real  # noqa: E402
from combinaciones import rrf  # noqa: E402
from comparacion_final import riesgo  # noqa: E402
from d1_linea_base import CORTES  # noqa: E402
from d15_variables import NUEVAS, entrenar, lista_121, nivel  # noqa: E402
from d17_configuracion import ALERTA, alerta  # noqa: E402
from etiquetas import K, Hechos  # noqa: E402
from evaluacion import armar_corte  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def main():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    c116 = [c for c in lista_121() if c not in NUEVAS]
    assert len(c116) == 116, f"variables: {len(c116)}"
    solo = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for tr_a, te_a in CORTES:
        if solo and te_a != solo:
            continue
        paso(f"corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        tr["nivel"], te["nivel"] = nivel(tr["fuentes"]), nivel(te["fuentes"])
        paso(f"   entrenamiento {len(tr):,} casos, {int(tr['y'].sum()):,} positivos, por nivel {pd.Series(tr['nivel']).value_counts().sort_index().to_dict()}; "
             f"prueba {len(te):,} casos, {int(te['y'].sum()):,} positivos")
        s_bin, s_ord, its = entrenar(tr, te, c116, tr_a[-1])
        paso(f"   árboles: binario y umbrales {its}")
        fu = te["fuentes_estrictas"].fillna("").astype(str)
        d = te[K + ["grupo", "nivel", "y", "y_estricta", "c_monto", "u_parte_otras", "u_solo_otras", "c_pasa_tope_30d", "c_meses_superan_tope", "c_cerca_tope"]
               + [c for c in ALERTA if c in te.columns]].copy()
        d["y_rev"], d["y_imp"] = fu.str.contains("A|B").to_numpy(), fu.str.contains("C").to_numpy()
        d["y_imp_real"] = d["y_imp"].to_numpy() & impedimento_real(H, te)
        d["alerta"] = alerta(d)
        d["s_monto"] = te["c_monto"].to_numpy(dtype=float)
        d["s_tamano_proveedor"] = np.log1p(te[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1)).to_numpy()
        d["s_D1_binario"], d["s_D1_ord"] = s_bin, s_ord
        d["s_rustica"] = riesgo(d)
        d["s_alerta_monto"] = 100 * d["alerta"].to_numpy() + np.log1p(d["c_monto"].astype(float).to_numpy())
        d["s_R2"] = rrf(d, ["monto", "D1_ord"])
        d.to_parquet(OUT / f"puntajes_d18_{te_a}.parquet", index=False)
        paso(f"   guardado: {len(d):,} casos; impedimentos reales {int(d['y_imp_real'].sum())}; hallazgos {int(d['y_rev'].sum()):,}; alertas {int(d['alerta'].sum()):,}")
    paso("listo")


if __name__ == "__main__":
    main()
