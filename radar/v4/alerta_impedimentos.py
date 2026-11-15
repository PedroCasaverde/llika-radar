"""Alerta de impedimentos, aparte del ranking (HALLAZGOS 140; Pedro: «está bien»). Fijada antes de mirar.

Se enciende si el caso tiene, a la fecha, cualquiera de estas condiciones:
    autoridad: la autoridad electa misma le vende a una entidad de su territorio (t_autoridad_ordenes > 0);
    empresa: le vende la empresa de una autoridad, con los socios del corte del año (t_empresa_autoridad_ordenes > 0);
    pariente_raro: pariente de una autoridad con un par de apellidos raro, 100 personas o menos en el padrón
        (t_pariente_raro_ordenes > 0);
    padre_hijo: padre o madre e hijo con una autoridad, con edad y sexo, probabilidad de 50 % o más (t_padre_hijo_prob);
    familiar_dj: posible familiar de un funcionario de decisión en las declaraciones visibles al cierre del año, con
        probabilidad de 50 % o más (f_familiar_prob).
Se mide en la prueba de detección de 2020, 2021 y 2023 (etiqueta amplia, lo conocido al 31/12/2025): alertas por año,
precisión para impedimento (C núcleo) y para irregularidad (A, B o C), cobertura de los impedimentos, y lo mismo para
la misma cantidad de casos tomados por monto, de mayor a menor, en todo el país.

    python -I -X utf8 alerta_impedimentos.py
Desde HALLAZGOS 141, padre o madre e hijo se reporta como componente pero no enciende la alerta (va a la ficha).

Salida: resultados/alerta_impedimentos.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etiquetas import Hechos  # noqa: E402
from evaluacion import cargar_x, familia_a_la_decision, universo  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"


def impedimento_real(H, x):
    """HALLAZGOS 148: el caso tiene una sanción del Tribunal con la orden hallada y con literal del art. 11 (impedimento
    de verdad); la C núcleo sin literal es una sanción por otra causa."""
    from etiquetas import BASE
    lit = pd.read_parquet(BASE / "resoluciones.parquet", columns=["resolucion", "impedimento_literal"]).drop_duplicates("resolucion")
    lit = lit.set_index("resolucion")["impedimento_literal"]
    c = H.ev[H.ev["fuente"] == "C"][["documento", "ruc_entidad", "ruc", "anio_hechos"]].copy()
    c["real"] = c["documento"].map(lit).notna()
    c = c[c["real"]].astype({"ruc_entidad": str, "ruc": str})
    llaves = set(zip(c["ruc_entidad"], c["ruc"], c["anio_hechos"].astype(int)))
    return np.array([(e, r, int(a)) in llaves for e, r, a in zip(x["ruc_entidad"].astype(str), x["ruc"].astype(str), x["anio"])])


def main():
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
        imp_c, est = fu.str.contains("C").to_numpy(), fu.str.contains("A|B|C").to_numpy()
        imp = imp_c & impedimento_real(H, x)          # 148: solo impedimentos de verdad (con literal del art. 11)
        cero = pd.Series(0, index=x.index)
        cond = {"autoridad": x["t_autoridad_ordenes"] > 0, "empresa": x["t_empresa_autoridad_ordenes"] > 0,
                "pariente_raro": x["t_pariente_raro_ordenes"] > 0,
                "padre_hijo": x.get("t_padre_hijo_prob", cero) >= 0.5,
                "familiar_dj": x["f_familiar_prob"] >= 0.5,
                "empresa_pariente_aut": x.get("t_empresa_pariente_aut_ordenes", cero) > 0,        # 148
                "empresa_funcionario": x.get("f_empresa_funcionario_ordenes", cero) > 0,          # 148
                "funcionario_propio": x.get("f_funcionario_propio_ordenes", cero) > 0}            # 148
        alerta = np.zeros(len(x), dtype=bool)
        anterior = np.zeros(len(x), dtype=bool)          # la alerta de 141, para medir lo que suma la 148
        for n, c in cond.items():
            c = c.to_numpy()
            if n != "padre_hijo":                  # desde 141, padre o madre e hijo va a la ficha, no a la alerta
                alerta |= c
            if n in ("autoridad", "empresa", "pariente_raro", "familiar_dj"):
                anterior |= c
            filas.append({"año": a, "condición": n, "alertas": int(c.sum()), "impedimentos": int(imp[c].sum()),
                          "precisión_impedimento": imp[c].mean() if c.any() else np.nan,
                          "irregularidades": int(est[c].sum()), "precisión_irregularidad": est[c].mean() if c.any() else np.nan})
        orden = np.argsort(-x["c_monto"].to_numpy(dtype=float))
        top = np.zeros(len(x), dtype=bool)
        top[orden[:int(alerta.sum())]] = True
        for n, m in (("ALERTA de 141", anterior), ("ALERTA con 148", alerta), ("mismo número por monto", top),
                     ("base (todos los casos)", np.ones(len(x), dtype=bool))):
            filas.append({"año": a, "condición": n, "alertas": int(m.sum()), "impedimentos": int(imp[m].sum()),
                          "precisión_impedimento": imp[m].mean(), "irregularidades": int(est[m].sum()),
                          "precisión_irregularidad": est[m].mean(),
                          "cobertura_impedimentos": imp[m].sum() / max(imp.sum(), 1),
                          "otras_sanciones_C": int((imp_c & ~imp)[m].sum())})
        print(pd.DataFrame(filas[-12:]).to_string(index=False), flush=True)
    t = pd.DataFrame(filas)
    for c in ("precisión_impedimento", "precisión_irregularidad", "cobertura_impedimentos"):
        t[c] = (100 * t[c]).round(2)
    (OUT / "alerta_impedimentos.md").write_text("# Alerta de impedimentos (140), detección, etiqueta a la fecha (porcentajes)\n\n"
                                                + t.to_markdown(index=False), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
