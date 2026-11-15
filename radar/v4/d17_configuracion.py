"""Paso 2 del cierre (HALLAZGOS 157): la comparación final con la configuración aprobada. Puntajes guardados.

Vista 1, señal adversa (etiqueta amplia L0): D1 binario frente al tamaño del proveedor (comparador), el monto, la regla
    rústica de las banderas del tope y el azar; una hora por caso.
Vista 2, hallazgos del control: por hora (una hora por orden), de d16_hallazgos_por_hora.
Vista 3, impedimentos reales (C con literal del art. 11): D1 ordinal frente al monto, el tamaño, la regla rústica y el
    azar; una hora por caso.
Grave (A, B o C): solo se reporta.
Alerta de impedimentos (148): impedimentos reales que halla frente al mismo número de casos por monto, en todo el país.
Las variables de D1 son las que dejó d15_variables (116 o 121). Precisión en 5 por entidad-año con algún positivo, los
tres cortes juntos; razones con IC al 90 % remuestreando entidades. Sin auditor promedio (155).

    python -I -X utf8 d17_configuracion.py
Salida: resultados/d17_configuracion.json y .md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combinaciones import rrf  # noqa: E402
from comparacion_final import RIESGO, riesgo  # noqa: E402
from etiquetas import K  # noqa: E402
from evaluacion import cargar_x, familia_a_la_decision, por_grupo_k  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
PRUEBA = (2020, 2021, 2023)
RNG = np.random.default_rng(0)
ALERTA = ["t_autoridad_ordenes", "t_empresa_autoridad_ordenes", "t_pariente_raro_ordenes", "f_familiar_prob",
          "t_empresa_pariente_aut_ordenes", "f_empresa_funcionario_ordenes", "f_funcionario_propio_ordenes"]


def alerta(x):
    f = lambda c: x[c].astype(float).fillna(0) if c in x else pd.Series(0.0, index=x.index)  # noqa: E731
    a = (f("t_autoridad_ordenes") > 0) | (f("t_empresa_autoridad_ordenes") > 0) | (f("t_pariente_raro_ordenes") > 0) \
        | (f("f_familiar_prob") >= 0.5) | (f("t_empresa_pariente_aut_ordenes") > 0) | (f("f_empresa_funcionario_ordenes") > 0) \
        | (f("f_funcionario_propio_ordenes") > 0)
    return a.to_numpy()


def por_grupo(te, s, y):
    return por_grupo_k(te[y], s, te["grupo"].to_numpy(), (5,), desempates=10)["P@5"]


def azar_grupo(te, y):
    g = te.groupby("grupo")[y].agg(["sum", "size"])
    g = g[g["sum"] > 0]
    return (g["sum"] / np.minimum(5, g["size"]) * np.minimum(5, g["size"]) / g["size"]).rename("P@5")


def razon(filas, n=2000):
    d = pd.concat(filas)
    pe = d.groupby("e")[["m1", "m0"]].sum()
    boot = [pe.iloc[RNG.integers(0, len(pe), len(pe))].pipe(lambda m: m["m1"].sum() / max(m["m0"].sum(), 1e-12)) for _ in range(n)]
    return {"razon": float(d["m1"].sum() / d["m0"].sum()), "ic90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))]}


def comparar(cortes, y, metodos, base):
    """P@5 conjunta de cada método y su razón frente a `base` (un método o 'azar')."""
    out = {"precision_en_5": {}, "frente_a": {}}
    grupos = {m: [] for m in list(metodos) + ["azar"]}
    for te in cortes:
        for m, c in metodos.items():
            grupos[m].append(por_grupo(te, te[c].to_numpy(dtype=float), y))
        grupos["azar"].append(azar_grupo(te, y))
    for m, partes in grupos.items():
        out["precision_en_5"][m] = float(pd.concat(partes).mean())
    for b in base:
        out["frente_a"][b] = {}
        for m in metodos:
            if m == b:
                continue
            filas = []
            for a1, a0 in zip(grupos[m], grupos[b]):
                a0 = a0.reindex(a1.index)
                filas.append(pd.DataFrame({"m1": a1.to_numpy(), "m0": a0.to_numpy(), "e": pd.Series(a1.index).str.split("|").str[0].to_numpy()}))
            out["frente_a"][b][m] = razon(filas)
    out["horas_por_acierto"] = {m: 1 / max(v, 1e-12) for m, v in out["precision_en_5"].items()}
    return out


def main():
    queda = json.loads((OUT / "d15_variables.json").read_text(encoding="utf-8"))["queda"]
    print(f"variables de D1: {queda}", flush=True)
    cortes, alertas = [], []
    for a in PRUEBA:
        te = pd.read_parquet(OUT / f"puntajes_d5_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        nuevos = pd.read_parquet(OUT / f"puntajes_d15_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        te = te.drop(columns=["s_D1_binario", "s_D1_ord"]).merge(
            nuevos[K + ["y_imp_real", f"s_D1_binario_{queda}", f"s_D1_ord_{queda}"]].rename(
                columns={f"s_D1_binario_{queda}": "s_D1_binario", f"s_D1_ord_{queda}": "s_D1_ord"}), on=K, how="left")
        x = familia_a_la_decision(cargar_x([a]), a)
        x = x[K + [c for c in RIESGO + ALERTA if c in x.columns]].astype({"ruc_entidad": str, "ruc": str, "anio": int})
        te = te.merge(x, on=K, how="left")
        te[RIESGO] = te[RIESGO].fillna(0)
        te["s_rustica"] = riesgo(te)
        te["s_R2"] = rrf(te, ["monto", "D1_ord"])
        for c in ("y", "y_estricta", "y_rev", "y_imp_real"):
            te[c] = te[c].astype(bool)
        al = alerta(te)
        orden = np.argsort(-te["c_monto"].to_numpy(dtype=float))
        top = np.zeros(len(te), dtype=bool)
        top[orden[: int(al.sum())]] = True
        alertas.append({"año": a, "alertas": int(al.sum()), "impedimentos_alerta": int(te.loc[al, "y_imp_real"].sum()),
                        "impedimentos_monto_mismo_numero": int(te.loc[top, "y_imp_real"].sum()),
                        "impedimentos_en_la_prueba": int(te["y_imp_real"].sum())})
        print(f"   {a}: {len(te):,} casos; alerta {alertas[-1]}", flush=True)
        cortes.append(te)
    m = {"D1_binario": "s_D1_binario", "D1_ordinal": "s_D1_ord", "R2": "s_R2", "monto": "s_monto",
         "tamano": "s_tamano_proveedor", "rustica": "s_rustica"}
    res = {"variables_D1": queda,
           "vista_1_senal_adversa": comparar(cortes, "y", m, ["tamano", "monto", "rustica", "azar"]),
           "vista_3_impedimentos_reales": comparar(cortes, "y_imp_real", m, ["monto", "tamano", "rustica", "azar"]),
           "grave_estricta": comparar(cortes, "y_estricta", m, ["monto", "tamano", "azar"]),
           "hallazgos_por_caso": comparar(cortes, "y_rev", m, ["monto", "azar"]),
           "alerta": {"por_anio": alertas, "total": {k: int(sum(d[k] for d in alertas)) for k in alertas[0] if k != "año"}}}
    v2 = OUT / "d16_hallazgos_por_hora.json"
    res["vista_2_hallazgos_por_hora"] = json.loads(v2.read_text(encoding="utf-8")) if v2.exists() else None
    (OUT / "d17_configuracion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    L = [f"# Comparación final con la configuración del pre-registro (HALLAZGOS 157), variables de D1: {queda}", ""]
    for v, comp in (("Vista 1: señal adversa (amplia)", "vista_1_senal_adversa"), ("Vista 3: impedimentos reales", "vista_3_impedimentos_reales"),
                    ("Grave (A, B o C), solo se reporta", "grave_estricta"), ("Hallazgos del control por caso, solo se reporta", "hallazgos_por_caso")):
        f = res[comp]
        L += [f"## {v}", "", "| Método | Precisión en 5 | Horas por acierto (una por caso) |", "|---|---|---|"]
        L += [f"| {k} | {p:.4f} | {f['horas_por_acierto'][k]:.1f} |" for k, p in f["precision_en_5"].items()]
        for b, d in f["frente_a"].items():
            L += ["", f"Frente a {b}: " + "; ".join(f"{k} {r['razon']:.2f} [{r['ic90'][0]:.2f} a {r['ic90'][1]:.2f}]" for k, r in d.items())]
        L.append("")
    L += ["## Alerta de impedimentos", "", f"{res['alerta']}", ""]
    if res["vista_2_hallazgos_por_hora"]:
        L += ["## Vista 2 por hora", "", f"Hipótesis principal: {res['vista_2_hallazgos_por_hora']['hipotesis_principal']}", ""]
    (OUT / "d17_configuracion.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
