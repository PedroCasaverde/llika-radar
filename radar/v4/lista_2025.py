"""Lista de 2025 congelada (HALLAZGOS 160, sección 55): la configuración de los exámenes, sin cambios, con lo sabido al 31/12/2025. Corre una sola vez. No imprime ni guarda ninguna etiqueta de 2025.

    python -I -X utf8 lista_2025.py
Salida: resultados/por_caso/lista_2025.parquet (trae RUC: no sale de datos/) y lista_2025_manifiesto.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import corrida_sellada as CS  # noqa: E402
import p1_tablas as P1  # noqa: E402
import u2_ronda_final as U2  # noqa: E402
from casos import DIR, SALIDA, etiquetas_caso  # noqa: E402
from d15_variables import NUEVAS, lista_121  # noqa: E402
from d17_configuracion import alerta  # noqa: E402
from etiquetas import K, Hechos  # noqa: E402
from evaluacion import armar_corte  # noqa: E402
from m6_proveedor import NUEVAS_P, Fuentes, modelo_proveedor  # noqa: E402
sys.argv = _a

A, AJUSTE, CORTE = 2025, [2018, 2019, 2020, 2021, 2022, 2023], "2025-12-31"
SAL, MAN = SALIDA / "lista_2025.parquet", SALIDA / "lista_2025_manifiesto.json"
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def huella(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    assert not MAN.exists(), "la lista se congela una sola vez"
    CS.comprobar_huellas()
    CS.comprobar_portal()
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F = Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    P = P1.tabla_proveedor(H, F, AJUSTE + [A]).astype({"ruc": str, "anio": int})
    x = U2.ampliaciones(U2.contrato_a_ruc(), True)
    x = x[x["f"] <= pd.Timestamp(CORTE)]
    paso("tablas listas")

    def preparar(t):
        t = t.astype({"ruc": str, "ruc_entidad": str, "anio": int}).merge(P[["ruc", "anio"] + NUEVAS_P], on=["ruc", "anio"], how="left")
        t["h_prov_ampl"] = 0.0
        for b, g in t.groupby("anio"):
            s = x[x["f"] <= pd.Timestamp(f"{b}-12-31")].groupby("ruc")["contrato"].nunique()
            t.loc[g.index, "h_prov_ampl"] = g["ruc"].map(s).fillna(0).to_numpy(dtype=float)
        t["h_prov_ampl_parte"] = t["h_prov_ampl"] / t["p_contratos_hasta_t"].astype(float).clip(lower=1)
        return t

    c116 = [c for c in lista_121() if c not in NUEVAS]
    assert len(c116) == 116, f"variables: {len(c116)}"
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    tr, te = armar_corte(H, AJUSTE, A, fecha_prueba=CORTE)
    tr, te = preparar(tr.reset_index(drop=True)), preparar(te.reset_index(drop=True))
    v = etiquetas_caso(E, tr, CORTE, ventana=None)
    yS, yP = v["y_S"].to_numpy(), v["y_P"].to_numpy()
    paso(f"ajuste {len(tr):,} casos (S {int(yS.sum()):,}; P {int(yP.sum()):,}); lista {len(te):,} casos")
    d = te[K + ["grupo", "c_monto", "c_ordenes"]].copy()
    d["alerta"] = alerta(te)
    d["s_monto"] = te["c_monto"].to_numpy(dtype=float)
    d["s_tamano_proveedor"] = np.log1p(te[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1)).to_numpy()
    d["s_alerta_monto"] = 100 * d["alerta"].to_numpy() + np.log1p(d["c_monto"].astype(float).to_numpy())
    d["s_S_prov"] = modelo_proveedor(tr, yS, te, cprov, (CS.ARBOLES_P,))[CS.ARBOLES_P][0]
    paso("modelo S listo")
    d["s_P_ampl"] = modelo_proveedor(tr, yP, te, cprov + CS.NV, (CS.ARBOLES_P,))[CS.ARBOLES_P][0]
    paso("modelo P con ampliaciones listo")
    assert not any(c.startswith(("y", "fuentes", "conocido", "peso")) for c in d.columns), "la lista no guarda etiquetas"
    d.to_parquet(SAL, index=False)
    raiz = CS.BASE.parents[3]
    scripts = ["radar/v4/lista_2025.py", "radar/v4/corrida_sellada.py", "radar/v4/casos.py", "radar/v4/p1_tablas.py", "radar/v4/u2_ronda_final.py", "radar/v4/m6_proveedor.py", "radar/v4/etiquetas.py", "radar/v4/evaluacion.py", "radar/v4/d15_variables.py", "radar/v4/d17_configuracion.py", "radar/v4/d1_linea_base.py"]
    m = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "anio": A, "ajuste": AJUSTE, "corte": CORTE, "casos": len(d), "alertas": int(d["alerta"].sum()), "entidades": int(d["ruc_entidad"].nunique()), "proveedores": int(d["ruc"].nunique()),
         "soluciones": {"caso_1": "s_alerta_monto", "caso_3": "s_S_prov", "caso_4": "s_P_ampl", "caso_5": "órdenes por monto (regla, sin puntaje)"}, "varas": {"tamano": "s_tamano_proveedor", "monto": "s_monto"},
         "sha256_lista": huella(SAL), "sha256_scripts": {s: huella(raiz / s) for s in scripts}, "sha256_eventos": huella(DIR / "eventos_examen.parquet")}
    MAN.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    paso(f"CONGELADA: {m['casos']:,} casos; {m['alertas']:,} alertas; huella {m['sha256_lista'][:16]}")
    paso("LISTO")


if __name__ == "__main__":
    main()
