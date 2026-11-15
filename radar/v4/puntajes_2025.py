"""El orden de todos los proveedores de cada entidad en 2025, para la pantalla (HALLAZGOS, sección 94).

La lista congelada guarda solo los cinco primeros de cada entidad. La pantalla muestra hasta 100, así que hace falta el
puntaje de todos los casos. Este script rehace la lista limpia de 2025 con el mismo código, los mismos modelos y el mismo
ajuste que la congeló (lista_unica_limpia.py congelar, como importancia_parecido.py), sin tocar nada de lo congelado:

    1. Arma los dos modelos (ajuste 2018 a 2023, corte 31/12/2025) y el universo limpio de 2025.
    2. Se detiene si no repite la lista congelada: mismas filas, motivos, órdenes visibles y percentiles. Además informa si
       el archivo rehecho es idéntico byte por byte (huella 189af867…).
    3. Guarda el puntaje y el puesto de todos los casos: del 1 al 5, los de la lista, en su orden; del 6 en adelante, el
       mismo puntaje de mayor a menor (s_lista; en empate, más monto y luego el RUC).

    python -I -X utf8 puntajes_2025.py
Salida, fuera de git: resultados/lista_unica_limpia/puntajes_2025.parquet y puntajes_2025_manifiesto.json. Trae RUC.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import lista_unica as LU  # noqa: E402
import lista_unica_limpia as LL  # noqa: E402
import m6_proveedor as M6  # noqa: E402
import p1_tablas as P1  # noqa: E402
from casos import DIR  # noqa: E402
from etiquetas import Hechos  # noqa: E402

sys.argv = _a

A = LU.ANIO_LISTA
HUELLA_LISTA = "189af867041d1270bf132ab6ca25041e9a3e782713171758dadf22a761a4aff2"
COLUMNAS = LU.K + ["grupo", "puesto", "en_lista", "es_quinto", "s_lista", "puntaje", "R", "pS", "pP", "grado", "alerta", "percentil_historial",
                   "c_monto", "c_ordenes", "c_local", "M_p", "ordenes_caso", "orden_mayor", "monto_orden_mayor", "orden_con_bandera", "monto_orden_con_bandera",
                   "f30", "f_cerca", "t_autoridad_ordenes", "t_empresa_autoridad_ordenes", "t_pariente_raro_ordenes", "t_empresa_pariente_aut_ordenes",
                   "h_prov_penalidades", "h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones", "p_monto", "p_exp_ocds_monto", "p_cat_monto"]


def puestos(ql, L):
    """Del 1 al 5, la lista en su orden (cuatro por puntaje y el quinto); después, el resto por s_lista.
    En las entidades con cinco casos o menos entran todos, y la lista no les da orden: van por s_lista."""
    lista = L[["ruc_entidad", "ruc", "es_quinto"]].assign(_o=np.arange(len(L)))
    q = ql.merge(lista, on=["ruc_entidad", "ruc"], how="left")
    q["en_lista"] = q["_o"].notna()
    q["es_quinto"] = q["es_quinto"].eq(True)
    q["grande"] = q.groupby("grupo")["ruc"].transform("size") > 5
    q["_k1"] = (~q["en_lista"]).astype(int)
    q["_k2"] = np.where(q["grande"] & q["en_lista"], q["_o"].fillna(0), 0)
    q = q.sort_values(["grupo", "_k1", "_k2", "s_lista", "c_monto", "ruc"], ascending=[True, True, True, False, False, True], kind="mergesort")
    q["puesto"] = q.groupby("grupo").cumcount() + 1
    return q.drop(columns=["_o", "_k1", "_k2"])


def main():
    t0 = time.time()
    sal, man = LL.SAL / "puntajes_2025.parquet", LL.SAL / "puntajes_2025_manifiesto.json"
    congelada = LL.SAL / "lista_2025_unica.parquet"
    assert LU.huella(congelada) == HUELLA_LISTA, "la lista congelada no tiene su huella: no se sigue"
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F_ = M6.Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    cprov, n = LU.comunes()
    Pa = P1.tabla_proveedor(H, F_, LU.AJUSTE_LISTA + [A]).astype({"ruc": str, "anio": int})
    LU.paso("ETAPA 1: tabla de proveedores lista")
    # Para la ficha: los datos del proveedor en 2025 que miran los modelos (concursos, contratos, antigüedad...)
    Pa.loc[Pa["anio"] == A, ["ruc", "anio"] + [c for c in cprov if c in Pa.columns]].to_parquet(LL.SAL / "puntajes_2025_proveedor.parquet", index=False)
    d = LU.armar_anio(A, LU.AJUSTE_LISTA, H, E, Pa, cprov, n, LU.CORTE_LISTA, con_etiquetas=False)
    assert not any(c.startswith(("y", "fuentes", "conocido", "peso")) for c in d.columns), "no se leen etiquetas de 2025"
    q0 = LU.preparar_lista(d, A)
    del d, Pa
    s0 = LL.igual_a_la_guardada(LU.ficha(LU.lista(LL.sin_cambios(q0)).copy(), A, n, con_etiquetas=False), LL.VIEJA / "lista_2025_unica.parquet")
    ql = LL.limpio(q0)
    del q0
    L = LU.lista(ql)
    Fi = LU.ficha(L.copy(), A, n, con_etiquetas=False)
    s1 = LL.igual_a_la_guardada(Fi, congelada)
    G = pd.read_parquet(congelada).astype({"ruc_entidad": str, "ruc": str})
    dif_pct = float((Fi[["ruc_entidad", "ruc", "percentil_historial"]].merge(G[["ruc_entidad", "ruc", "percentil_historial"]], on=["ruc_entidad", "ruc"], suffixes=("", "_g"))
                     .eval("percentil_historial - percentil_historial_g").abs().max()))
    LU.paso(f"ETAPA 2: sin la limpieza {s0}; con ella {s1}; diferencia máxima de percentil {dif_pct:.2e}")
    for s in (s0, s1):
        assert s["llaves_distintas"] == 0 and s["motivos_distintos"] == 0 and s["ordenes_distintas"] == 0, "no repite la lista congelada: se detiene"
    assert dif_pct < 1e-9, "los percentiles no son los de la lista congelada"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "lista_2025_unica.parquet"
        Fi.to_parquet(f, index=False)
        identica = LU.huella(f) == HUELLA_LISTA
        igual_contenido = bool(Fi.reset_index(drop=True).astype({"ruc_entidad": str, "ruc": str}).equals(G))
    LU.paso(f"ETAPA 3: archivo rehecho idéntico byte por byte: {identica}; mismo contenido: {igual_contenido}")

    q = puestos(ql, L)
    top = q[q["puesto"] <= 5].merge(G[["ruc_entidad", "ruc"]].assign(_g=1), on=["ruc_entidad", "ruc"], how="outer", indicator=True)
    assert (top["_merge"] == "both").all(), "los cinco primeros no son la lista congelada"
    orden_g = G.groupby("ruc_entidad", sort=False).cumcount() + 1
    chk = G[["ruc_entidad", "ruc"]].assign(p_g=orden_g.to_numpy()).merge(q[["ruc_entidad", "ruc", "puesto", "grande"]], on=["ruc_entidad", "ruc"])
    chk = chk[chk["grande"]]
    assert (chk["p_g"] == chk["puesto"]).all(), "el orden de los cinco no es el de la lista congelada"
    q = q[[c for c in COLUMNAS if c in q.columns]]
    q.to_parquet(sal, index=False)
    m = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "anio": A, "ajuste": LU.AJUSTE_LISTA, "corte": LU.CORTE_LISTA, "casos": len(q), "entidades": int(q["ruc_entidad"].nunique()),
         "comprobado": {"sin_la_limpieza": s0, "con_la_limpieza": s1, "diferencia_maxima_de_percentil": dif_pct, "lista_rehecha_identica_byte_por_byte": identica,
                        "lista_rehecha_mismo_contenido": igual_contenido, "cinco_primeros_iguales_a_la_congelada": True},
         "sha256_lista_congelada": HUELLA_LISTA, "sha256_puntajes": LU.huella(sal), "minutos": round((time.time() - t0) / 60, 1),
         "huellas": {**LL.huellas(), "puntajes_2025.py": LU.huella(__file__)[:16]}}
    man.write_text(json.dumps(m, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    LU.paso(f"FIN: {len(q):,} casos de {m['entidades']:,} entidades; huella {m['sha256_puntajes'][:16]}")


if __name__ == "__main__":
    main()
