"""Tabla de evaluación de 2022 y 2024 para el notebook del Anexo D (etapa 4c del cierre, 15/11/2026).

No mide nada nuevo ni toca la segunda medición: rehace, con las mismas funciones de `lista_unica_limpia.segunda`, el
universo limpio de cada año y guarda por caso lo que el notebook necesita para repetir la evaluación sin reentrenar.
Antes de guardar comprueba dos cosas y se detiene si alguna falla:
    1. la lista rehecha es idéntica a la guardada (mismas filas, motivos y órdenes visibles);
    2. la tabla de las cinco listas (LLIKA, monto, tamaño, alerta y monto, azar) repite `segunda_medicion.json`: mismos
       conteos por verdad y por año, y las mismas razones con su intervalo al 90 %.

    python -I -X utf8 tabla_evaluacion.py

Salida, fuera de git (trae RUC): resultados/lista_unica_limpia/evaluacion_2022.parquet, evaluacion_2024.parquet y
evaluacion_manifiesto.json con sus huellas. Una fila por caso del universo limpio. Columnas: llave del caso, grupo,
etiquetas, en qué lista está cada caso, motivo y quinto puesto de la lista, y los segmentos (tipo de entidad,
proveedores de la entidad en el año, persona natural).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import lista_unica as LU  # noqa: E402
import lista_unica_limpia as LUL  # noqa: E402  (al importarse, cambia LU.compras por la versión limpia)
import p1_tablas as P1  # noqa: E402
from casos import DIR, SALIDA, TIPOS  # noqa: E402
from etiquetas import K, Hechos  # noqa: E402
from evaluacion import X_PATH  # noqa: E402
from m6_proveedor import Fuentes  # noqa: E402

sys.argv = _a
SAL = LUL.SAL
T0 = time.time()
LISTAS = {"lista única": "en_lista", "monto": "en_monto", "tamaño": "en_tamano", "alerta y monto": "en_alerta_monto", "azar": "en_azar"}


def pico_gb():
    """Pico de memoria del proceso (Windows); en otro sistema, el máximo residente de resource."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**30
    except (AttributeError, OSError):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def paso(t):
    pico = pico_gb()
    print(f"[{(time.time() - T0) / 60:5.1f} min; pico {pico:.1f} GB] {t}", flush=True)


def segmentos(a):
    x = pd.read_parquet(X_PATH, columns=K + ["e_tipo", "e_proveedores", "u_persona_natural"], filters=[("anio", "==", a)])
    return x.astype(TIPOS)


def tabla_caso(q, partes, F, a):
    t = q[K + ["grupo"] + LU.ETIQUETAS + ["alerta", "grado"]].copy()
    for nombre, col in LISTAS.items():
        t[col] = t.index.isin(partes[nombre].index)
    m = F[K + ["motivo"]].astype(TIPOS)
    t = t.astype(TIPOS).merge(m, on=K, how="left")
    quinto = partes["lista única"].loc[partes["lista única"]["es_quinto"], K].astype(TIPOS).assign(es_quinto=True)
    t = t.merge(quinto, on=K, how="left")
    t["es_quinto"] = t["es_quinto"].eq(True)
    t = t.merge(segmentos(a), on=K, how="left")
    assert int(t["en_lista"].sum()) == len(F) and int(t["motivo"].notna().sum()) == len(F), "la tabla no trae todas las filas de la lista"
    return t


def comparar(r, guardada):
    """Mismos conteos por verdad y por año, y mismas razones e intervalos que la segunda medición guardada."""
    dif = []
    for v, f in guardada["por_verdad"].items():
        for n, x in f.items():
            if r["por_verdad"][v][n]["proveedores"] != x["proveedores"] or r["por_verdad"][v][n]["filas"] != x["filas"]:
                dif.append(f"{v}/{n}: {r['por_verdad'][v][n]} frente a {x}")
    for a, f in guardada["por_anio"].items():
        for n, x in f.items():
            if r["por_anio"][int(a)][n]["proveedores"] != x["proveedores"]:
                dif.append(f"{a}/{n}: {r['por_anio'][int(a)][n]['proveedores']} frente a {x['proveedores']}")
    for v, f in guardada["razones"].items():
        for c, x in f.items():
            y = r["razones"][v][c]
            if x["razon"] is None:
                continue
            if abs(y["razon"] - x["razon"]) > 1e-9 or any(abs(p - q) > 1e-9 for p, q in zip(y["ic90"], x["ic90"])):
                dif.append(f"razón {v}/{c}: {y} frente a {x}")
    return dif


def main():
    guardada = json.loads((SAL / "segunda_medicion.json").read_text(encoding="utf-8"))
    H = Hechos(conocimiento="publicacion", abrir_sellado=True)
    H.etiqueta = H.etiqueta_amplia
    F_ = Fuentes(abrir=True)
    E = pd.read_parquet(DIR / "eventos_examen.parquet")
    cprov, n = LU.comunes()
    partes, tablas, comprobado = {}, {}, {}
    for a, anios in LU.SEGUNDA.items():
        Pa = P1.tabla_proveedor(H, F_, anios + [a]).astype({"ruc": str, "anio": int})
        paso(f"{a}: tabla de proveedores lista")
        d = LU.armar_anio(a, anios, H, E, Pa, cprov, n, LU.FECHA_SEGUNDA)
        q = LUL.limpio(LU.preparar_lista(d, a))
        del d, Pa
        L = LU.lista(q)
        F = LU.ficha(L.copy(), a, n)
        igual = LUL.igual_a_la_guardada(F, SAL / f"lista_{a}.parquet")
        paso(f"{a}: frente a la lista guardada {igual}")
        assert igual["filas"] == igual["filas_guardadas"] and igual["llaves_distintas"] == 0 and igual["motivos_distintos"] == 0 and igual["ordenes_distintas"] == 0, \
            "la lista rehecha no es la guardada: se detiene sin guardar"
        partes[a] = LU.partes_de(q, L)
        tablas[a] = tabla_caso(q, partes[a], F, a)
        comprobado[a] = igual
        paso(f"{a}: {len(tablas[a]):,} casos; {int(tablas[a]['en_lista'].sum()):,} filas en la lista")
        del q, L, F
    r = LU.tabla(partes)
    dif = comparar(r, guardada)
    assert not dif, "no repite la segunda medición: " + "; ".join(dif[:5])
    total = r["por_verdad"]["señal adversa"]["lista única"]["proveedores"]
    paso(f"repite la segunda medición: {total} frente a {r['por_verdad']['señal adversa']['monto']['proveedores']} del monto; razón {r['razones']['señal adversa']['monto']}")
    man = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "comprobado": {str(a): c for a, c in comprobado.items()}, "repite_segunda_medicion": True, "archivos": {}}
    for a, t in tablas.items():
        f = SAL / f"evaluacion_{a}.parquet"
        t.to_parquet(f, index=False)
        man["archivos"][f.name] = {"filas": len(t), "sha256": LU.huella(f)[:16]}
    man["huellas"] = {**LUL.huellas(), "tabla_evaluacion.py": LU.huella(__file__)[:16]}
    (SAL / "evaluacion_manifiesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    paso(f"LISTO: {man['archivos']}")


if __name__ == "__main__":
    main()
