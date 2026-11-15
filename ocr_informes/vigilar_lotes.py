"""Vigila los lotes de Pedro en Kaggle mientras la PC está prendida (PENDIENTES.md, sección 2).

Cada 10 minutos:
- baja el resultado de cada lote que terminó a resultados/lote_NN (solo el zip, los resúmenes y los registros);
- lanza los lotes pendientes cuando se libera una sesión (Kaggle deja 2 de GPU y 5 de CPU a la vez por cuenta).
  07a, 07b y 07c van sin GPU: son informes de rutina, casi todo texto digital (el 08a, parecido, tardó 22 minutos).
  El 16 son 23 informes: también va sin GPU.
Un lote que termina con error se baja (para ver su registro) pero no se relanza solo.
Termina cuando todo terminó y está bajado. Registro en log_vigilante.txt.

    export KAGGLE_API_TOKEN=...   (el de Pedro, del .env)
    <python con kaggle> -X utf8 vigilar_lotes.py
    <python con kaggle> -X utf8 vigilar_lotes.py --corriendo 06a:cpu,07a:cpu --lanzar 07c:gpu,16:gpu
"""
import argparse
import contextlib
import csv
import io
import re
import sys
import time
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI / "para_el_equipo" / "Pedro"))  # ahí están los notebooks de Pedro
import lanzar_en_kaggle as lk  # noqa: E402

USUARIO = "pedrocasaverdeayma"
CORRIENDO = {"01": "gpu", "02": "gpu", "06a": "cpu", "06b": "cpu", "06c": "cpu", "15a": "cpu", "15b": "cpu"}
POR_LANZAR = [("07a", "cpu"), ("07b", "cpu"), ("07c", "cpu"), ("16", "cpu")]
TOPE = {"gpu": 2, "cpu": 5}
FIN = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
LOG = AQUI / "log_vigilante.txt"


def log(texto):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%d/%m %H:%M')} {texto}\n")


def estado(n):
    m = re.search(r"KernelWorkerStatus\.(\w+)", lk.kaggle("kernels", "status", f"{USUARIO}/llika-ocr-lote-{n}"))
    return m.group(1) if m else "?"


def bajar(n, e):
    destino = AQUI / "resultados" / f"lote_{n}"
    destino.mkdir(parents=True, exist_ok=True)
    lk.kaggle("kernels", "output", f"{USUARIO}/llika-ocr-lote-{n}", "-p", str(destino), "--file-pattern",
              r"\.(zip|csv|log)$")
    resumen = destino / f"textos_lote_{n}" / "resumen.csv"
    if resumen.exists():
        with resumen.open(encoding="utf-8") as f:
            r = list(csv.DictReader(f))
        ok = sum(x["estado"] == "ok" for x in r)
        paginas = sum(int(float(x["paginas"] or 0)) for x in r)
        ocr = sum(int(float(x["paginas_ocr"] or 0)) for x in r)
        # contra el lote completo, no contra el resumen: si un trabajador muere al empezar, el resumen solo trae la
        # mitad que sí se leyó (pasó en los lotes 01, 02 y 07c el 31/10)
        lote = AQUI / "lotes" / f"lote_{n}.csv"
        if lote.exists():
            with lote.open(encoding="utf-8") as f:
                total = sum(1 for _ in csv.DictReader(f))
        else:
            total = len(r)
        falta = f"; FALTAN {total - len(r)} sin leer: relanzarlos" if len(r) < total else ""
        log(f"lote {n}: {e}; bajado: {ok} de {total} informes ok, {paginas:,} páginas ({ocr:,} con OCR){falta}"
            .replace(",", " "))
    else:
        log(f"lote {n}: {e}; sin resumen, revisar el registro en {destino.name}")


def lanzar(n, tipo):
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        lk.lanzar([n], cpu=tipo == "cpu")
    return salida.getvalue()


def main(corriendo, por_lanzar):
    lanzados, pendientes, bajados = dict(corriendo), list(por_lanzar), set()
    log(f"Inicio. Corriendo: {' '.join(lanzados)}. Por lanzar: {' '.join(n for n, _ in pendientes)}")
    limite = time.time() + 30 * 3600
    while time.time() < limite:
        est = {n: estado(n) for n in lanzados if n not in bajados}
        for n, e in est.items():
            if e in FIN:
                bajar(n, e)
                bajados.add(n)
        activos = {"gpu": 0, "cpu": 0}
        for n, e in est.items():
            if e not in FIN:  # un estado desconocido («?») cuenta como activo, por si acaso
                activos[lanzados[n]] += 1
        for n, tipo in list(pendientes):
            if activos[tipo] >= TOPE[tipo]:
                continue
            salida = lanzar(n, tipo)
            if "successfully pushed" in salida:
                lanzados[n] = tipo
                pendientes.remove((n, tipo))
                activos[tipo] += 1
                log(f"lote {n}: lanzado ({tipo.upper()})")
            else:
                log(f"lote {n}: Kaggle no lo aceptó todavía ({salida.strip()[-150:]})")
                break
        if not pendientes and all(n in bajados for n in lanzados):
            log("Todo terminado y bajado.")
            return
        time.sleep(600)
    log("Se cumplieron 30 horas: me detengo.")


def pares(texto):
    """«07c:gpu,16:cpu» -> [("07c", "gpu"), ("16", "cpu")]"""
    return [tuple(x.strip().split(":")) for x in texto.split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corriendo", default=",".join(f"{k}:{v}" for k, v in CORRIENDO.items()),
                    help="lotes que ya corren en Kaggle, con su tipo de sesión")
    ap.add_argument("--lanzar", default=",".join(f"{k}:{v}" for k, v in POR_LANZAR),
                    help="lotes por lanzar, en orden, cuando haya una sesión libre de su tipo")
    a = ap.parse_args()
    main(dict(pares(a.corriendo)), pares(a.lanzar))
