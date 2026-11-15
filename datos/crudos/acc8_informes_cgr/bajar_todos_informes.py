"""Baja los PDF de los 14 895 informes de la lista ampliada (HALLAZGOS 65) y anota si cada uno trae texto digital o
es escaneado. Se puede cortar y volver a correr: salta los que ya bajó.

Uso:  python -X utf8 bajar_todos_informes.py
      python -X utf8 bajar_todos_informes.py --lotes 20 21 22   # solo los informes de esos lotes de ocr_informes/lotes
      (1/11: los que Kaggle no pudo bajar porque la Contraloría no le contestaba a esas sesiones; esta PC sí llega)
Salida: pdf_todos/<año>/<informe>.pdf  e  inventario_pdf.csv
"""
import argparse
import csv
import re
import time
from pathlib import Path

import fitz
import pandas as pd
import requests

AQUI = Path(__file__).parent
DESTINO = AQUI / "pdf_todos"
INVENTARIO = AQUI / "inventario_pdf.csv"
PAUSA = 1.5                               # segundos entre descargas, para no cargar el servidor de la Contraloría


def tipo_de_pdf(ruta):
    """Devuelve páginas y cuántas traen texto digital."""
    d = fitz.open(ruta)
    con_texto = sum(1 for p in d if len(p.get_text().strip()) > 50)
    return d.page_count, con_texto


def main(lotes=None):
    lista = pd.read_csv(AQUI / "listas" / "lista_unica_informes_ampliada.csv", dtype=str)
    if lotes:   # solo esos lotes: no se baja nada que Kaggle ya esté leyendo
        sel = pd.concat([pd.read_csv(AQUI.parents[2] / "ocr_informes" / "lotes" / f"lote_{n}.csv", dtype=str)
                         for n in lotes])
        lista = lista[lista.CresCodigoFormato.isin(set(sel.CresCodigoFormato))]
    # Primero 2022-2024, donde la prueba encontró PDF con texto digital (los que se leen sin OCR); 2025 y 2026
    # salieron todos escaneados. Después, el resto de años, de más reciente a más antiguo.
    prioridad = {"2022": 0, "2023": 1, "2024": 2}
    lista = lista[lista.RutaCloudInforme.notna()].copy()
    lista["_orden"] = lista.Periodo.map(prioridad).fillna(3)
    lista = lista.sort_values(["_orden", "Periodo"], ascending=[True, False])
    hechos = set()
    if INVENTARIO.exists():
        hechos = set(pd.read_csv(INVENTARIO, dtype=str).CresCodigoFormato)
    nuevo = not INVENTARIO.exists()
    sesion = requests.Session()
    with open(INVENTARIO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["CresCodigoFormato", "NumeroInforme", "Periodo", "archivo", "bytes", "paginas",
                        "paginas_con_texto", "estado"])
        pendientes = lista[~lista.CresCodigoFormato.isin(hechos)]
        print(f"{len(hechos)} ya bajados; faltan {len(pendientes)}", flush=True)
        for i, fila in enumerate(pendientes.itertuples(), 1):
            carpeta = DESTINO / str(fila.Periodo)
            carpeta.mkdir(parents=True, exist_ok=True)
            ruta = carpeta / (re.sub(r"[^\w-]", "_", fila.NumeroInforme or fila.CresCodigoFormato) + ".pdf")
            estado, paginas, con_texto, tam = "ok", "", "", ""
            for intento in range(3):
                try:
                    r = sesion.get(fila.RutaCloudInforme.replace("http://", "https://"), timeout=300)
                    r.raise_for_status()
                    if not r.content.startswith(b"%PDF"):
                        estado = "no_es_pdf"
                        break
                    ruta.write_bytes(r.content)
                    tam = len(r.content)
                    paginas, con_texto = tipo_de_pdf(ruta)
                    estado = "ok"
                    break
                except Exception as e:      # noqa: BLE001 - se anota y se sigue con el siguiente
                    estado = f"error: {type(e).__name__}"
                    time.sleep(10 * (intento + 1))
            w.writerow([fila.CresCodigoFormato, fila.NumeroInforme, fila.Periodo, ruta.name if estado == "ok" else "",
                        tam, paginas, con_texto, estado])
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(pendientes)}", flush=True)
            time.sleep(PAUSA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotes", nargs="*", default=None)
    main(ap.parse_args().lotes)
