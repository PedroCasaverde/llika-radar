"""Primera pasada sobre los informes descargados (HALLAZGOS 65): saca gratis el texto de las páginas que ya lo traen
(PyMuPDF) y deja anotadas las páginas escaneadas para la pasada de OCR. Se puede correr las veces que haga falta:
solo procesa los PDF nuevos del inventario.

Uso:  python -X utf8 leer_texto_digital.py
Salida: texto_todos/<año>/<informe>.txt  y  paginas_para_ocr.csv  (informe, página)
"""
import csv
from pathlib import Path

import fitz
import pandas as pd

AQUI = Path(__file__).parent
MIN_CHARS = 50                      # menos que esto en una página = escaneada
SALIDA = AQUI / "texto_todos"
PARA_OCR = AQUI / "paginas_para_ocr.csv"
HECHOS = AQUI / "texto_digital_hechos.csv"


def main():
    inv = pd.read_csv(AQUI / "inventario_pdf.csv", dtype=str)
    inv = inv[(inv.estado == "ok") & (inv.paginas_con_texto.astype(float) > 0)]
    hechos = set(pd.read_csv(HECHOS, dtype=str).archivo) if HECHOS.exists() else set()
    pendientes = inv[~inv.archivo.isin(hechos)]
    print(f"PDF con texto digital: {len(inv)}; nuevos: {len(pendientes)}")
    n_pag = n_ocr = 0
    with open(PARA_OCR, "a", newline="", encoding="utf-8") as f_ocr, \
         open(HECHOS, "a", newline="", encoding="utf-8") as f_hechos:
        w_ocr, w_hechos = csv.writer(f_ocr), csv.writer(f_hechos)
        if f_ocr.tell() == 0:
            w_ocr.writerow(["Periodo", "archivo", "pagina"])
        if f_hechos.tell() == 0:
            w_hechos.writerow(["archivo"])
        for fila in pendientes.itertuples():
            doc = fitz.open(AQUI / "pdf_todos" / fila.Periodo / fila.archivo)
            textos = []
            for i, pag in enumerate(doc, 1):
                t = pag.get_text()
                if len(t.strip()) >= MIN_CHARS:
                    textos.append(t)
                else:
                    textos.append("[[PAGINA_PARA_OCR]]")
                    w_ocr.writerow([fila.Periodo, fila.archivo, i])
                    n_ocr += 1
            (SALIDA / fila.Periodo).mkdir(parents=True, exist_ok=True)
            (SALIDA / fila.Periodo / (Path(fila.archivo).stem + ".txt")).write_text("\n\f".join(textos),
                                                                                    encoding="utf-8")
            w_hechos.writerow([fila.archivo])
            n_pag += doc.page_count
    print(f"Listo: {len(pendientes)} informes, {n_pag} páginas; {n_ocr} quedan para OCR")


if __name__ == "__main__":
    main()
