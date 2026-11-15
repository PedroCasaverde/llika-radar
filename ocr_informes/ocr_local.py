"""Lee en esta computadora los informes que ya están descargados en datos/crudos/acc8_informes_cgr/pdf_todos (HALLAZGOS
65), con la misma salida que el notebook: un .txt por informe con una marca por página y un resumen.csv.

Corre con el entorno .venv-docling de lector-tarifas (torch con CUDA y RapidOCR PP-OCRv6 ya instalados); para no
tocar ese entorno, lee los PDF con pypdfium2, que ya trae, en vez de PyMuPDF.

Uso:  E:/Python/pibot/ocr/lector-tarifas/.venv-docling/Scripts/python.exe -X utf8 ocr_local.py
Se puede cortar y volver a correr: salta los informes que ya están en el resumen.
Para usar más la GPU (un solo proceso la deja a medias, porque dibuja cada página en el procesador): dos procesos con
--parte 0 --de 2 y --parte 1 --de 2. Cada parte escribe su propio resumen (resumen.csv y resumen_p1.csv).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from rapidocr import EngineType, RapidOCR

ACC8 = Path(__file__).resolve().parent.parent / "datos" / "crudos" / "acc8_informes_cgr"
SALIDA = ACC8 / "textos_local"
RESUMEN = SALIDA / "resumen.csv"
DPI = 200
MIN_CHARS = 50

motor = RapidOCR(params={"Det.engine_type": EngineType.TORCH, "Cls.engine_type": EngineType.TORCH,
                         "Rec.engine_type": EngineType.TORCH, "EngineConfig.torch.use_cuda": True})


def ocr_pagina(pag):
    img = np.array(pag.render(scale=DPI / 72).to_pil().convert("RGB"))
    res = motor(img)
    if res.boxes is None:
        return "", 0.0
    cajas = sorted(zip(res.boxes, res.txts), key=lambda c: (c[0][:, 1].mean(), c[0][:, 0].min()))
    lineas, actual, y_prev = [], [], None
    for caja, txt in cajas:
        y, alto = caja[:, 1].mean(), np.ptp(caja[:, 1])
        if y_prev is not None and abs(y - y_prev) > alto * 0.6:
            lineas.append(" ".join(t for _, t in sorted(actual)))
            actual = []
        actual.append((caja[:, 0].min(), txt))
        y_prev = y
    lineas.append(" ".join(t for _, t in sorted(actual)))
    return "\n".join(lineas), float(np.mean(res.scores))


def leer(ruta):
    doc = pdfium.PdfDocument(ruta)
    partes, n_ocr, confs = [], 0, []
    for i in range(len(doc)):
        pag = doc[i]
        t = pag.get_textpage().get_text_range()
        if len(t.strip()) >= MIN_CHARS:
            partes.append(f"[[pagina {i + 1} | digital]]\n{t}")
        else:
            t, conf = ocr_pagina(pag)
            n_ocr += 1
            confs.append(conf)
            partes.append(f"[[pagina {i + 1} | ocr | {conf:.2f}]]\n{t}")
    return "\n".join(partes), len(doc), n_ocr, (float(np.mean(confs)) if confs else None)


def main(parte=0, de=1):
    global RESUMEN
    inv = pd.read_csv(ACC8 / "inventario_pdf.csv", dtype=str)
    inv = inv[inv.estado == "ok"]
    SALIDA.mkdir(exist_ok=True)
    hechos = set()
    for r in SALIDA.glob("resumen*.csv"):
        hechos |= set(pd.read_csv(r, dtype=str).CresCodigoFormato)
    if parte:
        RESUMEN = SALIDA / f"resumen_p{parte}.csv"
    pendientes = inv[~inv.CresCodigoFormato.isin(hechos)]
    pendientes = pendientes[pd.util.hash_pandas_object(pendientes.CresCodigoFormato, index=False) % de == parte]
    # Fondo primero: la prioridad sale de clasificar las fichas resumen (HALLAZGOS 66, paso 2).
    prio = Path(__file__).resolve().parent / "prioridad_informes.csv"
    if prio.exists():
        p = pd.read_csv(prio, dtype={"CresCodigoFormato": str})[["CresCodigoFormato", "prioridad"]]
        pendientes = pendientes.merge(p, on="CresCodigoFormato", how="left")
        pendientes = pendientes.sort_values("prioridad", na_position="last")
    print(f"Ya hechos: {len(hechos)}; por hacer: {len(pendientes)}", flush=True)
    nuevo = not RESUMEN.exists()
    t_inicio = time.time()
    with open(RESUMEN, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["CresCodigoFormato", "NumeroInforme", "Periodo", "archivo", "estado", "paginas",
                        "paginas_ocr", "confianza_ocr", "segundos"])
        for k, fila in enumerate(pendientes.itertuples(), 1):
            t0 = time.time()
            nombre = Path(fila.archivo).stem
            estado, paginas, n_ocr, conf = "ok", "", "", ""
            try:
                texto, paginas, n_ocr, conf = leer(ACC8 / "pdf_todos" / fila.Periodo / fila.archivo)
                (SALIDA / fila.Periodo).mkdir(exist_ok=True)
                (SALIDA / fila.Periodo / f"{nombre}.txt").write_text(texto, encoding="utf-8")
                conf = "" if conf is None else round(conf, 3)
            except Exception as e:      # noqa: BLE001 - se anota y se sigue
                estado = f"error_lectura: {type(e).__name__}"
            w.writerow([fila.CresCodigoFormato, fila.NumeroInforme, fila.Periodo, f"{nombre}.txt", estado, paginas,
                        n_ocr, conf, round(time.time() - t0, 1)])
            f.flush()
            if k % 25 == 0 or k == len(pendientes):
                m = (time.time() - t_inicio) / 60
                print(f"{k}/{len(pendientes)} en {m:.0f} min (quedan ~{m / k * (len(pendientes) - k):.0f} min)",
                      flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", type=int, default=0)
    ap.add_argument("--de", type=int, default=1)
    a = ap.parse_args()
    main(a.parte, a.de)
