"""OCR de las resoluciones del Tribunal que salieron casi sin texto (PENDIENTES, sección 4): las que tienen la mitad o más
de sus páginas sin texto. Agrega a textos.jsonl una línea nueva por resolución, con "ocr": true; quien lee el archivo se
queda con la última línea de cada número.

Corre con el entorno .venv-docling de lector-tarifas (torch con CUDA y RapidOCR), igual que ocr_informes/ocr_local.py:
  E:/Python/pibot/ocr/lector-tarifas/.venv-docling/Scripts/python.exe -X utf8 ocr_resoluciones.py
"""
import json
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from rapidocr import EngineType, RapidOCR

TCE = Path(__file__).resolve().parents[1] / "crudos" / "tce_resoluciones"
DPI, MIN_CHARS = 200, 200

motor = RapidOCR(params={"Det.engine_type": EngineType.TORCH, "Cls.engine_type": EngineType.TORCH,
                         "Rec.engine_type": EngineType.TORCH, "EngineConfig.torch.use_cuda": True})


def ocr(pag):
    res = motor(np.array(pag.render(scale=DPI / 72).to_pil().convert("RGB")))
    if res.boxes is None:
        return ""
    cajas = sorted(zip(res.boxes, res.txts), key=lambda c: (round(c[0][:, 1].mean() / 15), c[0][:, 0].min()))
    return " ".join(t for _, t in cajas)


def main():
    ultimas = {}
    with (TCE / "textos.jsonl").open(encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            ultimas[d["numero"]] = d
    pendientes = [d for d in ultimas.values()
                  if not d.get("ocr") and d["paginas"] and d["paginas_sin_texto"] >= d["paginas"] / 2]
    print(f"{len(pendientes)} resoluciones con la mitad o más de páginas sin texto", flush=True)
    archivo = {}
    import csv
    with (TCE / "paginas.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("estado") == "ok":
                archivo[r["numero"]] = r["archivo"]
    with (TCE / "textos.jsonl").open("a", encoding="utf-8") as f:
        for k, d in enumerate(pendientes, 1):
            ruta = TCE / "pdf" / str(d["anio"]) / archivo.get(d["numero"], "")
            if not ruta.is_file():
                print(f"  sin PDF: {d['numero']}", flush=True)
                continue
            doc = pdfium.PdfDocument(ruta)
            partes, n_ocr = [], 0
            for i in range(len(doc)):
                t = doc[i].get_textpage().get_text_range()
                if len(t.strip()) < MIN_CHARS:
                    t, n_ocr = ocr(doc[i]), n_ocr + 1
                partes.append(t)
            nuevo = {**d, "texto": "\f".join(partes), "paginas_sin_texto": 0, "ocr": True, "paginas_ocr": n_ocr}
            f.write(json.dumps(nuevo, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {k}/{len(pendientes)} {d['numero']}: {len(doc)} páginas, {n_ocr} con OCR", flush=True)


if __name__ == "__main__":
    main()
