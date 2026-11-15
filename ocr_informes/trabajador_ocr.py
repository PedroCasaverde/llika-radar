"""Un trabajador de la lectura de informes (HALLAZGOS 65 y 66): toma su parte de un lote (los informes i con
i % de == parte), baja cada PDF, copia el texto digital y aplica OCR a las páginas escaneadas, en la GPU que le toca o en
el procesador. El notebook lanza uno por GPU (T4 x2: dos a la vez) o dos en el procesador, y después junta los resúmenes.

Uso:  python trabajador_ocr.py --lote lote.csv --salida textos/ --parte 0 --de 2 --dispositivo cuda:0
      python trabajador_ocr.py --lote lote.csv --salida textos/ --parte 1 --de 2 --dispositivo cpu --hilos 2
"""
import argparse
import csv
import re
import time
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import requests
from rapidocr import EngineType, RapidOCR

MIN_CHARS = 50          # una página con menos texto que esto se considera escaneada
COLUMNAS = ["CresCodigoFormato", "NumeroInforme", "Periodo", "archivo", "estado", "paginas", "paginas_ocr",
            "confianza_ocr", "segundos", "trabajador"]


def motor_ocr(dispositivo, hilos):
    if dispositivo.startswith("cuda"):
        gpu = int(dispositivo.split(":")[1]) if ":" in dispositivo else 0
        return RapidOCR(params={"Det.engine_type": EngineType.TORCH, "Cls.engine_type": EngineType.TORCH,
                                "Rec.engine_type": EngineType.TORCH, "EngineConfig.torch.use_cuda": True,
                                "EngineConfig.torch.cuda_ep_cfg.device_id": gpu})
    return RapidOCR(params={"EngineConfig.onnxruntime.intra_op_num_threads": hilos})


def nombre_archivo(fila):
    base = fila.NumeroInforme if isinstance(fila.NumeroInforme, str) and fila.NumeroInforme else fila.CresCodigoFormato
    return re.sub(r"[^\w-]", "_", base)


class Bajador:
    """Baja con una pausa que sube si el servidor de la Contraloría se queja y baja de a poco cuando responde bien."""

    def __init__(self, pausa):
        self.base = self.pausa = pausa
        self.sesion = requests.Session()

    def bajar(self, url, destino):
        ultimo = "error"
        for intento in range(3):
            try:
                # 20 s para conectar: si el servidor no contesta desde esta sesión, se sabe rápido (lote 07a, 31/10)
                r = self.sesion.get(url.replace("http://", "https://"), timeout=(20, 300))
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(str(r.status_code))
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    return "no_es_pdf"
                destino.write_bytes(r.content)
                self.pausa = max(self.base, self.pausa * 0.9)
                return "ok"
            except Exception as e:      # noqa: BLE001 - se anota y se sigue
                ultimo = f"error: {type(e).__name__} {e}"[:80]
                self.pausa = min(30.0, self.pausa * 2)
                time.sleep(10 * (intento + 1))
        return ultimo


def ocr_pagina(motor, pagina, dpi):
    """Lee una página escaneada; devuelve el texto en orden de lectura y la confianza media del OCR."""
    px = pagina.get_pixmap(dpi=dpi)
    img = np.frombuffer(px.samples, dtype=np.uint8).reshape(px.height, px.width, px.n)[:, :, :3]
    res = motor(np.ascontiguousarray(img))
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


def leer(motor, pdf, dpi):
    """Texto de todo el informe. Cada página empieza con una marca: [[pagina N | digital]] o [[pagina N | ocr | 0.93]]."""
    doc = fitz.open(pdf)
    partes, n_ocr, confs = [], 0, []
    for i, pag in enumerate(doc, 1):
        t = pag.get_text()
        if len(t.strip()) >= MIN_CHARS:
            partes.append(f"[[pagina {i} | digital]]\n{t}")
        else:
            t, conf = ocr_pagina(motor, pag, dpi)
            n_ocr += 1
            confs.append(conf)
            partes.append(f"[[pagina {i} | ocr | {conf:.2f}]]\n{t}")
    return "\n".join(partes), doc.page_count, n_ocr, (float(np.mean(confs)) if confs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--parte", type=int, default=0)
    ap.add_argument("--de", type=int, default=1)
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pausa", type=float, default=0.3)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--hasta", type=float, default=0, help="hora (epoch) en que deja de tomar informes nuevos")
    a = ap.parse_args()

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    tmp = salida.parent / f"pdf_tmp_{a.parte}"
    tmp.mkdir(exist_ok=True)
    resumen = salida / f"resumen_{a.parte}.csv"
    lote = pd.read_csv(a.lote, dtype=str)
    if a.limite:
        lote = lote.head(a.limite)
    mia = lote.iloc[a.parte::a.de]
    hechos = set(pd.read_csv(resumen, dtype=str).CresCodigoFormato) if resumen.exists() else set()
    pendientes = mia[~mia.CresCodigoFormato.isin(hechos)]
    print(f"trabajador {a.parte} ({a.dispositivo}): {len(pendientes)} informes por leer", flush=True)

    motor = motor_ocr(a.dispositivo, a.hilos)
    bajador = Bajador(a.pausa)
    nuevo = not resumen.exists()
    t_inicio = time.time()
    sin_conexion = 0    # informes seguidos que no se pudieron bajar por falta de conexión
    with open(resumen, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(COLUMNAS)
        for k, fila in enumerate(pendientes.itertuples(), 1):
            if a.hasta and time.time() > a.hasta:     # para no pasar las 12 horas de Kaggle: lo leído se guarda
                print(f"trabajador {a.parte}: límite de tiempo; quedan {len(pendientes) - k + 1} sin leer", flush=True)
                break
            t0 = time.time()
            nombre = nombre_archivo(fila)
            pdf = tmp / f"{nombre}.pdf"
            estado = bajador.bajar(fila.RutaCloudInforme, pdf)
            paginas = n_ocr = conf = ""
            if estado == "ok":
                try:
                    texto, paginas, n_ocr, conf = leer(motor, pdf, a.dpi)
                    (salida / str(fila.Periodo)).mkdir(exist_ok=True)
                    (salida / str(fila.Periodo) / f"{nombre}.txt").write_text(texto, encoding="utf-8")
                    conf = "" if conf is None else round(conf, 3)
                except Exception as e:      # noqa: BLE001
                    estado = f"error_lectura: {type(e).__name__}"
                pdf.unlink(missing_ok=True)
            w.writerow([fila.CresCodigoFormato, fila.NumeroInforme, fila.Periodo, f"{nombre}.txt", estado, paginas,
                        n_ocr, conf, round(time.time() - t0, 1), a.parte])
            f.flush()
            # Si la Contraloría no contesta a esta sesión (le pasó al lote 07a el 31/10: 11 horas sin bajar nada, mientras
            # otras sesiones bajaban bien), se detiene para no gastar tiempo; lo que falte se relanza en otra sesión.
            sin_conexion = sin_conexion + 1 if estado.startswith("error") and ("Connect" in estado or
                                                                               "Timeout" in estado) else 0
            if sin_conexion >= 8:
                print(f"trabajador {a.parte}: 8 informes seguidos sin conexión con la Contraloría; me detengo. "
                      f"Quedan {len(pendientes) - k} sin leer", flush=True)
                break
            if k % 10 == 0 or k == len(pendientes):
                m = (time.time() - t_inicio) / 60
                print(f"trabajador {a.parte}: {k}/{len(pendientes)} en {m:.0f} min", flush=True)
            time.sleep(bajador.pausa)


if __name__ == "__main__":
    main()
