"""Resoluciones del Tribunal de Contrataciones (TCE hasta abril de 2025; TCP desde entonces), publicadas en gob.pe.
Fuente D3 de TAREAS_FUENTES_Y_DATOS.md; plan en HALLAZGOS 67.

  python -X utf8 tce_resoluciones.py indice     # paso 1: índice de todas las resoluciones publicadas desde 2018
  python -X utf8 tce_resoluciones.py bajar      # paso 2: página (resumen completo) y PDF de las que sancionaron
  python -X utf8 tce_resoluciones.py bajar --todas-pas   # además, los demás procedimientos sancionadores
  python -X utf8 tce_resoluciones.py bajar --pausa 2 --esperar 90   # más lento, y empieza en 90 minutos
  python -X utf8 tce_resoluciones.py texto      # paso 3: texto de cada PDF bajado (se puede repetir: solo lee lo nuevo)

El listado de gob.pe corta en ~400 páginas, así que se recorre mes por mes con el filtro de fechas (100 por página).
Una sola conexión y pausa corta; si el servidor responde con error, espera y baja el ritmo. Un 403 o 404 no se
reintenta: repetir el pedido no cambia la respuesta. Los PDF que fallaron se reintentan una vez al final de la
siguiente corrida. Si 10 PDF nuevos seguidos fallan, pide 200 bytes de uno que ya bajó bien: si llegan, sigue; si no,
el servidor está rechazando y se detiene sin insistir.
El 31/10 los 403 no fueron un bloqueo ni archivos faltantes: el enlace se leía cortado (ver PDF, más abajo).
Salida en datos/crudos/tce_resoluciones/: indice.csv, paginas.csv (resumen completo y enlace al PDF), pdf/<año>/ y
textos.jsonl.
"""
import argparse
import csv
import html
import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATOS = Path(__file__).resolve().parents[1]
SALIDA = DATOS / "crudos" / "tce_resoluciones"
COLECCIONES = {
    "TCE": "https://www.gob.pe/institucion/oece/colecciones/716-resoluciones-del-tribunal-de-contrataciones-del-estado",
    "TCP": "https://www.gob.pe/institucion/oece/colecciones/68030-resoluciones-del-tribunal-de-contrataciones-publicas",
}
MESES = {m: i for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                                      "septiembre", "octubre", "noviembre", "diciembre"], 1)}
MESES["setiembre"] = 9
PAUSA = 0.5
sesion = requests.Session()
sesion.headers["User-Agent"] = "LLIKA-UTEC (proyecto académico; contacto: pedro.casaverde@utec.edu.pe)"


def pedir(url, **kw):
    pausa = PAUSA
    for intento in range(5):
        try:
            r = sesion.get(url, timeout=120, **kw)
            if r.status_code in (403, 404, 410):  # el archivo no está disponible: reintentar no lo cambia
                print(f"  {r.status_code}: {url}"[:120], flush=True)
                time.sleep(PAUSA)
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            time.sleep(PAUSA)
            return r
        except Exception as e:      # noqa: BLE001
            pausa = min(60, pausa * 3)
            print(f"  reintento {intento + 1}: {type(e).__name__} {e}"[:120], flush=True)
            time.sleep(pausa)
    return None


def entrega(url):
    """¿El servidor entrega todavía los primeros 200 bytes de un PDF que ya bajó bien?"""
    try:
        r = sesion.get(url, timeout=60, headers={"Range": "bytes=0-199"})
        time.sleep(PAUSA)
        return r.status_code in (200, 206) and r.content.startswith(b"%PDF")
    except Exception:               # noqa: BLE001
        return False


def normalizar(numero, anio_respaldo=None):
    """'0484-2019-TCE-S3' y '484-2019-TCE-S3' -> '484-2019-TCE-S3'. Tolera los errores de tipeo de gob.pe:
    etiquetas HTML, separadores raros («1269-2018—TCE», «0178.2019», «2028- 2019») y años mal escritos («20189»,
    «2'019»); si el año no se puede leer, usa el de publicación (anio_respaldo)."""
    t = re.sub(r"<[^>]+>", "", str(numero)).upper().replace("'", "")
    m = re.match(r"\D*0*(\d+)\s*[-.—–]?\s*(\d{3,5})?\s*[-.—–]*\s*(T[A-Z]{1,3})\s*-?\s*(S\s*\d+)?", t)
    if not m:
        return None
    anio = m.group(2) or ""
    if not re.fullmatch(r"20[12]\d", anio):
        cand = re.search(r"20[12]\d", anio)
        anio = str(anio_respaldo) if anio_respaldo else (cand.group(0) if cand else None)
    if not anio:
        return None
    sala = re.sub(r"\s", "", m.group(4)) if m.group(4) else None
    return f"{m.group(1)}-{anio}-{m.group(3)}" + (f"-{sala}" if sala else "")


def fecha(texto):
    m = re.match(r"(\d{1,2}) de (\w+) de (\d{4})", texto.strip().lower())
    return date(int(m.group(3)), MESES[m.group(2)], int(m.group(1))).isoformat() if m and m.group(2) in MESES else None


TARJETA = re.compile(r'<div class="border-b border-gray-700 pb-2 font-medium mb-3">([^<]+)</div>'
                     r'<a class="leading-6 font-bold" href="(/institucion/oece/normas-legales/[^"]+)">([^<]+)</a>'
                     r'([^<]*)<div class="mt-3 leading-5 text-sm">([^<]*)</div>')


def indice():
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = []
    hoy = date.today()
    for coleccion, url in COLECCIONES.items():
        for anio in range(2018, hoy.year + 1):
            for mes in range(1, 13):
                if (anio, mes) > (hoy.year, hoy.month):
                    break
                ini = date(anio, mes, 1)
                fin = date(anio + (mes == 12), mes % 12 + 1, 1)
                pagina = 1
                while True:
                    r = pedir(url, params={"filter[start_date]": ini.isoformat(),
                                           "filter[end_date]": (fin.fromordinal(fin.toordinal() - 1)).isoformat(),
                                           "filter[per_page]": 100, "sheet": pagina})
                    if r is None:
                        print(f"{coleccion} {anio}-{mes:02d} página {pagina}: falló", flush=True)
                        break
                    tarjetas = TARJETA.findall(r.text)
                    for f, enlace, titulo, resumen, formato in tarjetas:
                        filas.append({"coleccion": coleccion, "fecha_publicacion": fecha(f), "titulo": html.unescape(titulo),
                                      "numero": normalizar(titulo.split("°")[-1], (fecha(f) or "")[:4]),
                                      "url": "https://www.gob.pe" + enlace,
                                      "resumen_corto": html.unescape(resumen).strip(), "formato": formato.strip()})
                    total = re.search(r"([\d.]+)\s*resultados", r.text)
                    total = int(total.group(1).replace(".", "")) if total else 0
                    if pagina * 100 >= total or not tarjetas:
                        break
                    pagina += 1
                print(f"{coleccion} {anio}-{mes:02d}: {total} resoluciones", flush=True)
    d = pd.DataFrame(filas).drop_duplicates("url")
    d.to_csv(SALIDA / "indice.csv", index=False)
    print(f"Índice: {len(d)} resoluciones")


DESCRIPCION = re.compile(r'(?s)<div class="[^"]*(?:description|leading-relaxed)[^"]*">(.*?)</div>')
# El enlace completo, hasta las comillas: muchos archivos de 2020 y 2021 se llaman «….pdf.pdf?v=…», y cortar en el
# primer «.pdf» deja un enlace que el CDN rechaza con 403. La vista previa («….pdf.jpg?v=…») no calza.
PDF = re.compile(r'https://cdn\.www\.gob\.pe/uploads/document/file/[^"\s<>\']+\.pdf(?:\?v=\d+)?(?=["\s<>\'])',
                 re.IGNORECASE)


def bajar(todas_pas):
    ind = pd.read_csv(SALIDA / "indice.csv", dtype=str)
    sanc = pd.read_parquet(DATOS / "procesados" / "radar" / "sanciones.parquet")
    sanc = sanc[sanc.inicio.dt.year >= 2018]
    codigos = {normalizar(c) for c in sanc.resolucion.dropna()} - {None}
    # En la transición de 2025 el mismo número figura como TCE en ACC-7 y como TCP en gob.pe (o al revés): si no hay
    # coincidencia exacta, se acepta el mismo número, año y sala con el otro nombre del Tribunal.
    sin_trib = lambda c: re.sub(r"-T[A-Z]+", "", c) if isinstance(c, str) else None
    flexibles = {sin_trib(c) for c in codigos}
    ind["sanciono"] = ind.numero.isin(codigos) | ind.numero.map(sin_trib).isin(flexibles)
    ind["es_pas"] = ind.resumen_corto.str.contains("sancionador", case=False, na=False)
    objetivo = ind[ind.sanciono | (ind.es_pas if todas_pas else False)].copy()
    objetivo["anio"] = objetivo.numero.str.extract(r"-(\d{4})-")[0]
    print(f"Sanciones desde 2018: {len(codigos)} resoluciones; en el índice: {ind.sanciono.sum()}; "
          f"a bajar: {len(objetivo)}", flush=True)
    registro = SALIDA / "paginas.csv"
    previo = pd.read_csv(registro, dtype=str, on_bad_lines="skip") if registro.exists() else None
    # lo que no bajó (sin página, sin enlace o PDF rechazado) se vuelve a intentar, al final y sin contar para el
    # corte; la fila nueva queda después de la vieja y es la que vale
    ultimo = previo if previo is not None else pd.DataFrame(columns=["url", "estado"])
    ultimo = ultimo.drop_duplicates("url", keep="last")
    hechos = set(ultimo.loc[ultimo.estado.eq("ok"), "url"])
    objetivo["reintento"] = objetivo.url.isin(set(ultimo.loc[ultimo.estado.ne("ok"), "url"]))
    bien = ultimo.loc[ultimo.estado.eq("ok"), "pdf_url"]
    control = bien.iloc[-1] if len(bien) else None  # un PDF que ya bajó bien: ¿el servidor sigue entregando?
    nuevo = not registro.exists()
    t0, seguidos = time.time(), 0
    with open(registro, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["url", "numero", "anio", "resumen_completo", "pdf_url", "archivo", "bytes", "estado"])
        pendientes = objetivo[~objetivo.url.isin(hechos)].sort_values("reintento", kind="stable")
        for k, fila in enumerate(pendientes.itertuples(), 1):
            r = pedir(fila.url)
            resumen, pdf_url, archivo, tam, estado = "", "", "", "", "sin_pagina"
            if r is not None:
                d = DESCRIPCION.search(r.text)
                resumen = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", d.group(1)))).strip() if d else ""
                m = PDF.search(r.text)
                pdf_url = m.group(0) if m else ""
                estado = "sin_pdf"
                if pdf_url:
                    p = pedir(pdf_url)
                    if p is not None and p.content.startswith(b"%PDF"):
                        destino = SALIDA / "pdf" / str(fila.anio)
                        destino.mkdir(parents=True, exist_ok=True)
                        archivo = f"{fila.numero}.pdf"
                        (destino / archivo).write_bytes(p.content)
                        tam, estado, control = len(p.content), "ok", pdf_url
                    else:
                        estado = "pdf_fallo"
            w.writerow([fila.url, fila.numero, fila.anio, resumen, pdf_url, archivo, tam, estado])
            f.flush()
            if not fila.reintento:
                seguidos = seguidos + 1 if estado == "pdf_fallo" else 0
            if seguidos >= 10:
                # si un PDF que ya bajó bien todavía llega, el servidor no nos rechaza: el problema es de esos PDF
                if control and entrega(control):
                    print(f"  10 PDF seguidos que no bajaron ({k} de {len(pendientes)}); el servidor entrega bien: "
                          "sigo", flush=True)
                    seguidos = 0
                else:
                    print(f"10 PDF nuevos seguidos rechazados ({k} de {len(pendientes)}): el servidor no acepta más "
                          "por ahora. Me detengo; correr de nuevo más tarde.", flush=True)
                    return
            if k % 200 == 0:
                m = (time.time() - t0) / 60
                print(f"{k}/{len(pendientes)} en {m:.0f} min (quedan ~{m / k * (len(pendientes) - k):.0f} min)", flush=True)
    print("Listo", flush=True)


def texto():
    """Paso 3: el texto de cada PDF bajado, con PyMuPDF, en textos.jsonl (una línea por resolución, páginas separadas
    por salto de página). Cuenta las páginas casi sin texto: son escaneos sin capa de texto, que irían al OCR. Al
    resumen de gob.pe se le quita la frase final «Esta norma pertenece al compendio…»."""
    import fitz
    pag = pd.read_csv(SALIDA / "paginas.csv", dtype=str, on_bad_lines="skip")
    pag = pag[pag.estado.eq("ok") & pag.archivo.notna()]
    destino = SALIDA / "textos.jsonl"
    hechos = set()
    if destino.exists():
        with open(destino, encoding="utf-8") as f:
            hechos = {json.loads(linea)["numero"] for linea in f}
    pendientes = pag[~pag.numero.isin(hechos)]
    t0, sin_texto, fallos = time.time(), 0, 0
    with open(destino, "a", encoding="utf-8") as f:
        for fila in pendientes.itertuples():
            try:
                with fitz.open(SALIDA / "pdf" / str(fila.anio) / fila.archivo) as d:
                    paginas = [p.get_text() for p in d]
            except Exception:  # noqa: BLE001
                fallos += 1
                continue
            vacias = sum(len(p.strip()) < 200 for p in paginas)
            sin_texto += vacias == len(paginas)
            resumen = re.sub(r"\s*Esta norma pertenece al compendio.*$", "", str(fila.resumen_completo)).strip()
            f.write(json.dumps({"numero": fila.numero, "anio": fila.anio, "resumen": resumen, "paginas": len(paginas),
                                "paginas_sin_texto": vacias, "texto": "\f".join(paginas)}, ensure_ascii=False) + "\n")
    print(f"Textos nuevos: {len(pendientes) - fallos} en {time.time() - t0:.0f} s; sin capa de texto en ninguna página: "
          f"{sin_texto}; PDF que no abren: {fallos}; ya estaban: {len(hechos)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["indice", "bajar", "texto"])
    ap.add_argument("--todas-pas", action="store_true")
    ap.add_argument("--pausa", type=float, default=PAUSA, help="segundos entre pedidos (0,5 por omisión)")
    ap.add_argument("--esperar", type=float, default=0, help="minutos de espera antes de empezar")
    a = ap.parse_args()
    PAUSA = a.pausa
    time.sleep(a.esperar * 60)
    {"indice": indice, "bajar": lambda: bajar(a.todas_pas), "texto": texto}[a.paso]()
