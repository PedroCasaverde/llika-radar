# Baja la historia de un adjunto de la wiki de datos abiertos del OECE: una versión por mes (la primera publicada en
# cada mes), o todas. Sirve para sancionados, multas, penalidades e inhabilitaciones, que la wiki versiona cada semana.
# Uso: python bajar_historia_wiki.py sancionados           (una versión por mes)
#      python bajar_historia_wiki.py penalidades --todas
import argparse
import csv
import datetime as dt
import json
import shutil
import time
import urllib.request
from pathlib import Path

DATOS = Path(__file__).resolve().parents[1]
REGISTRO = DATOS / "crudos" / "_registro_descargas.csv"
WIKI = "https://osce-gob-pe.atlassian.net/wiki"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
# nombre corto -> (página, id del adjunto, archivo, carpeta de destino)
ADJUNTOS = {
    "sancionados": ("106889269", "att106892480", "sancionados.csv", "acc7_sancionados"),
    "multas": ("106889276", "att106892486", "sancionados_multa.csv", "oece_multas"),
    "penalidades": ("106889271", "att106892471", "penalidades.csv", "oece_penalidades"),
    "penalidades2018": ("106889271", "att106892475", "penalidades2018.csv", "oece_penalidades"),
    "inhabilitaciones_judiciales": ("106889261", "att106892498", "inhabilitaciones_judiciales.csv", "oece_inhabilitaciones_judiciales"),
}


def versiones(adjunto):
    """Lista de (número, fecha de publicación) de todas las versiones del adjunto."""
    salida, inicio = [], 0
    while True:
        url = f"{WIKI}/rest/api/content/{adjunto}/version?limit=200&start={inicio}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            res = json.load(r)["results"]
        salida += [(v["number"], v["when"][:10]) for v in res]
        if len(res) < 200:
            return sorted(salida)
        inicio += 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fuente", choices=sorted(ADJUNTOS))
    ap.add_argument("--todas", action="store_true", help="todas las versiones, no solo una por mes")
    a = ap.parse_args()
    pagina, adjunto, archivo, carpeta = ADJUNTOS[a.fuente]
    destino = DATOS / "crudos" / carpeta / "_historia"
    destino.mkdir(parents=True, exist_ok=True)
    vs = versiones(adjunto)
    if not a.todas:                                   # la primera versión publicada en cada mes
        por_mes = {}
        for n, f in vs:
            por_mes.setdefault(f[:7], (n, f))
        vs = sorted(por_mes.values())
    print(f"{a.fuente}: {len(vs)} versiones a bajar", flush=True)
    with open(REGISTRO, "a", newline="", encoding="utf-8") as fr:
        w = csv.writer(fr)
        for n, f in vs:
            final = destino / f"{Path(archivo).stem}_v{n:03d}_{f}.csv"
            if final.exists():
                continue
            url = f"{WIKI}/download/attachments/{pagina}/{archivo}?version={n}&api=v2"
            parcial = final.with_suffix(".part")
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(parcial, "wb") as out:
                    shutil.copyfileobj(r, out, length=1 << 20)
                parcial.replace(final)
                estado = "bajado"
            except Exception as e:                    # una versión que falla no detiene las demás
                estado = f"error: {str(e)[:80]}"
            w.writerow([f"{carpeta}/_historia", final.name, url, final.stat().st_size if final.exists() else 0, f, estado,
                        dt.datetime.now().isoformat(timespec="seconds")])
            fr.flush()
            print(f"  v{n} ({f}) {estado}", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
