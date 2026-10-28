# Baja versiones anteriores de la conformación jurídica (ACC-3) desde la wiki de datos abiertos del OECE, que guarda
# una versión por mes desde noviembre de 2021. Las deja en crudos/acc3_conformacion/_historia/ con su fecha de corte.
# Uso: python bajar_historia_conformacion.py 28 39        (versiones 28 a 39 = cortes de 2024)
import csv
import datetime as dt
import json
import shutil
import sys
import urllib.request
from pathlib import Path

DATOS = Path(__file__).resolve().parents[1]
DESTINO = DATOS / "crudos" / "acc3_conformacion" / "_historia"
REGISTRO = DATOS / "crudos" / "_registro_descargas.csv"
WIKI = "https://osce-gob-pe.atlassian.net/wiki"
PAGINA, ADJUNTO = "106889267", "att106892503"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}


def versiones() -> dict:
    """Número de versión -> fecha de publicación, según la API de la wiki."""
    url = f"{WIKI}/rest/api/content/{ADJUNTO}/version?limit=200"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return {v["number"]: v["when"][:10] for v in json.load(r)["results"]}


def bajar(n: int, publicado: str) -> list:
    url = f"{WIKI}/download/attachments/{PAGINA}/conformacion_juridica.csv?version={n}&api=v2"
    parcial = DESTINO / f"v{n:02d}.part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(parcial, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        shutil.copyfileobj(r, f, length=1 << 20)
    if total and parcial.stat().st_size != total:
        return ["acc3_conformacion/_historia", parcial.name, url, parcial.stat().st_size, publicado, f"INCOMPLETO (esperado {total})"]
    with open(parcial, "rb") as f:
        f.readline()
        corte = f.readline().split(b"|")[0].decode().strip()
    final = DESTINO / f"conformacion_juridica_v{n:02d}_corte{corte}.csv"
    parcial.replace(final)
    return ["acc3_conformacion/_historia", final.name, url, final.stat().st_size, publicado, "bajado"]


if __name__ == "__main__":
    desde, hasta = int(sys.argv[1]), int(sys.argv[2])
    DESTINO.mkdir(parents=True, exist_ok=True)
    pub = versiones()
    with open(REGISTRO, "a", newline="", encoding="utf-8") as fr:
        w = csv.writer(fr)
        for n in range(desde, hasta + 1):
            if list(DESTINO.glob(f"conformacion_juridica_v{n:02d}_corte*.csv")):
                print(f"v{n:02d} ya estaba", flush=True)
                continue
            fila = bajar(n, pub.get(n, ""))
            w.writerow(fila + [dt.datetime.now().isoformat(timespec="seconds")])
            fr.flush()
            print(f"{fila[5]:>10} {fila[3] / 1e6:7.1f} MB  {fila[1]}  (publicado {fila[4]})", flush=True)
