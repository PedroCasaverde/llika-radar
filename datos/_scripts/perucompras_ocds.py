"""Órdenes de los catálogos electrónicos de Perú Compras (fuente C3 de TAREAS), que CONOSCE no trae: una parte de las
órdenes de las resoluciones y de los informes que no se hallaron en CONOSCE son de catálogo (HALLAZGOS 71).

  python -X utf8 perucompras_ocds.py

Usa la exportación pública en formato OCDS del buscador de órdenes (catalogos.perucompras.gob.pe/ConsultaOrdenesPub,
sin captcha, licencia Open Data Commons Attribution): una descarga por acuerdo marco y año, de 2018 a 2026. Una sola
conexión, 2 segundos entre pedidos. Guarda cada archivo comprimido en datos/crudos/perucompras_ocds/<año>/; si se corta,
sigue donde se quedó (registro en registro.csv) y no repite ninguno. Un 500 es un año sin órdenes de ese acuerdo:
no se reintenta (si no, cada uno costaba un minuto de esperas).
"""
import csv
import gzip
import time
from pathlib import Path

import requests

BASE = "https://catalogos.perucompras.gob.pe/ConsultaOrdenesPub/"
DEST = Path(__file__).resolve().parents[1] / "crudos" / "perucompras_ocds"
ANIOS = range(2018, 2027)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (proyecto académico UTEC; un pedido cada 2 segundos)"
    filtros = s.post(BASE + "obtenerFiltros", timeout=120).text
    acuerdos = sorted({a.split("^")[0].split("-")[0] for a in filtros.split("¯")[0].split("¬") if "^" in a}, key=int)
    reg = DEST / "registro.csv"
    hechos = set()
    if reg.exists():
        with reg.open(encoding="utf-8") as f:
            hechos = {(r["acuerdo"], r["anio"]) for r in csv.DictReader(f) if r["estado"] in ("ok", "vacio", "http_500")}
    pend = [(a, y) for a in acuerdos for y in ANIOS if (a, str(y)) not in hechos]
    print(f"{len(acuerdos)} acuerdos marco; pedidos pendientes: {len(pend)}", flush=True)
    nuevo = not reg.exists()
    with reg.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["acuerdo", "anio", "estado", "bytes", "releases"])
        for k, (a, y) in enumerate(pend, 1):
            url = (f"{BASE}DescargaJsonOCDS?pAcuerdo={a}&pEntidad=&pProveedor=&pFechaIni={y}0101"
                   f"&pFechaFin={y}1201")
            estado, tam, n = "error", 0, 0
            for intento in range(3):
                try:
                    r = s.get(url, timeout=900)
                    if r.status_code == 200 and r.content[:1] == b"{":
                        n = r.content.count(b'"ocid"')
                        tam = len(r.content)
                        if n:
                            (DEST / str(y)).mkdir(exist_ok=True)
                            (DEST / str(y) / f"acuerdo_{a}_{y}.json.gz").write_bytes(gzip.compress(r.content))
                        estado = "ok" if n else "vacio"
                        break
                    estado = f"http_{r.status_code}"
                    if r.status_code == 500:      # el servidor da 500 cuando el acuerdo no tuvo órdenes ese año
                        break
                except requests.RequestException as e:
                    estado = f"error_{type(e).__name__}"
                time.sleep(10 * (intento + 1))
            w.writerow([a, y, estado, tam, n])
            f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(pend)}", flush=True)
            time.sleep(2)
    print("Listo", flush=True)


if __name__ == "__main__":
    main()
