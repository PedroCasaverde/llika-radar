# Baja las fuentes de la hoja ACC del E4 a datos/crudos/ y deja un registro con tamaño y fecha de cada archivo.
# Uso: python bajar_fuentes.py            (no vuelve a bajar lo que ya está completo)
#      python bajar_fuentes.py --ordenes 2018 2023   (solo las órdenes de CONOSCE de esos años de registro)
import csv
import datetime as dt
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATOS = Path(__file__).resolve().parents[1]
CRUDOS = DATOS / "crudos"
REGISTRO = CRUDOS / "_registro_descargas.csv"
HOY = dt.date.today().strftime("%Y%m%d")
# datosabiertos.gob.pe responde 418 si no ve un navegador
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0 Safari/537.36"}
CONOSCE = "https://conosce.osce.gob.pe/buscador/assets/67ae6c4a/reportes"
MESES = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE",
         "NOVIEMBRE", "DICIEMBRE"]
ULTIMO_MES = (2026, 9)  # último archivo mensual publicado al 03/10/2026


def ordenes(anios):
    # ACC-1: órdenes de CONOSCE, un archivo por mes de registro (CONOSCE publica desde 2018-01)
    for anio in anios:
        for i, mes in enumerate(MESES, 1):
            if (anio, i) > ULTIMO_MES:
                break
            yield ("acc1_conosce_ordenes", f"{CONOSCE}/ordenes/{anio}/CONOSCE_ORDENESCOMPRA{mes}{anio}_0.xlsx",
                   f"ordenes_{anio}-{i:02d}.xlsx")


def fuentes():
    # Se bajan 2024-2026 porque los archivos van por fecha de registro y las órdenes emitidas en 2024 siguen
    # apareciendo en los archivos de 2025 y 2026.
    yield from ordenes((2024, 2025, 2026))
    yield ("acc3_conformacion", f"{CONOSCE}/conformacion/conformacion_juridica.csv", f"conformacion_juridica_{HOY}.csv")
    yield ("acc5_dji_empleos", "https://www.datosabiertos.gob.pe/sites/default/files/Reporte3.csv", f"Reporte3_{HOY}.csv")
    yield ("acc6_padron_sunat", "https://www2.sunat.gob.pe/padron_reducido_ruc.zip", f"padron_reducido_ruc_{HOY}.zip")
    yield ("acc7_sancionados", f"{CONOSCE}/sancionados/sancionados.csv", f"sancionados_{HOY}.csv")
    yield ("acc9_entidades", "https://osce-gob-pe.atlassian.net/wiki/download/attachments/106889265/entidades_contratantes.csv",
           f"entidades_contratantes_{HOY}.csv")


def bajar(item):
    carpeta, url, nombre = item
    destino = CRUDOS / carpeta / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            total = int(r.headers.get("Content-Length") or 0)
            modificado = r.headers.get("Last-Modified", "")
            if destino.exists() and total and destino.stat().st_size == total:
                return [carpeta, nombre, url, total, modificado, "ya estaba"]
            parcial = destino.with_suffix(destino.suffix + ".part")
            with open(parcial, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
        if total and parcial.stat().st_size != total:
            return [carpeta, nombre, url, parcial.stat().st_size, modificado, f"INCOMPLETO (esperado {total})"]
        parcial.replace(destino)
        return [carpeta, nombre, url, destino.stat().st_size, modificado, "bajado"]
    except Exception as e:  # se registra y se sigue con lo demás
        return [carpeta, nombre, url, 0, "", f"ERROR: {e}"]


if __name__ == "__main__":
    CRUDOS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 4 and sys.argv[1] == "--ordenes":
        lista = ordenes(range(int(sys.argv[2]), int(sys.argv[3]) + 1))
    else:
        lista = fuentes()
    with ThreadPoolExecutor(max_workers=4) as ex:
        filas = list(ex.map(bajar, lista))
    nuevo = not REGISTRO.exists()
    with open(REGISTRO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["carpeta", "archivo", "url", "bytes", "last_modified", "estado", "registrado"])
        for fila in filas:
            w.writerow(fila + [dt.datetime.now().isoformat(timespec="seconds")])
            print(f"{fila[5]:>10} {fila[3] / 1e6:8.1f} MB  {fila[0]}/{fila[1]}")
