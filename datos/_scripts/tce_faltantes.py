"""Busca en gob.pe, una por una, las resoluciones de sanción (desde 2018) que no salieron en el índice por meses
(PENDIENTES, sección 4: ~830; 400 son de 2025, de la transición de TCE a TCP). Las que encuentra se agregan a
indice.csv con coleccion «busqueda», y después «tce_resoluciones.py bajar» y «texto» las bajan y leen como a las demás.

  python -X utf8 tce_faltantes.py

Una sola conexión, 1 segundo entre pedidos. Guarda cada búsqueda en faltantes_busqueda.csv: si se corta, sigue donde
se quedó y no repite ninguna.
"""
import csv
import re
import time

import pandas as pd
import requests

from tce_resoluciones import DATOS, SALIDA, normalizar

BUSCA = "https://www.gob.pe/busquedas"
ENLACE = re.compile(r"normas-legales/(\d+)-([a-z0-9-]+)")


def sin_trib(c):
    return re.sub(r"-T[A-Z]+", "", c) if isinstance(c, str) else None


def main():
    s = pd.read_parquet(DATOS / "procesados" / "radar" / "sanciones.parquet")
    s = s[s["inicio"] >= "2018-01-01"]
    ind = pd.read_csv(SALIDA / "indice.csv", dtype=str)
    en_indice = {sin_trib(normalizar(x)) for x in ind["numero"]}
    s["k"] = s["resolucion"].map(lambda x: sin_trib(normalizar(x)))
    faltan = s[~s["k"].isin(en_indice)].drop_duplicates("k")
    registro = SALIDA / "faltantes_busqueda.csv"
    hechos = set(pd.read_csv(registro, dtype=str)["k"]) if registro.exists() else set()
    pendientes = faltan[~faltan["k"].isin(hechos)]
    print(f"Faltan en el índice: {len(faltan)}; ya buscadas: {len(hechos)}; por buscar: {len(pendientes)}", flush=True)
    sesion = requests.Session()
    sesion.headers["User-Agent"] = "Mozilla/5.0 (proyecto académico UTEC; una consulta por segundo)"
    nuevo = not registro.exists()
    with registro.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["k", "resolucion", "numero", "url"])
        for i, fila in enumerate(pendientes.itertuples(), 1):
            hallado = ("", "")
            variantes = [fila.resolucion, fila.resolucion.replace("TCE", "TCP"), fila.resolucion.replace("TCP", "TCE")]
            for q in dict.fromkeys(variantes):
                try:
                    r = sesion.get(BUSCA, params={"term": q, "institucion[]": "oece"}, timeout=60)
                    time.sleep(1)
                except requests.RequestException:
                    time.sleep(10)
                    continue
                for ident, slug in ENLACE.findall(r.text):
                    num = normalizar(slug.upper())
                    if sin_trib(num) == fila.k:
                        hallado = (num, f"https://www.gob.pe/institucion/oece/normas-legales/{ident}-{slug}")
                        break
                if hallado[0]:
                    break
            w.writerow([fila.k, fila.resolucion, *hallado])
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(pendientes)}", flush=True)
    b = pd.read_csv(registro, dtype=str)
    hallados = b[b["url"].notna() & ~b["url"].isin(set(ind["url"]))]
    if len(hallados):
        extra = pd.DataFrame({"coleccion": "busqueda", "fecha_publicacion": "", "titulo": "", "numero": hallados["numero"],
                              "url": hallados["url"], "resumen_corto": "", "formato": ""})
        extra.drop_duplicates("url").to_csv(SALIDA / "indice.csv", mode="a", header=False, index=False)
    print(f"Halladas: {b['url'].notna().sum()} de {len(b)}; agregadas al índice: {len(hallados)}", flush=True)


if __name__ == "__main__":
    main()
