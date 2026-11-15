"""Paso 1 de HALLAZGOS 66: baja la ficha resumen (TIPOARCHIVO=RE) de cada informe de la lista ampliada. Es digital y
pesa ~30 KB; el servidor responde en ~0,12 s. Una sola conexión y una pausa corta; si el servidor da error, espera y
baja el ritmo. Se puede cortar y volver a correr: salta las que ya están.

Uso:  python -X utf8 bajar_fichas.py [--limite N]
Salida: fichas/pdf/<año>/<CresCodigoFormato>.pdf  e  fichas/inventario_fichas.csv
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import requests

AQUI = Path(__file__).parent
DESTINO = AQUI / "fichas" / "pdf"
INVENTARIO = AQUI / "fichas" / "inventario_fichas.csv"
PAUSA = 0.2


def main(limite):
    lista = pd.read_csv(AQUI / "listas" / "lista_unica_informes_ampliada.csv", dtype=str)
    lista = lista[lista.ResumenEjecutivo.notna()]
    DESTINO.mkdir(parents=True, exist_ok=True)
    hechos = set(pd.read_csv(INVENTARIO, dtype=str).CresCodigoFormato) if INVENTARIO.exists() else set()
    pendientes = lista[~lista.CresCodigoFormato.isin(hechos)]
    if limite:
        pendientes = pendientes.head(limite)
    print(f"Ya bajadas: {len(hechos)}; por bajar: {len(pendientes)}", flush=True)
    nuevo = not INVENTARIO.exists()
    sesion = requests.Session()
    pausa, t0 = PAUSA, time.time()
    with open(INVENTARIO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["CresCodigoFormato", "Periodo", "archivo", "bytes", "estado"])
        for k, fila in enumerate(pendientes.itertuples(), 1):
            ruta = DESTINO / str(fila.Periodo) / f"{fila.CresCodigoFormato}.pdf"
            ruta.parent.mkdir(exist_ok=True)
            estado, tam = "", ""
            for intento in range(4):
                try:
                    r = sesion.get(fila.ResumenEjecutivo.replace("http://", "https://"), timeout=60)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(str(r.status_code))
                    r.raise_for_status()
                    if not r.content.startswith(b"%PDF"):
                        estado = "no_es_pdf"
                        break
                    ruta.write_bytes(r.content)
                    estado, tam = "ok", len(r.content)
                    pausa = max(PAUSA, pausa * 0.9)        # vuelve de a pocos al ritmo normal
                    break
                except Exception as e:      # noqa: BLE001 - se anota y se sigue
                    estado = f"error: {e}"[:80]
                    pausa = min(5.0, pausa * 2)             # el servidor se queja: más lento
                    time.sleep(5 * (intento + 1))
            w.writerow([fila.CresCodigoFormato, fila.Periodo, ruta.name if estado == "ok" else "", tam, estado])
            f.flush()
            if k % 500 == 0:
                print(f"{k}/{len(pendientes)} en {(time.time() - t0) / 60:.0f} min", flush=True)
            time.sleep(pausa)
    print(f"Listo en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int)
    main(ap.parse_args().limite)
