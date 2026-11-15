"""Última ronda (HALLAZGOS 160, sección 46): descarga del portal público del OECE y paso a parquet. Se reanuda: lo ya bajado con el mismo tamaño no se repite.

    python -I -X utf8 u1_portal.py bajar
    python -I -X utf8 u1_portal.py columnas   # solo nombres de columnas y filas por archivo
    python -I -X utf8 u1_portal.py parquet
Salida: datos/crudos/oece_portal/<tabla>/*.xlsx, datos/crudos/oece_portal/manifiesto.csv y datos/procesados/v4/portal/<tabla>.parquet
"""
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import requests

R = Path(__file__).resolve().parents[2]
DIR = R / "datos" / "crudos" / "oece_portal"
SAL = R / "datos" / "procesados" / "v4" / "portal"
BASE = "https://conosce.osce.gob.pe/buscador/assets/67ae6c4a/reportes"
PARAM = {"userid": "public", "password": "key"}        # los que el propio portal publica en su código
TABLAS = {"arbitraje": ("arbitraje", "ARBITRAJE"), "ampliaciones": ("ampliaciones", "AMPLIACIONES"), "consorcios": ("consorcios", "CONSORCIO"),
          "miembroscomite": ("miembroscomite", "MIEMBROCOMITE"), "contratos": ("contratos", "CONTRATOS"), "convocatorias": ("convocatorias", "CONVOCATORIAS")}
ANIOS = range(2018, 2026)
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def bajar():
    filas = []
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (LLIKA, proyecto academico)"
    for tabla, (carpeta, pref) in TABLAS.items():
        (DIR / tabla).mkdir(parents=True, exist_ok=True)
        for y in ANIOS:
            url, f = f"{BASE}/{carpeta}/{y}/CONOSCE_{pref}{y}_0.xlsx", DIR / tabla / f"{pref}{y}.xlsx"
            h = s.head(url, params=PARAM, timeout=60)
            if h.status_code != 200:
                paso(f"ERROR {tabla} {y}: estado {h.status_code}")
                filas.append({"tabla": tabla, "anio": y, "estado": h.status_code})
                continue
            tam = int(h.headers.get("content-length", 0))
            if not (f.exists() and f.stat().st_size == tam):
                for intento in range(4):
                    try:
                        with s.get(url, params=PARAM, stream=True, timeout=120) as r:
                            r.raise_for_status()
                            with open(f.with_suffix(".tmp"), "wb") as w:
                                for b in r.iter_content(1 << 20):
                                    w.write(b)
                        f.with_suffix(".tmp").replace(f)
                        break
                    except Exception as e:      # noqa: BLE001
                        paso(f"reintento {intento + 1} {tabla} {y}: {type(e).__name__}")
                        time.sleep(20 * (intento + 1))
                else:
                    paso(f"ERROR {tabla} {y}: no bajó")
                    continue
                time.sleep(1)
            filas.append({"tabla": tabla, "anio": y, "estado": 200, "bytes": f.stat().st_size, "servidor": h.headers.get("last-modified"), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bajado": time.strftime("%Y-%m-%d")})
            paso(f"{tabla} {y}: {f.stat().st_size / 1e6:.1f} MB")
    pd.DataFrame(filas).to_csv(DIR / "manifiesto.csv", index=False)
    paso("BAJADA LISTA")


def leer(f, **k):
    return pd.read_excel(f, dtype=str, engine="calamine", **k)


def columnas():
    for tabla in TABLAS:
        vistos = {}
        for f in sorted((DIR / tabla).glob("*.xlsx")):
            x = pd.ExcelFile(f, engine="calamine")
            d = x.parse(x.sheet_names[0], dtype=str, nrows=3)
            vistos.setdefault(tuple(d.columns), []).append(f.stem)
            print(tabla, f.stem, "hojas", x.sheet_names[:4], flush=True)
        for c, v in vistos.items():
            print("  ", v, list(c), flush=True)


def parquet():
    SAL.mkdir(parents=True, exist_ok=True)
    for tabla in TABLAS:
        partes = []
        for f in sorted((DIR / tabla).glob("*.xlsx")):
            d = leer(f)
            d.columns = [str(c).strip().upper() for c in d.columns]
            partes.append(d.assign(ARCHIVO_ANIO=int(f.stem[-4:])))
            paso(f"{f.stem}: {len(d):,} filas")
        d = pd.concat(partes, ignore_index=True)
        d.to_parquet(SAL / f"{tabla}.parquet", index=False)
        paso(f"{tabla}: {len(d):,} filas, {d.shape[1]} columnas")
    paso("PARQUET LISTO")


if __name__ == "__main__":
    {"bajar": bajar, "columnas": columnas, "parquet": parquet}[sys.argv[1]]()
