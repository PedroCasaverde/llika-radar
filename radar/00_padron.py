"""Paso 0: cuántas personas con RUC 10 tienen cada nombre completo y cada par de apellidos, por departamento.

Sale del padrón reducido de SUNAT (todas las personas naturales con RUC). Sirve para dos cosas:
- saber si un nombre completo es único en el país, para no confundir homónimos;
- medir qué tan común es un par de apellidos en el departamento, que es la base del cálculo bayesiano de parentesco.
Guarda solo conteos, sin nombres sueltos de personas. Solo imprime agregados.

    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 00_padron.py
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA, norm_serie, ultimo  # noqa: E402

# código de departamento del ubigeo (INEI) -> nombre, como viene en el maestro de entidades
DEPTO = {"01": "AMAZONAS", "02": "ANCASH", "03": "APURIMAC", "04": "AREQUIPA", "05": "AYACUCHO", "06": "CAJAMARCA",
         "07": "CALLAO", "08": "CUSCO", "09": "HUANCAVELICA", "10": "HUANUCO", "11": "ICA", "12": "JUNIN",
         "13": "LA LIBERTAD", "14": "LAMBAYEQUE", "15": "LIMA", "16": "LORETO", "17": "MADRE DE DIOS", "18": "MOQUEGUA",
         "19": "PASCO", "20": "PIURA", "21": "PUNO", "22": "SAN MARTIN", "23": "TACNA", "24": "TUMBES", "25": "UCAYALI"}

SALIDA.mkdir(parents=True, exist_ok=True)
partes = []
with zipfile.ZipFile(ultimo("acc6_padron_sunat", "*.zip")) as z, z.open("padron_reducido_ruc.txt") as fh:
    for t in pd.read_csv(fh, sep="|", dtype=str, encoding="latin-1", usecols=[0, 1, 4], chunksize=2_000_000,
                         on_bad_lines="skip"):
        t.columns = ["ruc", "nombre", "ubigeo"]
        t = t[t["ruc"].str.startswith("10", na=False)]
        partes.append(pd.DataFrame({"nombre": norm_serie(t["nombre"].fillna("")),
                                    "depto": t["ubigeo"].fillna("").str[:2].map(DEPTO).fillna("SIN DEPTO")}))
p = pd.concat(partes, ignore_index=True)
p = p[p["nombre"].ne("")]
print(f"personas con RUC 10 en el padrón: {len(p):,}")

nombres = p.groupby("nombre").size().rename("n").reset_index()
nombres.to_parquet(SALIDA / "padron_nombres.parquet", index=False)
print(f"nombres completos distintos: {len(nombres):,}; únicos en el país: {(nombres['n'] == 1).mean():.1%}")

filas = []
tokens = p["nombre"].str.split()
for k in (2, 3, 4):  # apellidos compuestos («DE LA CRUZ») ocupan más de una palabra
    ok = tokens.str.len() > k
    filas.append(pd.DataFrame({"depto": p.loc[ok, "depto"], "par": tokens[ok].str[:k].str.join(" ")}))
pares = pd.concat(filas).groupby(["depto", "par"]).size().rename("n").reset_index()
total = p.groupby("depto").size().rename("personas").reset_index()
pares.to_parquet(SALIDA / "padron_pares.parquet", index=False)
total.to_parquet(SALIDA / "padron_deptos.parquet", index=False)
print(f"pares de apellidos por departamento: {len(pares):,}; personas por departamento: "
      f"{dict(zip(total['depto'], total['personas']))}")
