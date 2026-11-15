"""Arma los notebooks de lectura de informes completos (HALLAZGOS 65 y 66), uno por lote, con el lote ADENTRO: quien lo
corre solo lo importa en Kaggle y presiona Save & Run All, sin subir ningún archivo.

Uso:  python -X utf8 armar_notebook.py [--tam 970]
- Orden: si existe prioridad_informes.csv (CresCodigoFormato, prioridad; menor = antes), que sale de clasificar las
  fichas resumen, los lotes 01, 02… llevan primero los informes de fondo. Si no existe, van del más reciente al más
  antiguo.
- Excluye los informes que ya están bajados en esta computadora (los lee ocr_local.py).
- Escribe lotes/lote_NN.csv (código, número, año y enlace: sin datos de personas) y notebooks/ocr_lote_NN.ipynb.
"""
import argparse
import base64
import gzip
from pathlib import Path

import nbformat as nbf
import pandas as pd

AQUI = Path(__file__).parent
ACC8 = AQUI.parent / "datos" / "crudos" / "acc8_informes_cgr"

MD_INICIO = """# Lectura de informes de control de la Contraloría (LLIKA)

Este notebook lee los informes de control de **tu lote** y devuelve un archivo de texto por informe. Para cada informe:

1. baja el PDF desde el portal de la Contraloría;
2. si la página trae texto digital, lo copia tal cual (es gratis e instantáneo);
3. si la página es escaneada, la lee con OCR (RapidOCR, con los modelos PP-OCRv6 de PaddleOCR), en la tarjeta gráfica si
   hay una;
4. guarda el texto y borra el PDF, para no llenar el disco.

**Qué te pasan:** solo este notebook; la lista de informes de tu lote ya viene adentro.
**Qué devuelves:** nada a mano: compartes el notebook con `pedrocasaverdeayma` y él baja el resultado
(`textos_lote_NN.zip`).

Si se corta: en tu computadora o en Colab, vuelve a correr todo y sigue donde se quedó. En Kaggle cada versión empieza de cero, así que avisa a Pedro para armar un lote con lo que faltó."""

MD_KAGGLE = """## Antes de empezar

**En Kaggle (recomendado: tarjeta gráfica gratis unas 30 horas por semana)**
1. Verifica tu cuenta con el celular (*Settings → Phone Verification*); sin eso no hay internet en el notebook.
2. *Create → New Notebook → File → Import Notebook* y sube este archivo.
3. Panel derecho, *Session options*: **Accelerator: GPU T4 x2** (o P100) e **Internet: On**.
4. **Save Version → Save & Run All** (no lo corras celda por celda: así sigue aunque cierres el navegador).
5. *Share* → agrega a `pedrocasaverdeayma`. Él ve el avance y baja el resultado.

**En Google Colab:** *Entorno de ejecución → Cambiar tipo → T4 GPU* y corre todo.

**En tu computadora:** Python 3.10 o más; la primera celda instala lo que falta."""

CODIGO_CONFIG = """# === Lo único que se cambia ===
LIMITE = None        # para probar: un número (p. ej. 5) lee solo esos informes
DPI = 200            # resolución para el OCR (300 no mejoró en la prueba)
PAUSA = 1.0          # segundos entre descargas, para no cargar el servidor de la Contraloría (con muchas sesiones a la
                     # vez, a alguna dejó de contestarle el 31/10)
HORAS_MAX = 11       # deja de tomar informes nuevos a las 11 horas (Kaggle corta a las 12) y guarda lo leído"""

CODIGO_INSTALAR = """import importlib.util, subprocess, sys

def instalar(*paquetes):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *paquetes], check=True)

if importlib.util.find_spec("rapidocr") is None:
    instalar("rapidocr==3.9.2")
if importlib.util.find_spec("fitz") is None:
    instalar("pymupdf")

try:
    import torch
    N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
except ImportError:
    N_GPU = 0
if N_GPU == 0 and importlib.util.find_spec("onnxruntime") is None:
    instalar("onnxruntime==1.31.0")
if N_GPU:
    print(f"Tarjetas gráficas: {N_GPU} ({torch.cuda.get_device_name(0)}): un trabajador por tarjeta")
else:
    print("Sin tarjeta gráfica: dos trabajadores en el procesador (más lento)")"""

CODIGO_PREPARAR = """import base64, gzip, io, shutil, time
from pathlib import Path

import pandas as pd

EN_KAGGLE = Path("/kaggle/working").exists()
TRABAJO = Path("/kaggle/working") if EN_KAGGLE else Path.cwd()
lote = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(LOTE_DATOS))), dtype=str)
ARCHIVO_LOTE = TRABAJO / f"{NOMBRE}.csv"
lote.to_csv(ARCHIVO_LOTE, index=False)
SALIDA = TRABAJO / f"textos_{NOMBRE}"
SALIDA.mkdir(exist_ok=True)
(TRABAJO / "trabajador_ocr.py").write_text(base64.b64decode(TRABAJADOR).decode("utf-8"), encoding="utf-8")
print(f"{NOMBRE}: {len(lote)} informes; salida en {SALIDA}")"""

CODIGO_CORRER = """# Un trabajador por tarjeta gráfica (en T4 x2, dos a la vez: la cuota de Kaggle cuenta horas de sesión, no de tarjeta)
# o dos en el procesador. Cada uno toma informes alternados del lote y escribe su propio resumen.
# Antes, los modelos del OCR se bajan una sola vez: si los dos trabajadores los bajan a la vez, uno lee un archivo a medio
# escribir y muere al empezar, y el lote sale con la mitad (pasó en los lotes 01, 02 y 07c el 31/10).
T_INICIO = time.time()
dispositivos = [f"cuda:{k}" for k in range(N_GPU)] or ["cpu", "cpu"]
subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(TRABAJO)!r}); "
                f"from trabajador_ocr import motor_ocr; motor_ocr({dispositivos[0]!r}, 2)"], check=True)
procesos = []
for k, disp in enumerate(dispositivos):
    cmd = [sys.executable, str(TRABAJO / "trabajador_ocr.py"), "--lote", str(ARCHIVO_LOTE), "--salida", str(SALIDA),
           "--parte", str(k), "--de", str(len(dispositivos)), "--dispositivo", disp, "--hilos", "2",
           "--dpi", str(DPI), "--pausa", str(PAUSA), "--limite", str(LIMITE or 0),
           "--hasta", str(T_INICIO + HORAS_MAX * 3600)]
    log = open(TRABAJO / f"trabajador_{k}.log", "w")
    procesos.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))

total = min(len(lote), LIMITE) if LIMITE else len(lote)
t0 = time.time()
while any(p.poll() is None for p, _ in procesos):
    time.sleep(60)
    hechos = sum(max(0, sum(1 for _ in open(r, encoding="utf-8")) - 1) for r in SALIDA.glob("resumen_*.csv"))
    m = (time.time() - t0) / 60
    print(f"{hechos}/{total} informes en {m:.0f} min" + (f" (quedan ~{m / hechos * (total - hechos):.0f} min)"
                                                        if hechos else ""), flush=True)
for k, (p, log) in enumerate(procesos):
    log.close()
    if p.returncode != 0:
        print(f"El trabajador {k} terminó con error. Últimas líneas de su registro:")
        print("".join(open(TRABAJO / f"trabajador_{k}.log", encoding="utf-8").readlines()[-15:]))"""

CODIGO_ZIP = """partes = [pd.read_csv(r) for r in sorted(SALIDA.glob("resumen_*.csv"))]
r = pd.concat(partes, ignore_index=True)
r.to_csv(SALIDA / "resumen.csv", index=False)
print(f"Leídos {len(r)} de {total} informes del lote" +
      ("" if len(r) >= total else f". FALTAN {total - len(r)}: avisa a Pedro para relanzarlos"))
print(r.estado.value_counts().to_string())
ok = r[r.estado == "ok"]
print(f"Páginas: {int(ok.paginas.sum())}; con OCR: {int(ok.paginas_ocr.sum())}; "
      f"confianza media del OCR: {ok.confianza_ocr.mean():.3f}; segundos por página con OCR: "
      f"{ok.segundos.sum() / max(1, ok.paginas_ocr.sum()):.2f}")
zip_final = shutil.make_archive(str(TRABAJO / f"textos_{NOMBRE}"), "zip", SALIDA)
print("Resultado:", zip_final)"""


def notebook(nombre, datos_csv):
    """Un notebook por lote, con la lista de informes comprimida adentro."""
    datos = base64.b64encode(gzip.compress(datos_csv.encode("utf-8"))).decode("ascii")
    trabajador = base64.b64encode((AQUI / "trabajador_ocr.py").read_bytes()).decode("ascii")
    celda_lote = f'NOMBRE = "{nombre}"\nLOTE_DATOS = "{datos}"\nTRABAJADOR = "{trabajador}"'
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(MD_INICIO),
        nbf.v4.new_markdown_cell(MD_KAGGLE),
        nbf.v4.new_markdown_cell("## 1. Configuración"),
        nbf.v4.new_code_cell(CODIGO_CONFIG),
        nbf.v4.new_markdown_cell(f"Tu lote: **{nombre}** (la lista va en la celda siguiente; no la toques)."),
        nbf.v4.new_code_cell(celda_lote),
        nbf.v4.new_markdown_cell("## 2. Instalar lo que falta (en Kaggle y Colab casi todo ya viene)"),
        nbf.v4.new_code_cell(CODIGO_INSTALAR),
        nbf.v4.new_markdown_cell("## 3. Preparar el lote y el lector"),
        nbf.v4.new_code_cell(CODIGO_PREPARAR),
        nbf.v4.new_markdown_cell("## 4. Leer el lote\nMuestra el avance cada minuto. En tu computadora o en Colab, si "
                                 "se corta, vuelve a correr todo: salta los que ya están."),
        nbf.v4.new_code_cell(CODIGO_CORRER),
        nbf.v4.new_markdown_cell("## 5. Resumen y zip"),
        nbf.v4.new_code_cell(CODIGO_ZIP),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (AQUI / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, AQUI / "notebooks" / f"ocr_{nombre}.ipynb")


def lotes(tam):
    lista = pd.read_csv(ACC8 / "listas" / "lista_unica_informes_ampliada.csv", dtype=str)
    lista = lista[lista.RutaCloudInforme.notna()]
    bajados = set(pd.read_csv(ACC8 / "inventario_pdf.csv", dtype=str).CresCodigoFormato)
    resto = lista[~lista.CresCodigoFormato.isin(bajados)][["CresCodigoFormato", "NumeroInforme", "Periodo",
                                                           "RutaCloudInforme"]].copy()
    prio = AQUI / "prioridad_informes.csv"
    if prio.exists():
        p = pd.read_csv(prio, dtype={"CresCodigoFormato": str})
        p["orden"] = range(len(p))                     # el orden del archivo ya trae prioridad y relevancia
        resto = resto.merge(p[["CresCodigoFormato", "prioridad", "orden"]], on="CresCodigoFormato", how="left")
        resto["prioridad"] = resto.prioridad.fillna(resto.prioridad.max() + 1)
        resto = resto.sort_values(["orden", "Periodo"], ascending=[True, False], na_position="last").drop(columns="orden")
        print("Orden: prioridad de las fichas (fondo primero)")
    else:
        resto = resto.sort_values("Periodo", ascending=False)
        print("Orden: del más reciente al más antiguo (aún no hay prioridad_informes.csv)")
    (AQUI / "lotes").mkdir(exist_ok=True)
    for viejo in (AQUI / "lotes").glob("lote_*.csv"):
        viejo.unlink()
    for viejo in (AQUI / "notebooks").glob("ocr_lote_*.ipynb") if (AQUI / "notebooks").exists() else []:
        viejo.unlink()
    # Lotes por páginas estimadas, para que cada uno quepa holgado en una sesión de Kaggle (máximo 12 horas). Páginas por
    # informe medidas en los 1 274 bajados: prioridad 1, ~58; prioridad 2, ~29; prioridad 3, ~16; 4 (rutina), ~14; 5, ~18.
    # Con las dos T4 se leen ~0,9 s por página (prueba del 31/10): 18 000 páginas son ~4,5 h; si alguien elige una sola
    # tarjeta (P100), ~9 h, todavía dentro del límite. Los de fondo y los de rutina no se mezclan en un mismo lote.
    PAGINAS = {1: 58, 2: 29, 3: 16, 4: 14.3, 5: 18}
    TOPE_PAGINAS = 18_000
    if "prioridad" not in resto:
        resto["prioridad"] = 9
    partes, actual, pags, grupo_actual = [], [], 0.0, None
    for fila in resto.itertuples(index=False):
        grupo = "fondo" if fila.prioridad <= 3 else "rutina"
        est = PAGINAS.get(int(fila.prioridad) if fila.prioridad == fila.prioridad else 5, 18)
        if actual and (pags + est > TOPE_PAGINAS or grupo != grupo_actual):
            partes.append(actual)
            actual, pags = [], 0.0
        actual.append(fila)
        pags += est
        grupo_actual = grupo
    if actual:
        partes.append(actual)
    n = 0
    for n, filas in enumerate(partes, 1):
        parte = pd.DataFrame(filas).drop(columns="prioridad")
        nombre = f"lote_{n:02d}"
        parte.to_csv(AQUI / "lotes" / f"{nombre}.csv", index=False)
        notebook(nombre, parte.to_csv(index=False))
    resto = resto.drop(columns="prioridad")
    prueba = resto.groupby("Periodo").head(1).head(5)
    notebook("lote_prueba", prueba.to_csv(index=False))
    print(f"{len(resto)} informes en {n} lotes de hasta {TOPE_PAGINAS} páginas estimadas; notebooks en notebooks/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tam", type=int, default=970)
    lotes(ap.parse_args().tam)
