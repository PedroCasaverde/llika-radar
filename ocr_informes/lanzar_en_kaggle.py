"""Lanza en TU cuenta de Kaggle los notebooks de lectura de informes (ocr_lote_NN.ipynb) que están en esta carpeta, con
GPU e internet, en modo «Save & Run All»: corren en los servidores de Kaggle y la computadora se puede apagar.

Requisitos (una sola vez):
  pip install kaggle
  Tu token de Kaggle: kaggle.com > Settings > API > Generate New Token. Guárdalo en la variable de entorno
  KAGGLE_API_TOKEN (o en ~/.kaggle/). Es tuyo: no lo compartas con nadie.
  La cuenta tiene que estar verificada con el celular (sin eso Kaggle no da internet al notebook).

Uso:
  python lanzar_en_kaggle.py lanzar 07          lanza ocr_lote_07.ipynb
  python lanzar_en_kaggle.py lanzar 07 08 09    lanza varios (si Kaggle no deja tantos a la vez, avisa)
  python lanzar_en_kaggle.py lanzar 20 --cpu    lanza sin tarjeta gráfica (no gasta la cuota de GPU; más lento)
  python lanzar_en_kaggle.py estado             estado de todos tus lotes lanzados
  python lanzar_en_kaggle.py cuota              horas de GPU usadas y cuándo se renuevan
  python lanzar_en_kaggle.py bajar 07           baja el resultado (textos_lote_07.zip) a ./resultados/

Después de lanzar, comparte cada notebook con pedrocasaverdeayma desde la web (Share > Add collaborators > Can view):
la API de Kaggle no permite compartir.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).parent


def kaggle(*args):
    # Con el mismo Python que corre este script (donde se instaló kaggle), para no depender del PATH de Windows.
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return (r.stdout or "") + (r.stderr or "")


def usuario():
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m:
        sys.exit("No encuentro tu usuario de Kaggle: revisa tu token (KAGGLE_API_TOKEN o ~/.kaggle/).")
    return m.group(1)


def lanzar(numeros, cpu=False):
    u = usuario()
    for n in numeros:
        nb = AQUI / f"ocr_lote_{n}.ipynb"
        if not nb.exists():
            print(f"No está {nb.name} en esta carpeta.")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(nb, Path(tmp) / nb.name)
            (Path(tmp) / "kernel-metadata.json").write_text(json.dumps({
                "id": f"{u}/llika-ocr-lote-{n}", "title": f"llika-ocr-lote-{n}", "code_file": nb.name,
                "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": not cpu,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}),
                encoding="utf-8")
            # GPU T4 x2: el notebook usa las dos tarjetas a la vez (la cuota cuenta horas de sesión, no de tarjeta)
            salida = kaggle("kernels", "push", "-p", tmp, *([] if cpu else ["--accelerator", "NvidiaTeslaT4"]))
        print(f"lote {n}: {salida.strip().splitlines()[-1] if salida.strip() else 'sin respuesta'}")
    print("\nAhora comparte cada notebook con pedrocasaverdeayma (Share > Add collaborators > Can view).")


def estado():
    u = usuario()
    for nb in sorted(AQUI.glob("ocr_lote_*.ipynb")):
        n = nb.stem.split("_")[-1]
        s = kaggle("kernels", "status", f"{u}/llika-ocr-lote-{n}").strip()
        if "not found" in s.lower() or "404" in s:
            continue
        print(f"lote {n}: {s.splitlines()[-1] if s else '?'}")


def bajar(numeros):
    u = usuario()
    for n in numeros:
        destino = AQUI / "resultados" / f"lote_{n}"
        destino.mkdir(parents=True, exist_ok=True)
        # Solo el zip, los resúmenes y los registros: los textos sueltos ya vienen dentro del zip y bajarlos uno por
        # uno toma mucho más
        r = kaggle("kernels", "output", f"{u}/llika-ocr-lote-{n}", "-p", str(destino), "--file-pattern",
                   r"\.(zip|csv|log)$")
        print(f"lote {n}: {r.strip()[-200:]}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("lanzar", "estado", "bajar", "cuota"):
        sys.exit(__doc__)
    accion = sys.argv[1]
    cpu = "--cpu" in sys.argv
    nums = [x.zfill(2) for x in sys.argv[2:] if not x.startswith("--")]
    {"lanzar": lambda: lanzar(nums, cpu), "estado": estado, "bajar": lambda: bajar(nums),
     "cuota": lambda: print(kaggle("quota"))}[accion]()
