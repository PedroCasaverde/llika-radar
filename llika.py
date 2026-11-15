"""Punto de entrada de LLIKA: corre cada paso, revisa lo que necesita y registra su tiempo y su memoria.

    python llika.py revisar                     entorno, memoria, disco, datos y huellas: qué falta y qué paso lo arma
    python llika.py demo                        arma y prueba la pantalla con datos inventados (no necesita datos)
    python llika.py puntajes                    opcional: rehace el puntaje de todos los casos de 2025 y comprueba la lista congelada
    python llika.py entidad RUC [RUC ...]       los datos reales de esas entidades para la pantalla (en privado/)
    python llika.py pantalla privado/datos_X.json         la pantalla con esos datos (en privado/)
    python llika.py probar privado/llika_X.html privado/datos_X.json [--sin-red]
    python llika.py caso-real                   entidad, pantalla y prueba del caso del video: Ripán, con Pachas y Marías

Cada corrida deja una fila en resultados/corridas.jsonl (paso, segundos, memoria pico, código de salida), sin datos de
personas. El detalle de cada paso está en el README.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
V4, FUENTE, PRIVADO = RAIZ / "radar" / "v4", RAIZ / "demo" / "fuente", RAIZ / "privado"
LISTA = RAIZ / "datos/procesados/v4/resultados/lista_unica_limpia/lista_2025_unica.parquet"
HUELLA_LISTA = "189af867041d1270bf132ab6ca25041e9a3e782713171758dadf22a761a4aff2"
CASO_REAL = ["20144801459", "20194119403", "20168795093"]          # Ripán, Pachas y Marías (Dos de Mayo, Huánuco)
REGISTRO = RAIZ / "resultados" / "corridas.jsonl"


def correr(paso, script, args=(), cwd=None):
    """Corre un script con el mismo Python, aislado (-I) y en UTF-8; mide tiempo y memoria pico."""
    try:
        import psutil
    except ImportError:
        psutil = None
    cmd = [sys.executable, "-I", "-X", "utf8", str(script), *map(str, args)]
    print(f"\n== {paso}: {' '.join(cmd[3:])}", flush=True)
    t0, pico = time.time(), 0.0
    p = subprocess.Popen(cmd, cwd=cwd or Path(script).parent)
    while p.poll() is None:
        if psutil:
            try:
                q = psutil.Process(p.pid)
                pico = max(pico, sum(x.memory_info().rss for x in [q] + q.children(recursive=True)) / 1e9)
            except psutil.Error:
                pass
        time.sleep(0.5)
    fila = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "paso": paso, "segundos": round(time.time() - t0, 1), "memoria_pico_gb": round(pico, 2),
            "codigo": p.returncode, "equipo": f"{platform.system()} {platform.machine()}, {os.cpu_count()} núcleos"}
    REGISTRO.parent.mkdir(exist_ok=True)
    with REGISTRO.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"== {paso}: {'bien' if p.returncode == 0 else 'FALLÓ (código ' + str(p.returncode) + ')'} en {fila['segundos']} s, memoria pico {fila['memoria_pico_gb']} GB", flush=True)
    if p.returncode != 0:
        sys.exit(p.returncode)
    return fila


def revisar():
    import hashlib
    ok = True
    print(f"Python {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] < (3, 12):
        print("  ✗ hace falta Python 3.12 o más nuevo"); ok = False
    faltan = []
    for m in ("pandas", "pyarrow", "polars", "numpy", "scipy", "sklearn", "lightgbm", "psutil", "playwright"):
        try:
            __import__(m)
        except ImportError:
            faltan.append(m)
    print("  ✓ librerías" if not faltan else f"  ✗ faltan librerías: {', '.join(faltan)} (pip install -r requirements.txt)")
    ok &= not faltan
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        print(f"  {'✓' if ram >= 7 else '!'} memoria: {ram:.0f} GB (una entidad real pide 4,4 GB; rehacer los puntajes, opcional, 21 GB)")
    except ImportError:
        pass
    libre = shutil.disk_usage(RAIZ).free / 1e9
    print(f"  {'✓' if libre >= 30 else '!'} disco libre: {libre:.0f} GB (la cadena completa pide unos 30 GB)")
    print("\nDatos (no van a git; se arman con la cadena o los entrega el equipo):")
    for ruta, que in ((RAIZ / "datos/procesados/v4/base/ordenes.parquet", "base de órdenes"), (LISTA, "lista congelada de 2025"),
                      (LISTA.with_name("puntajes_2025.parquet"), "puntajes de 2025 (python llika.py puntajes)")):
        print(f"  {'✓' if ruta.exists() else '·'} {que}: {ruta.relative_to(RAIZ)}")
    if LISTA.exists():
        h = hashlib.sha256(LISTA.read_bytes()).hexdigest()
        print(f"  {'✓' if h == HUELLA_LISTA else '✗'} huella de la lista: {h[:16]} (debe ser {HUELLA_LISTA[:16]})")
        ok &= h == HUELLA_LISTA
    print("\nSin datos se puede correr: python llika.py demo")
    return ok


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "ayuda"):
        print(__doc__)
        return
    paso, resto = sys.argv[1], sys.argv[2:]
    if paso == "revisar":
        sys.exit(0 if revisar() else 1)
    elif paso == "demo":
        correr("demo: armar", FUENTE / "armar_demo.py")
        correr("demo: probar", FUENTE / "probar_demo.py", [RAIZ / "demo" / "llika_pe_demo.html", PRIVADO / "capturas_demo"])
    elif paso == "puntajes":
        correr("puntajes de 2025", V4 / "puntajes_2025.py")
    elif paso == "entidad":
        if not resto:
            sys.exit("ERROR: falta al menos un RUC de entidad. Ejemplo: python llika.py entidad 20144801459")
        correr("entidad", V4 / "exportar_entidad.py", ["--entidades", *resto])
    elif paso == "pantalla":
        if len(resto) != 1:
            sys.exit("ERROR: falta el JSON de la entidad. Ejemplo: python llika.py pantalla privado/datos_ripan_2025.json")
        correr("pantalla", FUENTE / "armar_demo.py", ["--real", Path(resto[0]).resolve()])
    elif paso == "probar":
        if len(resto) < 2:
            sys.exit("ERROR: faltan la pantalla y su JSON. Ejemplo: python llika.py probar privado/llika_ripan_2025.html privado/datos_ripan_2025.json")
        correr("probar", FUENTE / "probar_real.py", [Path(resto[0]).resolve(), Path(resto[1]).resolve(), PRIVADO / "capturas", *resto[2:]])
    elif paso == "caso-real":
        correr("entidad", V4 / "exportar_entidad.py", ["--entidades", *CASO_REAL, "--salida", PRIVADO / "datos_caso_real_2025.json"])
        correr("pantalla", FUENTE / "armar_demo.py", ["--real", PRIVADO / "datos_caso_real_2025.json", PRIVADO / "llika_caso_real_2025.html"])
        correr("probar", FUENTE / "probar_real.py", [PRIVADO / "llika_caso_real_2025.html", PRIVADO / "datos_caso_real_2025.json", PRIVADO / "capturas", *resto])
    else:
        sys.exit(f"ERROR: no conozco el paso «{paso}». Corre python llika.py ayuda para ver los pasos.")


if __name__ == "__main__":
    main()
