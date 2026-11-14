# Capturas de los puntos que tocó la prueba de uso, para comparar antes y después del cambio.
#   python capturas_prueba.py archivo.html carpeta prefijo
# Sirve para las dos versiones: se ubica por los títulos de la ficha, que no cambiaron.
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ARCHIVO, CAP, PRE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
CAP.mkdir(parents=True, exist_ok=True)
U = ARCHIVO.as_uri()
# (nombre, ruta, a qué parte de la ficha se baja; None es el inicio)
VISTAS = [("1_parecido", "#caso-wayrapampa-3", "p-calc"), ("2_lista", "#entidad-wayrapampa", None),
          ("3_compras", "#caso-wayrapampa-5", "p-ord"), ("4_aviso", "#caso-wayrapampa-1", None),
          ("4_coincidencia", "#caso-wayrapampa-2", "p-sen"), ("6_declarado", "#caso-wayrapampa-18", "p-sen")]

with sync_playwright() as pw:
    nav = pw.chromium.launch()
    ctx = nav.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="light")
    p = ctx.new_page()
    for nombre, ruta, ancla in VISTAS:
        p.goto(U + ruta, wait_until="networkidle")
        p.wait_for_timeout(450)
        if ruta.startswith("#entidad"):
            p.screenshot(path=str(CAP / f"{PRE}_{nombre}.png"), clip={"x": 0, "y": 0, "width": 1440, "height": 900})
            continue
        if ancla:
            p.evaluate(f"(() => {{ const c = document.getElementById('panel-cuerpo'), h = document.getElementById('{ancla}'); c.scrollTop = h.offsetTop - c.offsetTop - 16; }})()")
            p.wait_for_timeout(200)
        p.locator("#panel").screenshot(path=str(CAP / f"{PRE}_{nombre}.png"))
    ctx.close()
    nav.close()
print("capturas en:", CAP)
