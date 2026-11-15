# Capturas de la etapa 2 del cierre (color y marca, y lo que faltaba del contrato de alcance), para comparar antes y después.
#   python capturas_etapa2.py archivo.html carpeta prefijo
# Con la versión anterior (sin decisión ni índices) saca solo las vistas que existen en las dos.
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ARCHIVO, CAP, PRE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
CAP.mkdir(parents=True, exist_ok=True)
U = ARCHIVO.as_uri()


def bajar_a(p, ancla):
    p.evaluate(f"(() => {{ const c = document.getElementById('panel-cuerpo'), h = document.getElementById('{ancla}'); c.scrollTop = h.offsetTop - c.offsetTop - 16; }})()")
    p.wait_for_timeout(200)


with sync_playwright() as pw:
    nav = pw.chromium.launch()
    for disp, w, h, tema in (("pc", 1440, 900, "light"), ("oscuro", 1440, 900, "dark"), ("cel", 390, 844, "light")):
        ctx = nav.new_context(viewport={"width": w, "height": h}, device_scale_factor=2 if disp == "cel" else 1, color_scheme=tema, has_touch=(disp == "cel"))
        p = ctx.new_page()
        nueva = None
        vistas = {"pc": [("1_inicio", "#inicio", None), ("2_entidad", "#entidad-wayrapampa", None), ("3_ficha", "#caso-wayrapampa-1", None),
                         ("4_ficha_senales", "#caso-wayrapampa-5", "p-sen"), ("8_como_funciona", "#como-funciona", None), ("9_datos_y_fuentes", "#datos", "pagina")],
                  "oscuro": [("1_inicio", "#inicio", None), ("2_entidad", "#entidad-wayrapampa", None), ("3_ficha", "#caso-wayrapampa-1", None)],
                  "cel": [("1_inicio", "#inicio", None), ("2_entidad", "#entidad-wayrapampa", None), ("3_ficha", "#caso-wayrapampa-1", None)]}[disp]
        for nombre, ruta, extra in vistas:
            p.goto(U + ruta, wait_until="networkidle")
            p.wait_for_timeout(500)
            if nueva is None:
                nueva = p.locator("#panel-pie").count() > 0
            if extra == "pagina":
                p.screenshot(path=str(CAP / f"{PRE}_{disp}_{nombre}.png"), full_page=True)
                continue
            if extra:
                bajar_a(p, extra)
            p.screenshot(path=str(CAP / f"{PRE}_{disp}_{nombre}.png"))
        if disp == "pc" and nueva:
            # Solo en la versión nueva: los índices, decidir con su motivo y la lista con la decisión
            p.goto(U + "#inicio", wait_until="networkidle")
            p.wait_for_timeout(400)
            p.locator("#t-ind").scroll_into_view_if_needed()
            caja = p.locator("#t-ind").bounding_box()
            p.screenshot(path=str(CAP / f"{PRE}_{disp}_5_indices.png"), clip={"x": 0, "y": caja["y"] - 24 + p.evaluate("scrollY"), "width": w, "height": 720}, full_page=True)
            p.goto(U + "#caso-wayrapampa-5", wait_until="networkidle")
            p.wait_for_timeout(400)
            bajar_a(p, "p-sen")
            p.click('#panel-pie [data-dec="confirmar"]')
            p.click('[data-motivo="Las compras merecen revisarse"]')
            p.wait_for_timeout(150)
            p.screenshot(path=str(CAP / f"{PRE}_{disp}_6a_decidir.png"))
            p.click('#form-dec button[type="submit"]')
            p.wait_for_timeout(250)
            p.screenshot(path=str(CAP / f"{PRE}_{disp}_6b_decidido.png"))
            p.goto(U + "#caso-wayrapampa-2", wait_until="networkidle")
            p.wait_for_timeout(300)
            p.click('#panel-pie [data-dec="descartar"]')
            p.click('[data-motivo="Es un homónimo: no es pariente"]')
            p.click('#form-dec button[type="submit"]')
            p.wait_for_timeout(200)
            p.goto(U + "#entidad-wayrapampa", wait_until="networkidle")
            p.wait_for_timeout(400)
            p.locator("#dec-resumen").scroll_into_view_if_needed()
            p.evaluate("window.scrollBy(0, -120)")
            p.wait_for_timeout(200)
            p.screenshot(path=str(CAP / f"{PRE}_{disp}_7_lista_con_decisiones.png"))
            p.evaluate("localStorage.removeItem('llika-decisiones')")
        ctx.close()
    nav.close()
print("capturas en:", CAP)
