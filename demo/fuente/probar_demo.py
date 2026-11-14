# Prueba la demo en Chromium: errores de consola, desborde horizontal, reglas del sistema de diseño (pesos, familias,
# tamaños y radios), área de toque en el celular, interacción, y capturas a 1440 y 375 px en claro y oscuro.
#   python probar_demo.py [archivo.html] [carpeta_de_capturas]
# Sin argumentos prueba ../llika_pe_demo.html y deja las capturas en la carpeta temporal del sistema.
import json
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

AQUI = Path(__file__).resolve().parent
ARCHIVO = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent / "llika_pe_demo.html"
CAP = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / "llika_demo_capturas"
CAP.mkdir(parents=True, exist_ok=True)
U = ARCHIVO.resolve().as_uri()
RUTAS = [("inicio", "#inicio"), ("entidad", "#entidad-wayrapampa"), ("ficha", "#caso-wayrapampa-1"), ("ficha3", "#caso-wayrapampa-3"), ("ficha5", "#caso-wayrapampa-5"),
         ("ficha12", "#caso-wayrapampa-12"), ("como", "#como-funciona"), ("datos", "#datos"), ("no", "#nada")]

# El área que de verdad responde al toque se mide con elementFromPoint: cuenta también los seudoelementos que la amplían.
MEDIR = """() => {
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const els = [...document.querySelectorAll('body *')].filter(vis);
  const pesos = new Set(), familias = new Set(), tamanos = new Set(), radios = new Set();
  els.forEach(el => { const s = getComputedStyle(el); if (el.childNodes.length && [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) { pesos.add(s.fontWeight); familias.add(s.fontFamily.split(',')[0].replace(/"/g, '').trim()); tamanos.add(s.fontSize); }
    ['borderTopLeftRadius'].forEach(k => { if (s[k] !== '0px') radios.add(s[k]); }); });
  const panel = document.getElementById('panel'), abierto = panel && !panel.hidden;
  const toque = el => { el.scrollIntoView({ block: 'center' }); const r = el.getBoundingClientRect(); const cx = r.left + Math.min(r.width / 2, 16), cy = r.top + r.height / 2; let h = 0, w = 0;
    for (let y = cy - 40; y <= cy + 40; y++) { const t = document.elementFromPoint(cx, y); if (t && (t === el || el.contains(t))) h++; }
    for (let x = r.left - 40; x <= r.right + 40; x++) { const t = document.elementFromPoint(x, cy); if (t && (t === el || el.contains(t))) w++; }
    return { h: Math.max(h, Math.round(r.height)), w: Math.max(w, Math.round(r.width)) }; };
  const toca = [...document.querySelectorAll('a[href], button, input, select, textarea, summary')].filter(vis).filter(el => !abierto || panel.contains(el)).map(el => { const m = toque(el);
    return { t: (el.getAttribute('aria-label') || el.textContent || el.placeholder || el.id || el.tagName).trim().slice(0, 28), w: m.w, h: m.h, enTexto: !!el.closest('p, li, td.prov, .nota-pie, small, .migas, .pie') && el.tagName === 'A' }; });
  window.scrollTo(0, 0);
  return { pesos: [...pesos].sort(), familias: [...familias].sort(), tamanos: [...tamanos].sort((a, b) => parseFloat(a) - parseFloat(b)), radios: [...radios].sort((a, b) => parseFloat(a) - parseFloat(b)),
    desborde: document.documentElement.scrollWidth - document.documentElement.clientWidth, toca };
}"""

res, errores = {}, []
with sync_playwright() as pw:
    nav = pw.chromium.launch()
    for nombre, w, h in (("pc", 1440, 900), ("cel", 375, 812)):
        for tema in ("light", "dark"):
            ctx = nav.new_context(viewport={"width": w, "height": h}, device_scale_factor=1, color_scheme=tema, has_touch=(nombre == "cel"))
            p = ctx.new_page()
            p.on("console", lambda m: errores.append(m.text) if m.type == "error" else None)
            p.on("pageerror", lambda e: errores.append(str(e)))
            for vista, ruta in RUTAS:
                p.goto(U + ruta, wait_until="networkidle")
                p.wait_for_timeout(450)
                if tema == "light" or vista in ("entidad", "ficha", "como"):
                    if nombre == "cel" and vista == "entidad":
                        p.screenshot(path=str(CAP / f"{nombre}_{tema}_{vista}.png"), full_page=True, clip={"x": 0, "y": 0, "width": w, "height": 2400})
                    else:
                        p.screenshot(path=str(CAP / f"{nombre}_{tema}_{vista}.png"), full_page=(vista in ("como", "datos")))
                    if vista.startswith("ficha") and tema == "light":
                        # el resto de la ficha: se abre la lista de compras y se baja por tramos
                        p.evaluate("document.querySelectorAll('#panel details').forEach(d => { d.open = true; })")
                        alto = p.evaluate("document.getElementById('panel-cuerpo').scrollHeight")
                        for k, y in enumerate(range(h - 120, alto, h - 120)):
                            p.evaluate(f"document.getElementById('panel-cuerpo').scrollTop = {y}")
                            p.wait_for_timeout(120)
                            p.screenshot(path=str(CAP / f"{nombre}_{tema}_{vista}_{chr(98 + k)}.png"))
                        p.evaluate("document.getElementById('panel-cuerpo').scrollTop = 0; document.querySelectorAll('#panel details').forEach(d => { d.open = false; })")
                if tema == "light":
                    m = p.evaluate(MEDIR)
                    chicos = [x for x in m["toca"] if not x["enTexto"] and (x["h"] < (44 if nombre == "cel" else 24) or x["w"] < 24)]
                    res[f"{nombre}_{vista}"] = {"pesos": m["pesos"], "familias": m["familias"], "tamanos": m["tamanos"], "radios": m["radios"], "desborde": m["desborde"], "objetivos_chicos": chicos[:12]}
            if nombre == "pc" and tema == "light":
                # interacción: filtros, orden, estado vacío, búsqueda y teclado en la ficha
                p.goto(U + "#entidad-wayrapampa", wait_until="networkidle")
                p.click('[data-f="vin"]')
                n_vin = p.locator("#filas tr.fila").count()
                p.click('[data-orden="monto"]')
                primero = p.locator("#filas tr.fila .p-nom").first.inner_text()
                p.fill("#fq", "zzzz")
                vacio = p.locator(".vacio").inner_text()
                p.screenshot(path=str(CAP / "pc_light_vacio.png"))
                p.click("[data-limpiar]")
                n_todo = p.locator("#filas tr.fila").count()
                p.fill("#q-cab", "llamkay")
                p.wait_for_timeout(200)
                sug = p.locator("#sug-cab a").first.inner_text()
                p.screenshot(path=str(CAP / "pc_light_busqueda.png"))
                p.keyboard.press("Enter")
                p.wait_for_timeout(400)
                titulo = p.locator("#p-nombre").inner_text()
                calculo = p.locator(".calculo").inner_text()
                p.keyboard.press("Escape")
                p.wait_for_timeout(300)
                cerrado = p.locator("#panel").is_hidden()
                res["interaccion"] = {"filas_con_vinculo": n_vin, "primero_por_monto": primero, "vacio": vacio.replace("\n", " | "), "filas_sin_filtros": n_todo, "sugerencia": sug.replace("\n", " | "), "ficha": titulo,
                                      "calculo": calculo.replace("\n", " | ")[:600], "escape_cierra": cerrado}
            if nombre == "cel" and tema == "light":
                # en el celular: tocar una tarjeta de entidad y una de proveedor abre lo que corresponde
                p.goto(U + "#inicio", wait_until="networkidle")
                p.locator("#t-entidades tr").nth(1).tap(position={"x": 300, "y": 60})
                p.wait_for_timeout(300)
                abre_entidad = p.evaluate("location.hash")
                p.locator("#filas tr.fila").nth(2).tap(position={"x": 200, "y": 10})
                p.wait_for_timeout(400)
                res["toque_cel"] = {"tarjeta_entidad": abre_entidad, "tarjeta_proveedor": p.evaluate("location.hash"), "ficha": p.locator("#p-nombre").inner_text()}
            ctx.close()
    nav.close()

(CAP / "medidas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in res.items():
    if k in ("interaccion", "toque_cel"):
        print(k.upper(), json.dumps(v, ensure_ascii=False))
        continue
    print(k, "| pesos", v["pesos"], "| familias", v["familias"], "| tamaños", [x.replace("px", "") for x in v["tamanos"]], "| radios", [x.replace("px", "") for x in v["radios"]], "| desborde", v["desborde"])
    if v["objetivos_chicos"]:
        print("     chicos:", "; ".join(f"{x['t']} {x['w']}x{x['h']}" for x in v["objetivos_chicos"]))
print("ERRORES:", errores)
print("capturas en:", CAP)
