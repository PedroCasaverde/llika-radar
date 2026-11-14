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
            if nombre == "pc" and tema == "light":
                # Los cambios de la prueba de uso (HALLAZGOS 160, sección 92): cada uno con lo que debe cumplir
                cp = {}
                casos = p.evaluate("JSON.parse(document.getElementById('datos-llika').textContent).entidades.flatMap(e => e.casos.map(c => [e.id, c.rk, (c.vin || {}).tipo || '']))")
                con_porcentaje, sin_aviso, declarados, con_quien = [], [], 0, 0
                for eid, rk, vin in casos:
                    p.goto(U + f"#caso-{eid}-{rk}")
                    p.wait_for_timeout(20)
                    fija = p.locator("#panel details.fija")
                    if fija.count() and "%" in fija.inner_text():
                        con_porcentaje.append(f"{eid}-{rk}")
                    av = p.locator("#panel .p-aviso")
                    if not (av.count() and av.is_visible() and av.bounding_box()["y"] < h and "No constituye sanción" in av.inner_text()):
                        sin_aviso.append(f"{eid}-{rk}")
                    if vin == "declarado":
                        declarados += 1
                        con_quien += "Lo declaró:" in p.locator("#panel .senales").inner_text()
                cp["1_parecido_sin_porcentajes"] = (not con_porcentaje, f"{len(casos)} fichas; con % en el bloque: {con_porcentaje[:5]}")
                p.goto(U + "#caso-wayrapampa-3")
                p.wait_for_timeout(200)
                cp["1_parecido_en_orden"] = ("de más a menos importante" in p.locator("#panel details.fija").inner_text(), "encabezado de la tabla")
                p.goto(U + "#caso-wayrapampa-5")
                p.wait_for_timeout(300)
                t, g = p.locator("#panel .t-ords"), p.locator("#g-ords")
                cp["3_compras_abiertas_y_arriba"] = (t.is_visible() and t.bounding_box()["y"] < g.bounding_box()["y"] and not p.locator("#panel details.ords").count(),
                                                     f"{t.locator('tbody tr').count()} compras a la vista sin hacer clic")
                cp["4_aviso_sin_bajar"] = (not sin_aviso, f"en {len(casos) - len(sin_aviso)} de {len(casos)} fichas")
                p.goto(U + "#caso-wayrapampa-2")
                p.wait_for_timeout(200)
                co = p.locator("#panel .coincide")
                tam = co.evaluate("e => getComputedStyle(e).fontSize") if co.count() else "no está"
                cp["4_coincidencia_16px"] = (tam == "16px" and "sin confirmar" in co.inner_text(), f"{tam}; lista: {p.locator('#filas tr.fila[data-rk=\"2\"] td.ev').first.inner_text()}")
                cp["6_quien_declaro"] = (declarados == con_quien == 9, f"{con_quien} de {declarados} fichas con vínculo declarado")
                # Excel: se baja desde la lista y se abre
                import openpyxl
                p.goto(U + "#entidad-wayrapampa")
                p.wait_for_timeout(300)
                with p.expect_download() as bajada:
                    p.click("[data-excel]")
                xlsx = CAP / bajada.value.suggested_filename
                bajada.value.save_as(str(xlsx))
                lb = openpyxl.load_workbook(xlsx)
                hp, hc = lb["Proveedores"], lb["Compras"]
                filas_p = [r for r in hp.iter_rows(min_row=4, values_only=True) if r[0] is not None]
                filas_c = [r for r in hc.iter_rows(min_row=4, values_only=True) if r[0] is not None]
                total_p = round(sum(r[6] for r in filas_p), 2)
                total_c = round(sum(r[6] for r in filas_c), 2)
                rucs_ok = all(isinstance(r[3], str) and len(r[3]) == 11 for r in filas_p)
                cp["2_excel"] = (lb.sheetnames == ["Proveedores", "Compras"] and len(filas_p) == 100 and [r[0] for r in filas_p] == list(range(1, 101)) and rucs_ok and total_p == total_c
                                 and "no constituye sanción" in hp["A2"].value,
                                 f"{xlsx.name}: {len(filas_p)} proveedores, {len(filas_c)} compras, total S/ {total_p:,.2f} en las dos hojas; RUC como texto: {rucs_ok}")
                # con un filtro, baja solo lo que se ve
                p.click('[data-f="vin"]')
                with p.expect_download() as bajada2:
                    p.click("[data-excel]")
                x2 = CAP / ("filtrado_" + bajada2.value.suggested_filename)
                bajada2.value.save_as(str(x2))
                n_vis = p.locator("#filas tr.fila").count()
                n_xl = sum(1 for r in openpyxl.load_workbook(x2)["Proveedores"].iter_rows(min_row=4, values_only=True) if r[0] is not None)
                cp["2_excel_respeta_filtros"] = (n_vis == n_xl, f"{n_xl} filas en el Excel, {n_vis} en pantalla con el filtro de vínculo")
                res["prueba_de_uso"] = {k: {"ok": v[0], "detalle": v[1]} for k, v in cp.items()}
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
    if k == "prueba_de_uso":
        print("CAMBIOS DE LA PRUEBA DE USO")
        for c, r in v.items():
            print(f"  {'OK   ' if r['ok'] else 'FALLA'} {c}: {r['detalle']}")
        continue
    if k in ("interaccion", "toque_cel"):
        print(k.upper(), json.dumps(v, ensure_ascii=False))
        continue
    print(k, "| pesos", v["pesos"], "| familias", v["familias"], "| tamaños", [x.replace("px", "") for x in v["tamanos"]], "| radios", [x.replace("px", "") for x in v["radios"]], "| desborde", v["desborde"])
    if v["objetivos_chicos"]:
        print("     chicos:", "; ".join(f"{x['t']} {x['w']}x{x['h']}" for x in v["objetivos_chicos"]))
print("ERRORES:", errores)
print("capturas en:", CAP)
