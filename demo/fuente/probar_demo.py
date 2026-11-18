# Prueba la demo en Chromium: errores de consola, desborde horizontal, reglas del sistema de diseño (pesos, familias,
# tamaños y radios), área de toque en el celular, interacción, y capturas a 1440 y 375 px en claro y oscuro.
#   python probar_demo.py [archivo.html] [carpeta_de_capturas]
# Sin argumentos prueba ../llika_pe_demo.html y deja las capturas en la carpeta temporal del sistema.
# También baja la carpeta de cada proveedor (un zip por ficha), la abre y revisa sus archivos, y comprueba que el
# formulario de corrección deje escrito un correo al buzón del equipo, con el RUC.
# Y que cada índice lleve su lugar entre las 2,647 entidades reales de la lista (el texto, el largo de la barra y el color),
# igual al que se recalcula aquí con la distribución que trae la página.
import bisect
import html
import io
import json
import math
import re
import sys
import tempfile
import urllib.parse
import zipfile
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

# Etapa 2: cada señal de la ficha trae su norma y su fuente con fecha; también la tabla de compras
FECHA = r"\d{1,2} (ene|feb|mar|abr|may|jun|jul|ago|set|oct|nov|dic) \d{4}"
REFS = """() => { const f = /""" + FECHA + """/, bien = r => !!r && /Norma/.test(r.textContent) && /Ley \\d{5}|Ninguna norma/.test(r.textContent) && f.test(r.textContent);
  const ss = [...document.querySelectorAll('#panel .senal')], c = document.querySelector('#p-ord ~ .ref, section[aria-labelledby="p-ord"] > .ref');
  return { senales: ss.length, malas: ss.filter(s => !bien(s.querySelector('.ref'))).length, compras: !document.querySelector('#panel .t-ords') || bien(c) }; }"""


BUZON = "pedro.casaverde@utec.edu.pe"
ARCHIVOS_CARPETA = ["LÉEME.txt", "ficha.html", "compras_y_senales.xlsx"]


def leer_carpeta(ruta):
    """Abre el zip de la carpeta del proveedor: sus archivos, si alguno está dañado, los dos textos y el Excel de dentro."""
    import openpyxl
    with zipfile.ZipFile(ruta) as z:
        nombres, danado = z.namelist(), z.testzip()
        leeme, ficha = z.read("LÉEME.txt").decode("utf-8-sig"), z.read("ficha.html").decode("utf-8")
        libro = openpyxl.load_workbook(io.BytesIO(z.read("compras_y_senales.xlsx")))
    filas = {h: [r for r in libro[h].iter_rows(min_row=4, values_only=True) if r[0] is not None] for h in libro.sheetnames}
    return {"nombres": nombres, "danado": danado, "leeme": leeme, "ficha": ficha, "hojas": libro.sheetnames, "compras": filas.get("Compras", []),
            "senales": filas.get("Señales", []), "nota": libro.worksheets[0]["A2"].value or ""}


def indices_python(D):
    """Los seis índices por entidad, recalculados aquí con la misma regla de la demo (compras partidas: dos o más en 30 días sobre el tope, o una de 90% del tope)."""
    from datetime import date
    tope, out = D["tope"], {}
    for e in D["entidades"]:
        cs, n = e["casos"], len(e["casos"])
        def partidas(c):
            os_ = sorted((date.fromisoformat(o[0]), float(o[3] or 0)) for o in c["ord"])
            cerca = any(m >= 0.9 * tope for _, m in os_)
            b30 = any(sum(m2 for t2, m2 in os_ if 0 <= (t - t2).days < 30) > tope and sum(1 for t2, _ in os_ if 0 <= (t - t2).days < 30) >= 2 for t, _ in os_)
            return cerca or b30
        totales = sorted((sum(float(o[3] or 0) for o in c["ord"]) for c in cs), reverse=True)
        compras = [float(o[3] or 0) for c in cs for o in c["ord"]]
        out[e["id"]] = {"vin": 100 * sum(1 for c in cs if c.get("vin")) / n,
                        "ant": 100 * sum(1 for c in cs if any((c.get("ant") or {}).get(k) for k in ("pen", "inh", "mul"))) / n,
                        "fra": 100 * sum(1 for c in cs if partidas(c)) / n,
                        "tope": 100 * sum(1 for m in compras if m >= 0.9 * tope) / len(compras),
                        "conc": 100 * sum(totales[:5]) / sum(totales),
                        "dep": 100 * sum(1 for c in cs if (c.get("clocal") or 0) >= 80) / n}
    return out


def lugar_python(pais, k, v):
    """El lugar de un valor entre las entidades del país, con la regla de la página: cuántas tienen menos, igual y más.
    Los valores van en centésimas de punto porcentual, ordenados."""
    a, x = pais["ind"][k], math.floor(100 * v + 0.5)
    menos, hasta = bisect.bisect_left(a, x), bisect.bisect_right(a, x)
    return {"menos": menos, "igual": hasta - menos, "mas": len(a) - hasta, "pct": 100 * menos / len(a)}


def lugar_corto(l):
    if l["menos"] == 0:
        return "ninguna tiene menos"
    if l["mas"] == 0:
        return "ninguna tiene más"
    return f"más que {l['menos']:,} {'entidad' if l['menos'] == 1 else 'entidades'}" if l["pct"] < 1 else f"más que el {math.floor(l['pct'])}%"


def lugar_largo(l, n):
    """La oración con que abre cada índice en la página de la entidad."""
    if l["menos"] == 0 or l["mas"] == 0:
        return f"Ninguna de las {n:,} entidades tiene {'menos' if l['menos'] == 0 else 'más'}."
    return f"Más que {l['menos']:,} de las {n:,} entidades." if l["pct"] < 1 else f"Más que el {math.floor(l['pct'])}% de las {n:,} entidades."


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
                sin_norma, n_senales = [], 0  # etapa 2: norma y fecha de la fuente en cada señal y en las compras
                for eid, rk, vin in casos:
                    p.goto(U + f"#caso-{eid}-{rk}")
                    p.wait_for_timeout(20)
                    refs = p.evaluate(REFS)
                    n_senales += refs["senales"]
                    if refs["malas"] or not refs["compras"]:
                        sin_norma.append(f"{eid}-{rk}")
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

                # Etapa 2 del cierre (HALLAZGOS 160, sección 93): lo que faltaba del contrato de alcance, y la marca
                ct = {}
                ct["norma_y_fecha_en_cada_senal"] = (not sin_norma and n_senales > 0, f"{n_senales} señales en {len(casos)} fichas, todas con norma y fecha de su fuente; también las compras. Sin: {sin_norma[:5]}")
                # Índices: los de la página contra los que se recalculan aquí, desde los mismos datos
                D = json.loads(p.evaluate("document.getElementById('datos-llika').textContent"))
                p.goto(U + "#inicio")
                p.wait_for_timeout(300)
                en_pagina = p.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#t-indices tr')].map(tr => [tr.dataset.href.replace('#entidad-', ''),
                  Object.fromEntries([...tr.querySelectorAll('.ind')].map(d => [d.dataset.k, parseFloat(d.querySelector('b').textContent)]))]))""")
                calc = indices_python(D)
                dif = [(e, k, en_pagina[e][k], round(v, 1)) for e, vs in calc.items() for k, v in vs.items() if abs(en_pagina[e][k] - v) > 0.051]
                ct["indices_iguales_a_python"] = (not dif and len(en_pagina) == 3 and all(len(v) == 6 for v in en_pagina.values()), f"3 entidades x 6 índices; diferencias: {dif[:4]}; Wayrapampa {en_pagina['wayrapampa']}")
                p.click('[data-ind="fra"]')
                p.wait_for_timeout(100)
                primera = p.locator("#t-indices tr").first.get_attribute("data-href")
                mayor = max(calc, key=lambda e: calc[e]["fra"])
                ct["indices_se_ordenan"] = (primera == f"#entidad-{mayor}", f"al ordenar por compras partidas, primero {primera}")
                p.goto(U + "#entidad-ccori-llacta")
                p.wait_for_timeout(300)
                ct["indices_en_cada_entidad"] = (p.locator(".compara > div").count() == 6 and p.locator(".compara .via").count() == 6, f"{p.locator('.compara > div').count()} índices con su barra")
                # Etapa B del cierre: cada índice va con su lugar entre las entidades reales del país (D.pais), calculado aquí con la misma regla
                cb = {}
                pais = D.get("pais") or {}
                cb["distribucion_del_pais"] = (pais.get("n") == 2647 and list(pais.get("ind", {})) == list(next(iter(calc.values()))) and all(len(v) == pais["n"] and v == sorted(v) for v in pais["ind"].values()),
                                               f"{pais.get('n')} entidades de la lista de {pais.get('anio')}, {len(pais.get('ind', {}))} índices, cada uno ordenado")
                p.goto(U + "#inicio")
                p.wait_for_timeout(300)
                celdas = p.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#t-indices tr')].map(tr => [tr.dataset.href.replace('#entidad-', ''),
                  Object.fromEntries([...tr.querySelectorAll('.ind')].map(d => [d.dataset.k, { texto: d.querySelector('small').textContent.replace(/\\u00a0/g, ' '),
                    barra: parseFloat(d.querySelector('.via span').style.getPropertyValue('--v')), alto: d.classList.contains('alto') }]))]))""")
                esperado = {e: {k: lugar_python(pais, k, v) for k, v in vs.items()} for e, vs in calc.items()}
                dif_l = [(e, k, celdas[e][k], lugar_corto(l)) for e, ls in esperado.items() for k, l in ls.items()
                         if celdas[e][k]["texto"] != lugar_corto(l) or abs(celdas[e][k]["barra"] - l["pct"]) > 0.051 or celdas[e][k]["alto"] != (l["pct"] >= 90)]
                cb["lugar_igual_a_python"] = (not dif_l and len(celdas) == 3, f"3 entidades x 6 índices: texto, largo de la barra y color; diferencias: {dif_l[:3]}; Wayrapampa: {[celdas['wayrapampa'][k]['texto'] for k in esperado['wayrapampa']]}")
                nota = p.locator("#nota-ind").inner_text()
                cb["nota_de_los_indices"] = ("inventadas" in nota and "2,647 entidades reales de la lista de 2025" in nota and "percentil" in nota and not re.search(r"se ajustarán|sin ajustar|encogimiento", p.content()),
                                             f"la demo dice que sus entidades son inventadas y que se comparan con las reales: «{nota[:110]}…»")
                clip = p.locator("section[data-vista='inicio'] .caja").bounding_box()
                p.screenshot(path=str(CAP / "etapa_b_1_indices.png"), clip={"x": 0, "y": clip["y"] + p.evaluate("scrollY") - 72, "width": w, "height": clip["height"] + 420}, full_page=True)
                p.goto(U + "#entidad-wayrapampa")
                p.wait_for_timeout(300)
                fichas_ind = p.evaluate("""() => Object.fromEntries([...document.querySelectorAll('.compara > div')].map(d => [d.dataset.k, d.querySelector('small').textContent.replace(/\\u00a0/g, ' ')]))""")
                parrafo = p.locator("section[aria-labelledby='t-compara'] p").inner_text()
                bien_e = all(t_.startswith(lugar_largo(esperado["wayrapampa"][k], pais["n"])) for k, t_ in fichas_ind.items())
                cb["lugar_en_cada_entidad"] = (bien_e and len(fichas_ind) == 6 and "2,647 entidades del país" in p.locator("#t-compara").inner_text() and "Esta entidad es inventada" in parrafo and "entidades reales" in parrafo,
                                               f"«{fichas_ind['fra']}» | «{parrafo[:95]}…»")
                caja_c = p.locator("section[aria-labelledby='t-compara']").bounding_box()
                p.screenshot(path=str(CAP / "etapa_b_2_entidad.png"), clip={"x": 0, "y": max(0, caja_c["y"] + p.evaluate("scrollY") - 230), "width": w, "height": caja_c["height"] + 250}, full_page=True)
                res["etapa_b"] = {k_: {"ok": v[0], "detalle": v[1]} for k_, v in cb.items()}
                # Confirmar o descartar con su motivo
                p.goto(U + "#caso-wayrapampa-1")
                p.wait_for_timeout(300)
                p.click('#panel-pie [data-dec="confirmar"]')
                p.click('#form-dec button[type="submit"]')
                sin_motivo = p.locator("#e-motivo").inner_text()
                p.click('[data-motivo="El vínculo se confirma"]')
                p.click('#form-dec button[type="submit"]')
                p.wait_for_timeout(150)
                guardado1 = p.locator("#panel-pie").inner_text()
                p.goto(U + "#caso-wayrapampa-2")
                p.wait_for_timeout(300)
                p.click('#panel-pie [data-dec="descartar"]')
                p.click('[data-motivo="Otro motivo"]')
                p.click('#form-dec button[type="submit"]')
                sin_nota = p.locator("#e-nota").inner_text()
                p.fill("#dec-nota", "Lo revisó el área usuaria en octubre")
                p.click('#form-dec button[type="submit"]')
                p.wait_for_timeout(150)
                guardado2 = p.locator("#panel-pie").inner_text()
                ct["decidir_pide_motivo"] = (sin_motivo == "Elige un motivo." and bool(sin_nota) and "Confirmado para revisar" in guardado1 and "El vínculo se confirma" in guardado1
                                             and "Descartado" in guardado2 and "Lo revisó el área usuaria" in guardado2, f"sin motivo: «{sin_motivo}»; «Otro motivo» sin nota: «{sin_nota}»")
                p.goto(U + "#caso-wayrapampa-3")
                p.wait_for_timeout(300)
                p.click('#panel-pie [data-dec="confirmar"]')
                p.keyboard.press("Escape")
                p.wait_for_timeout(150)
                esc1 = p.locator("#panel").is_visible() and not p.locator("#form-dec").count()
                p.keyboard.press("Escape")
                p.wait_for_timeout(300)
                ct["escape_cierra_primero_el_formulario"] = (esc1 and p.locator("#panel").is_hidden(), "el primer Escape cierra el formulario y el segundo, la ficha")
                p.reload()
                p.wait_for_timeout(400)
                etq = {rk: p.locator(f'#filas tr.fila[data-rk="{rk}"] .dec').inner_text() if p.locator(f'#filas tr.fila[data-rk="{rk}"] .dec').count() else "" for rk in (1, 2, 3)}
                cuenta_dec = {}
                for v in ("confirmar", "descartar", "sin", "todas"):
                    p.select_option("#fdec", v)
                    p.wait_for_timeout(80)
                    cuenta_dec[v] = p.locator("#filas tr.fila").count()
                ct["decision_sigue_al_recargar_y_se_filtra"] = (etq == {1: "Para revisar", 2: "Descartado", 3: ""} and cuenta_dec == {"confirmar": 1, "descartar": 1, "sin": 98, "todas": 100},
                                                                 f"en la lista: {etq}; filtro: {cuenta_dec}")
                with p.expect_download() as bajada3:
                    p.click("[data-excel]")
                x3 = CAP / ("decisiones_" + bajada3.value.suggested_filename)
                bajada3.value.save_as(str(x3))
                hp3 = openpyxl.load_workbook(x3)["Proveedores"]
                cab = [c.value for c in hp3[3]]
                filas3 = {r[0]: r for r in hp3.iter_rows(min_row=4, values_only=True) if r[0] is not None}
                i_dec, i_mot, i_nota, i_norma, i_fte = cab.index("Tu decisión"), cab.index("Motivo"), cab.index("Nota"), cab.index("Normas"), cab.index("Fuentes y fecha de sus datos")
                con_senal = [r for r in filas3.values() if r[i_fte]]
                ct["excel_trae_decision_norma_y_fecha"] = (filas3[1][i_dec] == "Confirmado para revisar" and filas3[1][i_mot] == "El vínculo se confirma" and filas3[2][i_dec] == "Descartado"
                                                         and filas3[2][i_nota] == "Lo revisó el área usuaria en octubre" and not filas3[3][i_dec] and len(filas3) == 100
                                                         and all(r[i_norma] and re.search(FECHA, r[i_fte]) for r in con_senal),
                                                         f"{len(cab)} columnas; decisión en las filas 1 y 2; {len(con_senal)} filas con señal, todas con norma y fecha")
                p.goto(U + "#caso-wayrapampa-2")
                p.wait_for_timeout(300)
                p.click("#panel-pie [data-dec-cambiar]")
                p.click("#panel-pie [data-dec-quitar]")
                p.wait_for_timeout(150)
                guardadas = p.evaluate("Object.keys(JSON.parse(localStorage.getItem('llika-decisiones') || '{}'))")
                ct["decision_se_puede_quitar"] = (guardadas == ["wayrapampa-1"] and p.locator('#panel-pie [data-dec="confirmar"]').count() == 1, f"quedan guardadas: {guardadas}")
                p.evaluate("localStorage.removeItem('llika-decisiones')")
                # Imágenes: todas cargan y tienen texto alternativo (vacío si son decorativas)
                malas_img, n_img = [], 0
                for vista, ruta in RUTAS:
                    p.goto(U + ruta)
                    p.wait_for_timeout(250)
                    r = p.evaluate("() => { const v = [...document.images].filter(i => i.getBoundingClientRect().width > 0); return { n: v.length, malas: v.filter(i => !(i.complete && i.naturalWidth > 0 && i.hasAttribute('alt'))).map(i => i.dataset.img || i.src.slice(0, 30)) }; }")
                    n_img += r["n"]
                    malas_img += [f"{vista}: {x}" for x in r["malas"]]
                ct["imagenes_cargan_con_alt"] = (not malas_img and n_img > 0, f"{n_img} imágenes visibles en {len(RUTAS)} vistas; con problemas: {malas_img[:5]}")
                ct["logo_y_marca"] = (p.locator(".cabecera .logo-red .centro").count() == 1 and p.evaluate("getComputedStyle(document.querySelector('.cabecera')).backgroundColor") == "rgb(77, 75, 76)",
                                      "logo con la telaraña y cabecera en el carbón de UTEC (#4D4B4C)")
                res["etapa_2"] = {k: {"ok": v[0], "detalle": v[1]} for k, v in ct.items()}

                # Etapa A del cierre (17/11): la carpeta del proveedor (un zip por ficha) y el pedido de corrección por correo
                ca, malas, n_carp, n_filas_sen = {}, [], 0, 0
                for e in D["entidades"]:
                    for c in e["casos"]:
                        p.goto(U + f"#caso-{e['id']}-{c['rk']}")
                        p.wait_for_timeout(20)
                        en_pantalla = [t.rstrip(".") for t in p.locator("#panel .senal > span:first-of-type > b").all_inner_texts()]
                        with p.expect_download() as baja:
                            p.click("[data-carpeta]")
                        z = CAP / "carpetas" / baja.value.suggested_filename
                        z.parent.mkdir(exist_ok=True)
                        baja.value.save_as(str(z))
                        k, n_carp = leer_carpeta(z), n_carp + 1
                        n_filas_sen += len(k["senales"])
                        ficha = html.unescape(k["ficha"])
                        etiquetas = {r[0] for r in k["senales"]}
                        fallas = [nombre_f for nombre_f, bien in (
                            ("archivos", k["nombres"] == ARCHIVOS_CARPETA and k["danado"] is None and z.name == f"llika_carpeta_{c['prov']['ruc']}_{e['id']}_{D['anio']}.zip"),
                            ("hojas", k["hojas"] == ["Compras", "Señales"] and "no constituye sanción" in k["nota"]),
                            ("compras", len(k["compras"]) == len(c["ord"]) and round(sum(r[6] for r in k["compras"]), 2) == round(sum(float(o[3] or 0) for o in c["ord"]), 2)),
                            ("señales", (set(en_pantalla) <= etiquetas and len(k["senales"]) >= len(en_pantalla) and all(r[2] and re.search(FECHA, r[4] or "") for r in k["senales"]))
                             if en_pantalla else [r[0] for r in k["senales"]] == ["Ninguna"]),
                            ("ficha", all(x in ficha for x in ("No constituye sanción, falta ni delito", "Demo con datos inventados", c["prov"]["nombre"], c["prov"]["ruc"], c["por"], e["nombre"]))),
                            ("léeme", all(x in k["leeme"] for x in (f"31 dic {D['anio']}", c["prov"]["ruc"], "no constituye sanción, falta ni delito", "inventados", BUZON))),
                            ("palabra", "expediente" not in (k["leeme"] + " ".join(re.findall(r"<h[23]>(.*?)</h[23]>", k["ficha"])) + " ".join(k["nombres"])).lower()),
                        ) if not bien]
                        if fallas:
                            malas.append(f"{e['id']}-{c['rk']}: {', '.join(fallas)}")
                        elif e["id"] != "wayrapampa" or c["rk"] > 3:
                            z.unlink()  # quedan de muestra las tres primeras de Wayrapampa, y las que fallen
                ca["carpeta_baja_y_trae_sus_archivos"] = (not malas and n_carp == len(casos), f"{n_carp} carpetas: {', '.join(ARCHIVOS_CARPETA)}; el Excel de dentro abre, con {n_filas_sen} filas de señales en total. Con fallas: {malas[:5]}")
                # La ficha de la carpeta lleva la decisión de quien revisó, si la hay
                p.goto(U + "#caso-wayrapampa-3")
                p.wait_for_timeout(300)
                p.click('#panel-pie [data-dec="confirmar"]')
                p.click("#form-dec .motivo >> nth=0")
                motivo = p.locator('#form-dec .motivo[aria-checked="true"]').inner_text()
                p.click('#form-dec button[type="submit"]')
                p.wait_for_timeout(150)
                p.locator("[data-carpeta]").scroll_into_view_if_needed()
                p.locator("#panel").screenshot(path=str(CAP / "etapa_a_1_boton.png"))
                with p.expect_download() as baja:
                    p.click("[data-carpeta]")
                z = CAP / "carpetas" / ("con_decision_" + baja.value.suggested_filename)
                baja.value.save_as(str(z))
                k = leer_carpeta(z)
                ca["carpeta_trae_la_decision"] = ("Confirmado para revisar" in k["ficha"] and html.escape(motivo) in k["ficha"], f"«Confirmado para revisar», motivo «{motivo}»")
                p.click("#panel-pie [data-dec-cambiar]")
                p.click("#panel-pie [data-dec-quitar]")
                p.evaluate("localStorage.removeItem('llika-decisiones')")
                # La ficha de la carpeta se abre sola en el navegador: sin errores ni desborde
                destino = CAP / "carpetas" / "abierta"
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(destino)
                pf = ctx.new_page()
                err_f = []
                pf.on("console", lambda m: err_f.append(m.text) if m.type == "error" else None)
                pf.goto((destino / "ficha.html").resolve().as_uri())
                des_f = pf.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                pf.screenshot(path=str(CAP / "etapa_a_2_ficha_de_la_carpeta.png"), full_page=True)
                pf.set_viewport_size({"width": 375, "height": 812})
                des_f375 = pf.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                pf.close()
                ca["ficha_de_la_carpeta_abre_sola"] = (not err_f and des_f <= 0 and des_f375 <= 0, f"sin errores de consola; desborde {des_f} px a 1440 y {des_f375} px a 375")
                # El pedido de corrección: el formulario valida y deja escrito un correo al buzón del equipo.
                # La prueba captura el enlace en vez de abrir el programa de correo de esta computadora.
                p.evaluate("""() => { window.__correo = []; const real = HTMLAnchorElement.prototype.click;
                  HTMLAnchorElement.prototype.click = function () { if (this.href.startsWith('mailto:')) { window.__correo.push(this.href); return; } return real.call(this); }; }""")
                c3 = next(c for c in D["entidades"][0]["casos"] if c["rk"] == 3)
                p.click("#abrir-cor")
                p.click('#cor button[type="submit"]')
                vacio_err = [t for t in p.locator("#cor .error").all_inner_texts() if t]
                sin_correo = p.evaluate("window.__correo.length") == 0 and p.locator("#c-ok").is_hidden()
                escrito = "La compra O/C 0017 figura dos veces: es una sola, de S/ 11,900."
                p.select_option("#c-tipo", "Compras o montos")
                p.fill("#c-det", escrito)
                p.fill("#c-mail", "ana@ejemplo.pe")
                p.click('#cor button[type="submit"]')
                p.wait_for_timeout(200)
                correos = p.evaluate("window.__correo")
                para, _, consulta = (correos[0] if correos else "").partition("?")
                campos = urllib.parse.parse_qs(consulta)
                asunto, cuerpo_c = campos.get("subject", [""])[0], campos.get("body", [""])[0]
                aviso_c = p.locator("#c-ok").inner_text()
                ca["formulario_valida_antes"] = (len(vacio_err) == 2 and sin_correo, f"vacío no abre el correo y avisa: {vacio_err}")
                ca["correo_al_buzon_con_el_ruc"] = (len(correos) == 1 and para == f"mailto:{BUZON}" and c3["prov"]["ruc"] in asunto
                                                    and all(x in cuerpo_c for x in (c3["prov"]["ruc"], c3["prov"]["nombre"], D["entidades"][0]["nombre"], D["entidades"][0]["ruc"], "Compras o montos", escrito, "ana@ejemplo.pe"))
                                                    and p.locator("#c-abrir").get_attribute("href") == correos[0],
                                                    f"para {para.replace('mailto:', '')}; asunto «{asunto}»; {len(correos[0]) if correos else 0} caracteres en el enlace")
                p.click("#c-copiar")
                p.wait_for_timeout(200)
                copiado = p.locator("#c-copiar").inner_text() == "Copiado" or (p.locator("#c-texto").is_visible() and c3["prov"]["ruc"] in p.locator("#c-texto").input_value())
                p.goto(U + "#datos")
                p.wait_for_timeout(200)
                textos_a = aviso_c + p.evaluate("document.body.innerText") + p.evaluate("document.querySelector('script:not([type])').textContent")
                ca["la_pagina_ya_no_dice_que_no_envia"] = (BUZON in aviso_c and copiado and not re.search(r"no se envía|no envía", textos_a) and BUZON in p.locator('section[data-vista="datos"]').inner_text(),
                                                           f"aviso: «{aviso_c.splitlines()[0][:110]}»; el pedido se puede copiar: {copiado}")
                p.goto(U + "#caso-wayrapampa-3")
                p.wait_for_timeout(300)
                p.click("#abrir-cor")
                p.select_option("#c-tipo", "Compras o montos")
                p.fill("#c-det", escrito)
                p.fill("#c-mail", "ana@ejemplo.pe")
                p.click('#cor button[type="submit"]')
                p.wait_for_timeout(200)
                p.evaluate("() => { const c = document.getElementById('panel-cuerpo'); c.scrollTop = c.scrollHeight; }")
                p.locator("#panel").screenshot(path=str(CAP / "etapa_a_3_correo.png"))
                (CAP / "etapa_a_3_correo.txt").write_text(f"Para: {para.replace('mailto:', '')}\nAsunto: {asunto}\n\n{cuerpo_c}\n".replace("\r\n", "\n"), encoding="utf-8")
                res["etapa_a"] = {k_: {"ok": v[0], "detalle": v[1]} for k_, v in ca.items()}
            if nombre == "cel" and tema == "light":
                # en el celular, el botón de la carpeta cabe en la ficha
                p.goto(U + "#caso-wayrapampa-3", wait_until="networkidle")
                p.wait_for_timeout(300)
                bc = p.locator("[data-carpeta]")
                bc.scroll_into_view_if_needed()
                caja = bc.bounding_box()
                res["carpeta_cel"] = {"ancho": round(caja["width"]), "alto": round(caja["height"]), "cabe": caja["x"] >= 0 and caja["x"] + caja["width"] <= w and bc.evaluate("b => b.scrollWidth <= b.clientWidth")}
                p.locator("#panel").screenshot(path=str(CAP / "etapa_a_1_boton_cel.png"))
                # en el celular: tocar una tarjeta de entidad y una de proveedor abre lo que corresponde
                p.goto(U + "#inicio", wait_until="networkidle")
                p.locator("#t-indices tr").nth(1).tap(position={"x": 300, "y": 60})
                p.wait_for_timeout(300)
                abre_entidad = p.evaluate("location.hash")
                p.locator("#filas tr.fila").nth(2).tap(position={"x": 200, "y": 10})
                p.wait_for_timeout(400)
                res["toque_cel"] = {"tarjeta_entidad": abre_entidad, "tarjeta_proveedor": p.evaluate("location.hash"), "ficha": p.locator("#p-nombre").inner_text()}
            ctx.close()
    nav.close()

(CAP / "medidas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in res.items():
    if k in ("prueba_de_uso", "etapa_2", "etapa_a", "etapa_b"):
        print({"prueba_de_uso": "CAMBIOS DE LA PRUEBA DE USO", "etapa_2": "ETAPA 2: CONTRATO DE ALCANCE Y MARCA", "etapa_a": "ETAPA A: CARPETA DEL PROVEEDOR Y CORREO DE CORRECCIÓN",
               "etapa_b": "ETAPA B: CADA ÍNDICE FRENTE A LAS ENTIDADES DEL PAÍS"}[k])
        for c, r in v.items():
            print(f"  {'OK   ' if r['ok'] else 'FALLA'} {c}: {r['detalle']}")
        continue
    if k in ("interaccion", "toque_cel", "carpeta_cel"):
        print(k.upper(), json.dumps(v, ensure_ascii=False))
        continue
    print(k, "| pesos", v["pesos"], "| familias", v["familias"], "| tamaños", [x.replace("px", "") for x in v["tamanos"]], "| radios", [x.replace("px", "") for x in v["radios"]], "| desborde", v["desborde"])
    if v["objetivos_chicos"]:
        print("     chicos:", "; ".join(f"{x['t']} {x['w']}x{x['h']}" for x in v["objetivos_chicos"]))
print("ERRORES:", errores)
print("capturas en:", CAP)
