# Prueba la pantalla con datos reales en Chromium (la que arma «armar_demo.py --real»):
#   python probar_real.py <llika_X.html> <datos_X.json> [carpeta_de_capturas] [--sin-red]
# Comprueba: 0 errores de consola y 0 desborde (1440 y 375 px, claro y oscuro); el aviso de datos reales y ningún «inventado»;
# en todas las fichas, que cada señal traiga su norma y su fuente con fecha, y que cada documento del JSON tenga su enlace en la
# ficha; los errores básicos (enlace a una entidad que no está, un número que no es RUC, una búsqueda sin resultados); y el
# Excel. Con red, además pide cada enlace a gob.pe y reporta su respuesta. Deja capturas y un resumen JSON en la carpeta.
# De cada ficha baja la carpeta del proveedor (un zip): que traiga sus tres archivos, que el Excel de dentro abra y que
# cada documento del JSON esté como enlace en la ficha y en la hoja «Señales». Y que el formulario de corrección deje
# escrito un correo al buzón del equipo, con el RUC (la prueba captura el enlace: no abre el programa de correo).
# Y los índices: que la página muestre los del JSON (calculados con todos los proveedores de la entidad), que la
# distribución que lleva sea la de distribucion_indices_2025.json y que el lugar de cada índice entre las 2,647 entidades
# (texto, largo de la barra y color) sea el que se recalcula aquí.
import bisect
import hashlib
import html
import io
import json
import math
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) < 2:
    sys.exit("Uso: python probar_real.py <llika_X.html> <datos_X.json> [carpeta_de_capturas] [--sin-red]")
ARCHIVO, DATOS = Path(args[0]), Path(args[1])
CAP = Path(args[2]) if len(args) > 2 else Path(tempfile.gettempdir()) / "llika_real_capturas"
CON_RED = "--sin-red" not in sys.argv
for p in (ARCHIVO, DATOS):
    if not p.exists():
        sys.exit(f"ERROR: no existe {p}")
CAP.mkdir(parents=True, exist_ok=True)
D = json.loads(DATOS.read_text(encoding="utf-8"))
U = ARCHIVO.resolve().as_uri()
E0 = D["entidades"][0]["id"]


def esperados(c):
    """Enlaces a documentos que el JSON trae para un caso."""
    u = [x["url"] for x in c.get("exp") or [] if x.get("url")] + [o["url"] for x in c.get("exp") or [] for o in x.get("otros") or []]
    u += [x["url"] for xs in (c.get("ant") or {}).values() for x in xs if isinstance(x, dict) and x.get("url")]
    return u


BUZON = "pedro.casaverde@utec.edu.pe"
ARCHIVOS_CARPETA = ["LÉEME.txt", "ficha.html", "compras_y_senales.xlsx"]
INDICES = ["vin", "ant", "fra", "tope", "conc", "dep"]
DISTRIBUCION = Path(__file__).resolve().parent / "distribucion_indices_2025.json"


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


def fallas_carpeta(ruta, e, c):
    """Abre el zip de la carpeta de un proveedor y devuelve lo que no cumple (vacío si todo está bien)."""
    import openpyxl
    with zipfile.ZipFile(ruta) as z:
        nombres, danado = z.namelist(), z.testzip()
        if nombres != ARCHIVOS_CARPETA or danado:
            return [f"archivos: {nombres}; dañado: {danado}"]
        leeme, ficha = z.read("LÉEME.txt").decode("utf-8-sig"), html.unescape(z.read("ficha.html").decode("utf-8"))
        libro = openpyxl.load_workbook(io.BytesIO(z.read("compras_y_senales.xlsx")))
    if libro.sheetnames != ["Compras", "Señales"]:
        return [f"hojas: {libro.sheetnames}"]
    compras = [r for r in libro["Compras"].iter_rows(min_row=4, values_only=True) if r[0] is not None]
    senales = [r for r in libro["Señales"].iter_rows(min_row=4, values_only=True) if r[0] is not None]
    en_hoja = {m.group(1) for r in senales for m in [re.match(r'=HYPERLINK\("([^"]+)"', str(r[5] or ""))] if m}
    docs = esperados(c)
    return [nombre for nombre, bien in (
        ("compras", len(compras) == len(c["ord"]) and round(sum(r[6] for r in compras), 2) == round(sum(float(o[3] or 0) for o in c["ord"]), 2)),
        ("señales sin norma o sin fecha", all(r[2] and r[3] and r[4] for r in senales if r[0] != "Ninguna")),
        ("enlaces en la hoja Señales", set(docs) <= en_hoja),
        ("enlaces en la ficha", all(f'href="{u}"' in ficha for u in docs)),
        ("ficha", all(x in ficha for x in ("No constituye sanción, falta ni delito", "Registros públicos del Estado", c["prov"]["nombre"], c["prov"]["ruc"], c["por"], e["nombre"]))),
        ("léeme", all(x in leeme for x in (f"31 dic {D['anio']}", D["lista"], c["prov"]["ruc"], "no constituye sanción, falta ni delito", "registros públicos del Estado", BUZON))),
        ("dice inventado o demo", not re.search(r"inventad\w*|ficticia|\bdemo\b", leeme + re.sub(r"<table>.*?</table>", "", ficha, flags=re.S), re.I)),
        ("palabra «expediente»", "expediente" not in (leeme + " ".join(re.findall(r"<h[23]>(.*?)</h[23]>", ficha)) + " ".join(nombres)).lower()),
    ) if not bien]


ct, errores, t0 = {}, [], time.time()
with sync_playwright() as pw:
    nav = pw.chromium.launch()
    for ancho, tema in ((1440, "light"), (375, "light"), (375, "dark")):
        ctx = nav.new_context(viewport={"width": ancho, "height": 900}, color_scheme=tema, accept_downloads=True)
        p = ctx.new_page()
        p.on("console", lambda m: errores.append(f"{ancho}/{tema} {m.text}") if m.type == "error" else None)
        p.on("pageerror", lambda e: errores.append(f"{ancho}/{tema} {e}"))
        t_carga = time.time()
        p.goto(U + "#inicio", wait_until="load")
        ct.setdefault("carga_ms", {})[f"{ancho}_{tema}"] = round(1000 * (time.time() - t_carga))
        for nombre, h in (("inicio", "#inicio"), ("entidad", f"#entidad-{E0}"), *[(f"ficha{k}", f"#caso-{E0}-{k}") for k in range(1, 6)]):
            p.goto(U + h)
            p.wait_for_timeout(250)
            des = p.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            ct.setdefault("desborde", {})[f"{nombre}_{ancho}_{tema}"] = des
            if ancho == 1440 or nombre in ("inicio", "entidad", "ficha1"):
                if nombre.startswith("ficha"):
                    p.locator("#panel").screenshot(path=str(CAP / f"{nombre}_{ancho}_{tema}.png"))
                else:
                    p.screenshot(path=str(CAP / f"{nombre}_{ancho}_{tema}.png"), full_page=nombre == "entidad" and ancho > 400)
        if ancho == 1440:
            txt = p.evaluate("document.body.innerText")
            ct["aviso_datos_reales"] = "Datos reales." in p.locator(".aviso").inner_text()
            # ni «inventado» ni «ficticia» ni «demo», en ninguna vista (inicio, entidad y fichas)
            textos = [txt]
            for h in (f"#entidad-{E0}", f"#caso-{E0}-1", "#datos", "#como-funciona"):
                p.goto(U + h)
                p.wait_for_timeout(150)
                textos.append(p.evaluate("document.body.innerText"))
            sobra = [m.group(0) for t_ in textos for m in re.finditer(r"inventad\w*|ficticia|\bdemo\b", t_, re.I)]
            ct["sin_inventado"] = (not sobra, sobra[:5])
            # Los índices: los del JSON (con todos los proveedores de la entidad) y su lugar entre las entidades del país
            pais = json.loads(p.evaluate("document.getElementById('datos-llika').textContent")).get("pais") or {}
            dist = json.loads(DISTRIBUCION.read_text(encoding="utf-8")) if DISTRIBUCION.exists() else {}
            ct["distribucion_del_pais"] = (bool(dist) and pais.get("ind") == dist.get("indices") and pais.get("n") == dist.get("entidades") == 2647 and list(pais["ind"]) == INDICES
                                           and D.get("distribucion") == hashlib.sha256(DISTRIBUCION.read_bytes()).hexdigest()[:16],
                                           f"{pais.get('n')} entidades de la lista de {pais.get('anio')}; huella de la distribución {D.get('distribucion')}")
            p.goto(U + "#inicio")
            p.wait_for_timeout(300)
            celdas = p.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#t-indices tr')].map(tr => [tr.dataset.href.replace('#entidad-', ''),
              Object.fromEntries([...tr.querySelectorAll('.ind')].map(d => [d.dataset.k, { valor: parseFloat(d.querySelector('b').textContent), texto: d.querySelector('small').textContent.replace(/\\u00a0/g, ' '),
                barra: parseFloat(d.querySelector('.via span').style.getPropertyValue('--v')), alto: d.classList.contains('alto') }]))]))""")
            esperado = {e["id"]: {k: lugar_python(pais, k, e["ind"][k]) for k in INDICES} for e in D["entidades"]}
            dif_v = [(e["id"], k, celdas[e["id"]][k]["valor"], e["ind"][k]) for e in D["entidades"] for k in INDICES if abs(celdas[e["id"]][k]["valor"] - e["ind"][k]) > 0.051]
            dif_l = [(e, k, celdas[e][k], lugar_corto(l)) for e, ls in esperado.items() for k, l in ls.items()
                     if celdas[e][k]["texto"] != lugar_corto(l) or abs(celdas[e][k]["barra"] - l["pct"]) > 0.051 or celdas[e][k]["alto"] != (l["pct"] >= 90)]
            ct["indices_del_json"] = (not dif_v and len(celdas) == len(D["entidades"]) and all(e["n_ind"] >= len(e["casos"]) for e in D["entidades"]),
                                      f"{len(celdas)} entidades x 6 índices, con {', '.join(str(e['n_ind']) for e in D['entidades'])} proveedores; diferencias: {dif_v[:3]}")
            ct["lugar_igual_a_python"] = (not dif_l, f"texto, largo de la barra y color; diferencias: {dif_l[:3]}; {E0}: {[celdas[E0][k]['texto'] for k in INDICES]}")
            nota = p.locator("#nota-ind").inner_text()
            ct["nota_de_los_indices"] = ("todos los proveedores" in nota and "2,647 entidades de la lista de 2025" in nota and "percentil" in nota and not re.search(r"se ajustarán|sin ajustar|encogimiento", p.content()), nota[:120])
            caja_i = p.locator("section[data-vista='inicio'] .caja").bounding_box()
            p.screenshot(path=str(CAP / "etapa_b_indices.png"), clip={"x": 0, "y": caja_i["y"] + p.evaluate("scrollY") - 72, "width": ancho, "height": caja_i["height"] + 420}, full_page=True)
            p.goto(U + f"#entidad-{E0}")
            p.wait_for_timeout(300)
            fichas_ind = p.evaluate("""() => Object.fromEntries([...document.querySelectorAll('.compara > div')].map(d => [d.dataset.k, d.querySelector('small').textContent.replace(/\\u00a0/g, ' ')]))""")
            parrafo = p.locator("section[aria-labelledby='t-compara'] p").inner_text()
            ct["lugar_en_la_entidad"] = (list(fichas_ind) == INDICES and all(fichas_ind[k].startswith(lugar_largo(esperado[E0][k], pais["n"])) for k in INDICES)
                                         and "2,647 entidades del país" in p.locator("#t-compara").inner_text() and f"{D['entidades'][0]['n_ind']:,} proveedores" in parrafo.replace(" ", " "),
                                         f"«{fichas_ind.get('fra')}» | «{parrafo[:90]}…»")
            caja_c = p.locator("section[aria-labelledby='t-compara']").bounding_box()
            p.screenshot(path=str(CAP / "etapa_b_entidad.png"), clip={"x": 0, "y": max(0, caja_c["y"] + p.evaluate("scrollY") - 230), "width": ancho, "height": caja_c["height"] + 250}, full_page=True)
            # Todas las fichas: norma y fuente en cada señal, y los enlaces del JSON en la ficha
            fichas, sin_ref, faltan, abre_ms, carpetas_mal, n_carpetas = 0, [], [], [], [], 0
            (CAP / "carpetas").mkdir(exist_ok=True)
            for e in D["entidades"]:
                for c in e["casos"]:
                    t1 = time.time()
                    p.goto(U + f"#caso-{e['id']}-{c['rk']}")
                    p.wait_for_selector("#panel:not([hidden]) #p-nombre")
                    abre_ms.append(1000 * (time.time() - t1))
                    fichas += 1
                    n_sen = p.locator("#panel-cuerpo li.senal").count()
                    n_ref = p.locator("#panel-cuerpo li.senal dl.ref").count()
                    if n_sen != n_ref:
                        sin_ref.append(f"{e['id']}-{c['rk']}")
                    hrefs = set(p.eval_on_selector_all("#panel-cuerpo li.senal a[href^='https://']", "as => as.map(a => a.href)"))
                    falta = [u for u in esperados(c) if u not in hrefs]
                    if falta:
                        faltan.append(f"{e['id']}-{c['rk']}: {len(falta)}")
            # La carpeta de cada proveedor, en otra pasada: así las descargas no entran en el tiempo de abrir una ficha
            for e in D["entidades"]:
                for c in e["casos"]:
                    p.goto(U + f"#caso-{e['id']}-{c['rk']}")
                    p.wait_for_selector("#panel:not([hidden]) #p-nombre")
                    with p.expect_download() as baja:
                        p.click("[data-carpeta]")
                    z = CAP / "carpetas" / baja.value.suggested_filename
                    baja.value.save_as(str(z))
                    n_carpetas += 1
                    mal = fallas_carpeta(z, e, c) + ([] if z.name == f"llika_carpeta_{c['prov']['ruc']}_{e['id']}_{D['anio']}.zip" else ["nombre del zip"])
                    if mal:
                        carpetas_mal.append(f"{e['id']}-{c['rk']}: {', '.join(mal)}")
                    elif e["id"] != E0 or c["rk"] > 3:
                        z.unlink()  # quedan de muestra las tres primeras de la primera entidad, y las que fallen
            ct["carpetas_bajadas"] = n_carpetas
            ct["carpeta_con_sus_archivos_y_enlaces"] = (not carpetas_mal and n_carpetas == fichas, carpetas_mal[:5])
            # El pedido de corrección: un correo ya escrito al buzón, con el RUC (se captura el enlace, no se abre el correo)
            c1 = D["entidades"][0]["casos"][0]
            p.goto(U + f"#caso-{E0}-{c1['rk']}")
            p.wait_for_selector("#panel:not([hidden]) #p-nombre")
            p.evaluate("""() => { window.__correo = []; const real = HTMLAnchorElement.prototype.click;
              HTMLAnchorElement.prototype.click = function () { if (this.href.startsWith('mailto:')) { window.__correo.push(this.href); return; } return real.call(this); }; }""")
            escrito = "El DNI que figura no corresponde a esta persona."
            p.click("#abrir-cor")
            p.select_option("#c-tipo", "Nombre, RUC o DNI")
            p.fill("#c-det", escrito)
            p.click('#cor button[type="submit"]')
            p.wait_for_timeout(200)
            correos = p.evaluate("window.__correo")
            para, _, consulta = (correos[0] if correos else "").partition("?")
            campos = urllib.parse.parse_qs(consulta)
            asunto, cuerpo_c = campos.get("subject", [""])[0], campos.get("body", [""])[0]
            aviso_c = p.locator("#c-ok").inner_text()
            ct["correo_de_correccion"] = (len(correos) == 1 and para == f"mailto:{BUZON}" and c1["prov"]["ruc"] in asunto
                                          and all(x in cuerpo_c for x in (c1["prov"]["ruc"], c1["prov"]["nombre"], D["entidades"][0]["nombre"], D["entidades"][0]["ruc"], "Nombre, RUC o DNI", escrito))
                                          and not re.search(r"inventad\w*|\bdemo\b", cuerpo_c, re.I) and BUZON in aviso_c and not re.search(r"no se envía|no envía", aviso_c),
                                          f"para {para.replace('mailto:', '')}; el asunto y el cuerpo llevan el RUC: {c1['prov']['ruc'] in asunto and c1['prov']['ruc'] in cuerpo_c}")
            p.evaluate("() => { const c = document.getElementById('panel-cuerpo'); c.scrollTop = c.scrollHeight; }")
            p.locator("#panel").screenshot(path=str(CAP / "etapa_a_correo.png"))
            p.locator("[data-carpeta]").scroll_into_view_if_needed()
            p.locator("#panel").screenshot(path=str(CAP / "etapa_a_boton.png"))
            ct["fichas_abiertas"] = fichas
            ct["senales_con_norma_y_fuente"] = (not sin_ref, sin_ref[:5])
            ct["documentos_con_enlace_en_la_ficha"] = (not faltan, faltan[:5])
            ct["abrir_ficha_ms"] = {"mediana": round(sorted(abre_ms)[len(abre_ms) // 2]), "max": round(max(abre_ms))}
            # Errores básicos
            p.goto(U + "#entidad-no-existe")
            p.wait_for_timeout(200)
            ct["entidad_que_no_esta"] = p.locator("#t-404").is_visible()
            p.goto(U + "#inicio")
            b = p.locator("input[type=search]:visible").first
            b.fill("123456789012")
            p.wait_for_timeout(300)
            ct["numero_que_no_es_ruc"] = "11 dígitos" in p.evaluate("document.body.innerText")
            b.fill("zzzzqqq")
            p.wait_for_timeout(300)
            ct["busqueda_sin_resultados"] = "No encontramos" in p.evaluate("document.body.innerText")
            # Excel de la lista
            p.goto(U + f"#entidad-{E0}")
            p.wait_for_timeout(300)
            boton = p.get_by_role("button", name=re.compile("Excel"))
            if boton.count():
                with p.expect_download() as dl:
                    boton.first.click()
                f = CAP / f"lista_{E0}.xlsx"
                dl.value.save_as(f)
                ct["excel_bytes"] = f.stat().st_size
        ctx.close()
    nav.close()

ct["errores_de_consola"] = errores[:10]
ct["desborde_max_px"] = max(ct["desborde"].values())
urls = sorted({u for e in D["entidades"] for c in e["casos"] for u in esperados(c)})
ct["enlaces_unicos"] = len(urls)
if CON_RED:
    res = {}
    for u in urls:
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (prueba de enlaces)"}), timeout=25)
            res[u] = r.status
        except Exception as err:  # noqa: BLE001
            res[u] = getattr(err, "code", type(err).__name__)
        time.sleep(1)
    ct["respuesta_de_los_enlaces"] = {str(k): sum(1 for v in res.values() if v == k) for k in set(res.values())}
    ct["enlaces_que_no_responden_200"] = [f"{u} -> {v}" for u, v in res.items() if v != 200][:10]
ct["segundos"] = round(time.time() - t0)
(CAP / "resumen_prueba_real.json").write_text(json.dumps(ct, ensure_ascii=False, indent=1), encoding="utf-8")
ok = not errores and ct["desborde_max_px"] <= 0 and ct["aviso_datos_reales"] and ct["sin_inventado"][0] and ct["senales_con_norma_y_fuente"][0] \
    and ct["documentos_con_enlace_en_la_ficha"][0] and ct["entidad_que_no_esta"] and ct["numero_que_no_es_ruc"] and ct["busqueda_sin_resultados"] \
    and ct["carpeta_con_sus_archivos_y_enlaces"][0] and ct["correo_de_correccion"][0] \
    and ct["distribucion_del_pais"][0] and ct["indices_del_json"][0] and ct["lugar_igual_a_python"][0] and ct["nota_de_los_indices"][0] and ct["lugar_en_la_entidad"][0]
print(json.dumps(ct, ensure_ascii=False, indent=1))
print("RESULTADO:", "todo bien" if ok else "HAY FALLAS")
sys.exit(0 if ok else 1)
