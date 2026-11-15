# Prueba la pantalla con datos reales en Chromium (la que arma «armar_demo.py --real»):
#   python probar_real.py <llika_X.html> <datos_X.json> [carpeta_de_capturas] [--sin-red]
# Comprueba: 0 errores de consola y 0 desborde (1440 y 375 px, claro y oscuro); el aviso de datos reales y ningún «inventado»;
# en todas las fichas, que cada señal traiga su norma y su fuente con fecha, y que cada documento del JSON tenga su enlace en la
# ficha; los errores básicos (enlace a una entidad que no está, un número que no es RUC, una búsqueda sin resultados); y el
# Excel. Con red, además pide cada enlace a gob.pe y reporta su respuesta. Deja capturas y un resumen JSON en la carpeta.
import json
import re
import sys
import tempfile
import time
import urllib.request
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
            # Todas las fichas: norma y fuente en cada señal, y los enlaces del JSON en la ficha
            fichas, sin_ref, faltan, abre_ms = 0, [], [], []
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
    and ct["documentos_con_enlace_en_la_ficha"][0] and ct["entidad_que_no_esta"] and ct["numero_que_no_es_ruc"] and ct["busqueda_sin_resultados"]
print(json.dumps(ct, ensure_ascii=False, indent=1))
print("RESULTADO:", "todo bien" if ok else "HAY FALLAS")
sys.exit(0 if ok else 1)
