# Arma la pantalla de llika.pe en un solo archivo: plantilla.html + datos + imagenes/*.webp.
#   python armar_demo.py                         la demo con datos inventados: escribe ../llika_pe_demo.html
#   python armar_demo.py otro.html               la demo, en otra ruta
#   python armar_demo.py --real datos.json [salida.html]
#                                                con datos reales (radar/v4/exportar_entidad.py); por defecto escribe
#                                                llika_<nombre>.html junto al JSON, en privado/, fuera de git
# Antes de escribir se comprueba que ningún número de compra se repita en una entidad, que el texto no use palabras que
# acusan, que el estilo no tenga colores fuera de los tokens y que cada par de colores de texto cumpla el contraste (4.5:1;
# 3:1 en bordes y marcas). En la demo, ningún RUC puede tener dígito verificador válido; con datos reales, todos deben
# tenerlo, y cada enlace a un documento debe ser de un sitio del Estado (gob.pe). La página trae la carpeta del proveedor,
# que nunca se llama «expediente», y el pedido de corrección sale por correo.
# Los índices de cada entidad se comparan con los de las 2 647 entidades de la lista de 2025, que van en
# distribucion_indices_2025.json (lo arma radar/v4/indices_entidades.py; son valores ordenados, sin el RUC de ninguna
# entidad). Con datos reales, el JSON dice con qué distribución se calcularon sus índices y aquí se exige que sea esa.
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REAL = len(sys.argv) > 2 and sys.argv[1] == "--real"
if len(sys.argv) > 1 and sys.argv[1] == "--real" and not REAL:
    sys.exit("ERROR: falta el JSON. Uso: python armar_demo.py --real datos.json [salida.html]")
if REAL:
    ORIGEN = Path(sys.argv[2])
    if not ORIGEN.exists():
        sys.exit(f"ERROR: no existe {ORIGEN}. Se arma con radar/v4/exportar_entidad.py (python llika.py entidad).")
    DESTINO = Path(sys.argv[3]) if len(sys.argv) > 3 else ORIGEN.with_name(ORIGEN.stem.replace("datos_", "llika_") + ".html")
else:
    ORIGEN = AQUI / "datos_demo.json"
    DESTINO = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent / "llika_pe_demo.html"

try:
    datos = json.loads(ORIGEN.read_text(encoding="utf-8"))
except json.JSONDecodeError as err:
    sys.exit(f"ERROR: {ORIGEN.name} no es un JSON válido ({err}).")
assert (datos.get("modo") == "real") == REAL, "con --real van datos reales (modo «real»), y sin él, los de la demo"

# La distribución de los índices en el país: la misma para la demo y para los datos reales
INDICES = ["vin", "ant", "fra", "tope", "conc", "dep"]
ruta_dist = AQUI / "distribucion_indices_2025.json"
if not ruta_dist.exists():
    sys.exit(f"ERROR: falta {ruta_dist.name} junto a la plantilla. Lo arma radar/v4/indices_entidades.py (python llika.py indices).")
huella_dist = hashlib.sha256(ruta_dist.read_bytes()).hexdigest()[:16]
dist = json.loads(ruta_dist.read_text(encoding="utf-8"))
assert list(dist["indices"]) == INDICES and all(len(v) == dist["entidades"] and v == sorted(v) and 0 <= v[0] and v[-1] <= 10000 for v in dist["indices"].values()), "la distribución de los índices no tiene la forma esperada"
if REAL:
    if datos.get("distribucion") != huella_dist:
        sys.exit(f"ERROR: los índices de {ORIGEN.name} se calcularon con otra distribución ({datos.get('distribucion')}) que la de {ruta_dist.name} ({huella_dist}). "
                 "Corre indices_entidades.py y exportar_entidad.py de nuevo (python llika.py caso-real); si los datos cambiaron, copia aquí la distribución nueva.")
    assert all(list(e.get("ind") or {}) == INDICES and e.get("n_ind", 0) >= len(e["casos"]) for e in datos["entidades"]), "falta algún índice de una entidad, calculado con todos sus proveedores"
else:
    assert "pais" not in datos and not any("ind" in e for e in datos["entidades"]), "en la demo, los índices se calculan en la página con los proveedores inventados"
datos["pais"] = {"n": dist["entidades"], "anio": dist["anio"], "ind": dist["indices"]}

t = (AQUI / "plantilla.html").read_text(encoding="utf-8")
assert t.count("__DATOS__") == 1 and t.count("__IMAGENES__") == 1

# Con datos reales, los textos fijos de la página dejan de decir que todo es inventado
if REAL:
    n = len(datos["entidades"])
    for viejo, nuevo in (
        ("<meta name=\"description\" content=\"Demo de llika.pe:", "<meta name=\"description\" content=\"llika.pe:"),
        ("Los casos son inventados.\">", "Con registros públicos del Estado.\">"),
        ("<b>Demo.</b> Las entidades, los proveedores, los RUC, los DNI y los montos son inventados. Las cifras del método, las normas y las fechas de las fuentes son las reales de LLIKA.",
         f"<b>Datos reales.</b> Registros públicos del Estado, con lo que se sabía al 31 dic {datos['anio']}, cruzados por LLIKA. Es un orden sugerido para revisar: aparecer aquí no constituye sanción, falta ni delito."),
        ("<noscript><p>Esta demo necesita JavaScript.</p></noscript>", "<noscript><p>Esta página necesita JavaScript.</p></noscript>"),
        ("<p class=\"chico\">En esta demo, las entidades, los proveedores, los RUC, los DNI y los montos son inventados; los RUC llevan un dígito verificador inválido para que no coincidan con ninguno real.</p>",
         "<p class=\"chico\">Esta página muestra datos reales: nombres, RUC, DNI y montos tal como los publica el Estado. El posible parentesco es una coincidencia de apellidos, no un hecho comprobado, y cualquier persona puede pedir que se corrija un dato.</p>"),
        ("El enlace puede estar incompleto, o esa entidad no está en la demo. En la demo hay tres entidades.", "El enlace puede estar incompleto, o esa entidad no está en esta página."),
    ):
        assert t.count(viejo) == 1, f"no se encontró una sola vez el texto de la demo: {viejo[:60]}…"
        t = t.replace(viejo, nuevo)

# Imágenes: una sola vez en el archivo, como data URI; deben estar todas las que pide la plantilla
imagenes = {p.stem: "data:image/webp;base64," + base64.b64encode(p.read_bytes()).decode("ascii") for p in sorted((AQUI / "imagenes").glob("*.webp"))}
pedidas = set(re.findall(r'data-img="([a-z_]+)"', t)) | {"entidad", "hospital"}
assert pedidas <= set(imagenes), f"faltan imágenes: {sorted(pedidas - set(imagenes))}"
js = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
t = t.replace("__DATOS__", js).replace("__IMAGENES__", json.dumps(imagenes, separators=(",", ":")))

# Texto que ve el usuario: la página sin las imágenes, más los textos de los datos
visible = re.sub(r'<script type="application/json".*?</script>', "", t, flags=re.S)
visible = re.sub(r'src="data:image/[^"]+"', 'src=""', visible)
de_datos = []
for e in datos["entidades"]:
    numeros = [o[1] for c in e["casos"] for o in c["ord"]]
    assert len(numeros) == len(set(numeros)), f"números de compra repetidos en {e['id']}"
    for c in e["casos"]:
        de_datos.append(c["por"])
        de_datos += [x["x"] for x in c.get("exp") or []]
        de_datos += [x["nota"] for x in c.get("exp") or [] if x.get("nota")]
        enlaces = [x.get("url") for x in c.get("exp") or []] + [o["url"] for x in c.get("exp") or [] for o in x.get("otros") or []] \
            + [x.get("url") for xs in (c.get("ant") or {}).values() for x in xs if isinstance(x, dict)]
        for u in filter(None, enlaces):
            assert re.match(r"^https://[a-z0-9.-]+\.gob\.pe/", u), f"enlace que no es de un sitio del Estado: {u}"
        if c.get("vin"):
            de_datos.append(c["vin"]["x"])
            # prueba de uso, punto 6: el vínculo declarado dice quién lo declaró
            if c["vin"]["tipo"] == "declarado":
                dj = c["vin"].get("decl") or {}
                assert all(dj.get(k) for k in ("nombre", "cargo", "area", "anio")), f"vínculo declarado sin quién lo declaró: {c['prov']['nombre']}"
                de_datos += [dj["nombre"], dj["cargo"], dj["area"]]
        for xs in (c.get("ant") or {}).values():
            de_datos += [x if isinstance(x, str) else x["x"] for x in xs] + [x["nota"] for x in xs if isinstance(x, dict) and x.get("nota")]
        r = c["prov"]["ruc"]
        dv = 11 - sum(int(a) * b for a, b in zip(r[:10], [5, 4, 3, 2, 7, 6, 5, 4, 3, 2])) % 11
        dv = 0 if dv == 10 else 1 if dv == 11 else dv
        if REAL:
            assert int(r[10]) == dv, f"RUC real con dígito verificador inválido: {r}"
        else:
            assert int(r[10]) != dv, f"RUC con dígito verificador válido: {r}"
for m in datos["motivos"].values():
    de_datos += [m.get("nombre", ""), m.get("explica", "")]
todo = (visible + "\n" + "\n".join(de_datos)).lower()
for mala in ("sospech", "riesgos", "fraccionamiento", "culpable", "corrup", " vos ", "acá", "capstone"):
    assert mala not in todo, f"palabra que no debe ir en la demo: {mala}"
# Números en formato es-PE (S/ 56,650; 3.8; 28.9%): ni espacio de miles, ni coma decimal, ni espacio antes de %
cuerpo = re.sub(r"<(style|script)\b.*?</\1>", "", visible, flags=re.S)
cuerpo = re.sub(r"<[^>]+>", " ", cuerpo).replace("&nbsp;", " ")
for texto in [cuerpo] + de_datos:
    m = re.search(r"\d[ \u00a0\u202f]\d{3}(?!\d)|\d,\d(?!\d{2})|\d[ \u00a0\u202f]%", texto)
    assert not m or "CLORURO" in texto, f"número fuera del formato es-PE: …{texto[max(0, m.start() - 30):m.end() + 10]}…"
assert "${NB}%" not in t and ".replace('.', ',')" not in t, "el JavaScript arma números fuera del formato es-PE"
assert not any("S/ " in x for x in de_datos) and "NB = '\u00a0'" in t, "entre S/ y el monto va un espacio que no corta la línea"
# prueba de uso, punto 1: en la ficha, lo que mira el programa va en orden y sin porcentajes
assert "del parecido</span>" not in t and "de más a menos importante" in t, "la ficha volvió a mostrar porcentajes del parecido"
# etapa A del cierre: la carpeta del proveedor nunca se llama «expediente», y el pedido de corrección sale por correo
assert "Descargar la carpeta de este proveedor" in t and "expediente" not in visible.lower(), "la carpeta del proveedor no está, o se la llama «expediente»"
assert "mailto:${BUZON}" in t and "no se envía nada" not in t, "el formulario de corrección volvió a decir que no envía"
# etapa B del cierre: cada índice va con su lugar entre las entidades del país, y la página ya no promete un ajuste que no hace
assert 'id="nota-ind"' in t and "PAIS.ind[k]" in t and "se ajustarán" not in t and "sin ajustar" not in t, "los índices no se comparan con las entidades del país, o la página promete un ajuste"
estilo = re.sub(r":root.*?\n}\n", "", visible.split("</style>")[0], flags=re.S).split("/* Base */")[1]
assert not re.search(r"#[0-9A-Fa-f]{6}\b", estilo), "hay un color suelto fuera de los tokens"


# Contraste con los colores finales (WCAG 2.x), en claro y en oscuro; el oscuro hereda lo que no redefine
def tokens(bloque):
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9A-Fa-f]{6})", bloque))


def luz(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contraste(a, b):
    la, lb = sorted((luz(a), luz(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


claro = tokens(t[t.index(":root {"):t.index("\n}\n", t.index(":root {"))])
oscuro = {**claro, **tokens(t[t.index(':root[data-theme="dark"] {'):t.index("\n}\n", t.index(':root[data-theme="dark"] {'))])}
TEXTO = [("text", "bg"), ("text", "surface"), ("text", "surface-2"), ("text", "signal-soft"), ("text", "action-soft"), ("text-muted", "bg"), ("text-muted", "surface"),
         ("text-muted", "surface-2"), ("text-subtle", "bg"), ("text-subtle", "surface"), ("action", "bg"), ("action", "surface"), ("action-hover", "action-soft"),
         ("signal", "bg"), ("signal", "signal-soft"), ("on-cian", "cian"), ("on-cian", "cian-hover"), ("brand-text", "brand"), ("brand-muted", "brand"), ("cian", "brand")]
MARCAS = [("border-strong", "bg"), ("mark", "bg"), ("focus", "bg"), ("cian", "brand"), ("signal-mark", "bg")]
contrastes = {}
for nombre, tk in (("claro", claro), ("oscuro", oscuro)):
    for pares, minimo in ((TEXTO, 4.5), (MARCAS, 3.0)):
        for a, b in pares:
            r = contraste(tk[a], tk[b])
            contrastes[f"{nombre} {a}/{b}"] = round(r, 2)
            assert r >= minimo, f"contraste bajo en {nombre}: --{a} sobre --{b} da {r:.2f}:1 (mínimo {minimo}:1)"

DESTINO.write_text(t, encoding="utf-8")
peor = min(contrastes.items(), key=lambda x: x[1])
print("escrito:", DESTINO, f"({len(t):,} caracteres; {len(imagenes)} imágenes; contraste más bajo: {peor[0]} {peor[1]}:1)")
