# Arma la demo de llika.pe en un solo archivo: plantilla.html + datos_demo.json + imagenes/*.webp.
#   python armar_demo.py            escribe ../llika_pe_demo.html
#   python armar_demo.py otro.html  escribe en otra ruta
# Los datos de la demo son inventados. Antes de escribir se comprueba que ningún RUC tenga dígito verificador válido,
# que ningún número de compra se repita en una entidad, que el texto no use palabras que acusan, que el estilo no tenga
# colores fuera de los tokens y que cada par de colores de texto cumpla el contraste (4.5:1; 3:1 en bordes y marcas).
import base64
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DESTINO = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent / "llika_pe_demo.html"

datos = json.loads((AQUI / "datos_demo.json").read_text(encoding="utf-8"))
t = (AQUI / "plantilla.html").read_text(encoding="utf-8")
assert t.count("__DATOS__") == 1 and t.count("__IMAGENES__") == 1

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
        if c.get("vin"):
            de_datos.append(c["vin"]["x"])
            # prueba de uso, punto 6: el vínculo declarado dice quién lo declaró
            if c["vin"]["tipo"] == "declarado":
                dj = c["vin"].get("decl") or {}
                assert all(dj.get(k) for k in ("nombre", "cargo", "area", "anio")), f"vínculo declarado sin quién lo declaró: {c['prov']['nombre']}"
                de_datos += [dj["nombre"], dj["cargo"], dj["area"]]
        for xs in (c.get("ant") or {}).values():
            de_datos += xs
        r = c["prov"]["ruc"]
        dv = 11 - sum(int(a) * b for a, b in zip(r[:10], [5, 4, 3, 2, 7, 6, 5, 4, 3, 2])) % 11
        dv = 0 if dv == 10 else 1 if dv == 11 else dv
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
