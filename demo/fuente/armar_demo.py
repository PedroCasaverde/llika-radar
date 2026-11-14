# Arma la demo de llika.pe en un solo archivo: plantilla.html + datos_demo.json + mascota.txt.
#   python armar_demo.py            escribe ../llika_pe_demo.html
#   python armar_demo.py otro.html  escribe en otra ruta
# Los datos de la demo son inventados. Antes de escribir se comprueba que ningún RUC tenga dígito verificador válido,
# que el texto no use palabras que acusan y que el estilo no tenga colores fuera de los tokens.
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DESTINO = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent / "llika_pe_demo.html"

datos = json.loads((AQUI / "datos_demo.json").read_text(encoding="utf-8"))
mascota = (AQUI / "mascota.txt").read_text(encoding="utf-8").strip("\n")
t = (AQUI / "plantilla.html").read_text(encoding="utf-8")
assert t.count("__DATOS__") == 1 and t.count("__MASCOTA__") == 1
js = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
t = t.replace("__MASCOTA__", mascota).replace("__DATOS__", js)

# Texto que ve el usuario: la página sin la imagen, más los textos de los datos
visible = re.sub(r'<script type="application/json".*?</script>', "", t, flags=re.S)
visible = re.sub(r'src="data:image/[^"]+"', 'src=""', visible)
de_datos = []
for e in datos["entidades"]:
    for c in e["casos"]:
        de_datos.append(c["por"])
        de_datos += [x["x"] for x in c.get("exp") or []]
        if c.get("vin"):
            de_datos.append(c["vin"]["x"])
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
cuerpo = re.sub(r"<[^>]+>", " ", cuerpo).replace("&nbsp;", " ")
for texto in [cuerpo] + de_datos:
    m = re.search(r"\d[   ]\d{3}(?!\d)|\d,\d(?!\d{2})|\d[   ]%", texto)
    assert not m or "CLORURO" in texto, f"número fuera del formato es-PE: …{texto[max(0, m.start() - 30):m.end() + 10]}…"
assert "${NB}%" not in t and ".replace('.', ',')" not in t, "el JavaScript arma números fuera del formato es-PE"
assert not any("S/ " in x for x in de_datos) and "NB = ' '" in t, "entre S/ y el monto va un espacio que no corta la línea"
estilo = re.sub(r":root.*?\n}\n", "", visible.split("</style>")[0], flags=re.S).split("/* Base */")[1]
assert not re.search(r"#[0-9A-Fa-f]{6}\b", estilo), "hay un color suelto fuera de los tokens"

DESTINO.write_text(t, encoding="utf-8")
print("escrito:", DESTINO, f"({len(t):,} caracteres)")
