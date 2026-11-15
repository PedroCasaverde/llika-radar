# Renumera las compras inventadas de la demo para que ningún número se repita en una entidad (HALLAZGOS 160, sección 93).
#   python renumerar_compras.py
# Los números salieron al azar y se repetían entre proveedores (69 en Wayrapampa, 80 en Ccori Llacta y 56 en Pukyu), algo
# que un auditor notaría. Ahora cada entidad numera en orden de fecha, con una serie para las órdenes de compra (O/C) y otra
# para las de servicio (O/S), como en el SIAF. Hay saltos entre un número y el siguiente porque la demo muestra solo los
# 100 primeros proveedores y la entidad emitió más órdenes. No cambia ninguna fecha, descripción ni monto.
import json
from pathlib import Path

ARCHIVO = Path(__file__).resolve().parent / "datos_demo.json"
d = json.loads(ARCHIVO.read_text(encoding="utf-8"))
for e in d["entidades"]:
    ords = [(o[0], c["rk"], k, o) for c in e["casos"] for k, o in enumerate(c["ord"])]
    salto = e["ordenes"] / len(ords)  # órdenes de la entidad por cada orden que muestra la demo (>= 1)
    for tipo in ("O/C", "O/S"):
        serie = sorted(x for x in ords if x[3][1].startswith(tipo))
        for i, (_, _, _, o) in enumerate(serie):
            o[1] = f"{tipo} {int(i * salto) + 1:04d}"
    nums = [o[1] for c in e["casos"] for o in c["ord"]]
    assert len(nums) == len(set(nums)), e["id"]
    print(e["id"], len(nums), "compras; último número", max(nums))
ARCHIVO.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
