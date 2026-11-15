# Prepara las imágenes de la demo en WebP, a dos veces el tamaño en que se ven, dentro de fuente/imagenes/.
#   python preparar_imagenes.py
# Origen: las ilustraciones y la mascota que se hicieron para el pitch del P1 (entregables/P1 - Pitch preliminar/P1_assets/_png/)
# y el logo oficial de UTEC Posgrado (de los documentos de la maestría). Solo hace falta correrlo si cambia una imagen:
# armar_demo.py usa lo que ya está en imagenes/.
from pathlib import Path

from PIL import Image

AQUI = Path(__file__).resolve().parent
P1 = AQUI.parents[1] / "P1 - Pitch preliminar" / "P1_assets" / "_png"
MARCA = AQUI.parents[3] / "deep-learning" / "tareas" / "paper2" / "video" / "estudio" / "marca"
SALIDA = AQUI / "imagenes"
SALIDA.mkdir(exist_ok=True)

# nombre en la demo: (archivo de origen, alto en píxeles de la WebP)
IMAGENES = {
    "mascota_lupa": (P1 / "mascota_04_eureka.png", 400),       # franja de inicio
    "mascota_dudas": (P1 / "mascota_01_problema.png", 240),    # página no encontrada
    "mascota_listo": (P1 / "mascota_06_pedido.png", 112),      # decisión guardada
    "mascota_saludo": (P1 / "mascota_00_portada.png", 240),    # sobre el proyecto
    "entidad": (P1 / "vec_02_entidad.png", 128),               # municipalidad y UGEL
    "hospital": (P1 / "vec_01_hospital.png", 128),             # hospital
    "red": (P1 / "vec_05_red_proveedor.png", 320),             # cómo funciona
    "fuentes": (P1 / "vec_06_datos_set.png", 240),             # datos y fuentes
    "utec": (MARCA / "logo-utec-posgrado.png", 64),            # logo sobre fondo claro
    "utec_blanco": (MARCA / "logo-utec-posgrado-blanco.png", 64),  # logo sobre carbón
}

faltan = [str(o) for o, _ in IMAGENES.values() if not o.exists()]
if faltan:
    # En el repositorio llika-radar no están los originales: las WebP de imagenes/ ya bastan para armar la demo
    raise SystemExit("No están las imágenes originales (se corre desde la carpeta del curso). Faltan:\n  " + "\n  ".join(faltan))
for nombre, (origen, alto) in IMAGENES.items():
    im = Image.open(origen).convert("RGBA")
    caja = im.getbbox()  # sin el margen transparente
    if caja:
        im = im.crop(caja)
    ancho = round(im.width * alto / im.height)
    im = im.resize((ancho, alto), Image.LANCZOS)
    destino = SALIDA / f"{nombre}.webp"
    im.save(destino, "WEBP", quality=82, method=6)
    print(f"{nombre:15s} {ancho}x{alto}  {destino.stat().st_size / 1024:.1f} KB")
