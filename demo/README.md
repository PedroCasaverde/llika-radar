# Demo de llika.pe

## Cómo verla

1. Descarga `llika_caso_real_2025.html` (en GitHub: abre el archivo y usa «Download raw file»).
2. Ábrelo con doble clic en cualquier navegador.

No hay que instalar nada ni tener servidor. Solo los enlaces a las resoluciones y a los informes de la Contraloría
necesitan internet.

## Qué trae

La pantalla real de LLIKA con tres municipalidades de Dos de Mayo (Huánuco): Ripán, Pachas y Marías, con sus compras
sin concurso de 2025 y lo que se sabía al 31 de diciembre de 2025. Son registros públicos del Estado. Aparecer en la
lista no constituye sanción, falta ni delito.

- **Inicio:** las tres entidades con sus seis índices, comparados con las 2 647 entidades del país.
- **Entidad:** sus proveedores en el orden sugerido o por monto, y el botón «Descargar en Excel».
- **Ficha de cada proveedor:** por qué está en ese puesto, sus señales con la norma, la fuente y el enlace al
  documento oficial, y sus compras. Ahí se confirma o se descarta con un motivo, se baja la carpeta del proveedor
  (un zip) y se puede pedir que se corrija un dato.

Las decisiones se guardan en el navegador de quien la usa, y cada entidad muestra sus 100 primeros proveedores.

## Cómo se rehace

`python llika.py caso-real` vuelve a escribir este archivo desde las tablas procesadas (unos 4 GB, fuera del
repositorio; se las pasamos a quien las pida). Los pasos están en el README principal. La página sale de
`fuente/plantilla.html` con `fuente/armar_demo.py`, y `fuente/probar_real.py` la prueba en Chromium.
