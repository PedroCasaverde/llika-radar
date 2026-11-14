# Demo de llika.pe

La pantalla de LLIKA en un solo archivo HTML, sin servidor ni instalación: basta con abrir `llika_pe_demo.html` en
cualquier navegador. Abre en la lista de entidades; de ahí se entra a la lista de proveedores de una entidad y a la
ficha de cada uno.

Las tres entidades, sus proveedores, nombres, RUC y montos son **inventados** (ningún RUC tiene dígito verificador
válido). Las cifras del método sí son reales: cuántos aciertos tiene la lista en 2022 y 2024, las tasas por motivo y qué
pesa más en el parecido, calculadas con la lista de 2025 congelada.

## Versión

Es la versión del 13/11/2026 que se usó en la prueba de uso con tres participantes.
SHA-256 de `llika_pe_demo.html`: `417c1381a1e64d50ad66800ad2df6f52193579baa71c67d9552d3e9e7bf5764f`.

## Cómo se arma y se prueba

| Archivo | Qué es |
|---|---|
| `fuente/plantilla.html` | La página: estructura, estilos y código |
| `fuente/datos_demo.json` | Los datos inventados de las tres entidades |
| `fuente/mascota.txt` | La imagen de la mascota, en base64 |
| `fuente/armar_demo.py` | Junta los tres archivos en `llika_pe_demo.html` |
| `fuente/probar_demo.py` | Abre la demo en Chromium y la mide |

```
cd demo/fuente
python armar_demo.py
python probar_demo.py ../llika_pe_demo.html capturas
```

`armar_demo.py` no escribe nada si un RUC tiene dígito verificador válido, si aparece una palabra que acusa
(«sospechoso», «culpable» y parecidas), si un número no está en formato es-PE o si hay un color fuera de los tokens
del estilo. Se edita siempre la plantilla o los datos, nunca el HTML final.

`probar_demo.py` necesita Playwright (`pip install playwright` y `python -m playwright install chromium`). Recorre
nueve vistas a 1440 y 375 px, en claro y oscuro, y reporta errores de consola, desborde horizontal, familias y pesos de
letra, tamaños, radios y el área de toque en el celular. Deja las capturas en la carpeta que se le indique.
