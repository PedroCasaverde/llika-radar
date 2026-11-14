# Demo de llika.pe

La pantalla de LLIKA en un solo archivo HTML, sin servidor ni instalación: basta con abrir `llika_pe_demo.html` en
cualquier navegador. Abre en la lista de entidades; de ahí se entra a la lista de proveedores de una entidad y a la
ficha de cada uno.

Las tres entidades, sus proveedores, nombres, RUC y montos son **inventados** (ningún RUC tiene dígito verificador
válido). Las cifras del método sí son reales: cuántos aciertos tiene la lista en 2022 y 2024, las tasas por motivo y qué
pesa más en el parecido, calculadas con la lista de 2025 congelada.

## Versión

La prueba de uso con tres participantes se hizo con la versión del 13/11/2026, que es el primer commit de esta carpeta
(SHA-256 de `llika_pe_demo.html`: `417c1381a1e64d50ad66800ad2df6f52193579baa71c67d9552d3e9e7bf5764f`). Esta versión
trae los cambios que salieron de esa prueba
(SHA-256: `81ab358bdd54b28cded3a5fa9c57b15cfa1562f4bfa94927ce04086b50cef8a8`).

| Lo que se vio en la prueba | Qué cambió |
|---|---|
| El bloque «En qué se fija el programa» confundía: los porcentajes se leían como si fueran de esa empresa, y no se podían explicar a un lector | Ahora se llama «Qué datos de este proveedor mira el programa». Va en orden de importancia, sin porcentajes, con una frase que dice qué mide el parecido. Los pesos siguen en «Cómo funciona» |
| Dos participantes pidieron bajar la lista a Excel para trabajarla en la oficina | Botón «Descargar en Excel» en la lista de cada entidad: una hoja de proveedores y otra de compras, con lo que se ve en pantalla (filtros y orden incluidos). El archivo se arma en el navegador, sin librerías ni conexión |
| La tabla de compras estaba escondida bajo el gráfico | La tabla va abierta, antes del gráfico, y con el total |
| Se pidió un aviso visible contra la lectura como acusación, y destacar que el parentesco por apellidos no está confirmado | Cintillo arriba de cada ficha: «Ficha para orientar una revisión. No constituye sanción, falta ni delito». El vínculo por apellidos dice «Coincidencia de apellidos, sin confirmar con RENIEC», en letra más grande |
| Se pidió saber quién declaró al familiar, para pedir el legajo | Nombre, cargo y área del funcionario que lo declaró, y el año de la declaración (inventados, como todo en la demo) |

## Cómo se arma y se prueba

| Archivo | Qué es |
|---|---|
| `fuente/plantilla.html` | La página: estructura, estilos y código |
| `fuente/datos_demo.json` | Los datos inventados de las tres entidades |
| `fuente/mascota.txt` | La imagen de la mascota, en base64 |
| `fuente/armar_demo.py` | Junta los tres archivos en `llika_pe_demo.html` |
| `fuente/probar_demo.py` | Abre la demo en Chromium, la mide y comprueba cada cambio de la prueba de uso |
| `fuente/capturas_prueba.py` | Saca las capturas de las partes que cambió la prueba, para comparar dos versiones |

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
letra, tamaños, radios y el área de toque en el celular. Además comprueba los cambios de la prueba de uso: que ninguna
ficha muestre porcentajes en el bloque del parecido, que la tabla de compras se vea sin hacer clic, que el aviso esté
arriba en las 296 fichas, que la coincidencia de apellidos mida 16 px, que las 9 fichas con vínculo declarado digan
quién lo declaró, y que el Excel baje completo (100 proveedores, RUC como texto, la suma de las compras igual al total)
y respete los filtros. Deja las capturas y el Excel de prueba en la carpeta que se le indique.

```
python capturas_prueba.py ../llika_pe_demo.html capturas despues
```
