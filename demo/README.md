# Demo de llika.pe

La pantalla de LLIKA en un solo archivo HTML, sin servidor ni instalación: basta con abrir `llika_pe_demo.html` en
cualquier navegador. Abre en la lista de entidades; de ahí se entra a la lista de proveedores de una entidad y a la
ficha de cada uno.

Las tres entidades, sus proveedores, nombres, RUC y montos son **inventados** (ningún RUC tiene dígito verificador
válido). Son reales las cifras del método (cuántos aciertos tiene la lista en 2022 y 2024, las tasas por motivo y qué
pesa más en el parecido, calculadas con la lista de 2025 congelada), las normas que aplica cada señal y la fecha de los
datos de cada fuente.

## Versiones

| Versión | SHA-256 de `llika_pe_demo.html` | Qué es |
|---|---|---|
| 13/11/2026 | `417c1381a1e64d50ad66800ad2df6f52193579baa71c67d9552d3e9e7bf5764f` | La que se usó en la prueba de uso con tres participantes (primer commit de esta carpeta) |
| 14/11/2026, prueba de uso | `81ab358bdd54b28cded3a5fa9c57b15cfa1562f4bfa94927ce04086b50cef8a8` | Los cinco cambios que salieron de esa prueba (segundo commit) |
| 14/11/2026, contrato de alcance | `bb261637ea9b1ada78f0f31db3c86b4bfcb266c135e763d329385e521321b464` | La marca de UTEC Posgrado y lo que faltaba del contrato de alcance (esta versión) |

### Cambios de la prueba de uso

| Lo que se vio en la prueba | Qué cambió |
|---|---|
| El bloque «En qué se fija el programa» confundía: los porcentajes se leían como si fueran de esa empresa, y no se podían explicar a un lector | Ahora se llama «Qué datos de este proveedor mira el programa». Va en orden de importancia, sin porcentajes, con una frase que dice qué mide el parecido. Los pesos siguen en «Cómo funciona» |
| Dos participantes pidieron bajar la lista a Excel para trabajarla en la oficina | Botón «Descargar en Excel» en la lista de cada entidad: una hoja de proveedores y otra de compras, con lo que se ve en pantalla (filtros y orden incluidos). El archivo se arma en el navegador, sin librerías ni conexión |
| La tabla de compras estaba escondida bajo el gráfico | La tabla va abierta, antes del gráfico, y con el total |
| Se pidió un aviso visible contra la lectura como acusación, y destacar que el parentesco por apellidos no está confirmado | Cintillo arriba de cada ficha: «Ficha para orientar una revisión. No constituye sanción, falta ni delito». El vínculo por apellidos dice «Coincidencia de apellidos, sin confirmar con RENIEC», en letra más grande |
| Se pidió saber quién declaró al familiar, para pedir el legajo | Nombre, cargo y área del funcionario que lo declaró, y el año de la declaración (inventados, como todo en la demo) |

### Contrato de alcance y marca

El contrato de alcance del Informe Preliminar pide cuatro funcionalidades; esta versión completa las que faltaban.

| Funcionalidad del contrato | Qué hay en la demo |
|---|---|
| (2) La ficha de cada caso con **la norma que aplica y la fecha de cada fuente** | Cada señal de la ficha dice qué norma aplica (Ley 32069 desde el 22/04/2025 y el TUO de la Ley 30225 antes; Leyes 31227, 27785, 28175 y 31564) o que ninguna la prohíbe, y de qué fuente sale, con la fecha de sus datos. Lo mismo la tabla de compras. «Datos y fuentes» trae la tabla de fuentes con su fecha y la de normas |
| (3) **Confirmar o descartar cada caso con su motivo** y exportarlo a Excel | Barra fija abajo en cada ficha: confirmar para revisar o descartar, con un motivo de la lista y una nota (obligatoria si el motivo es «otro»), y la fecha. La lista muestra la decisión bajo el nombre del proveedor y se puede filtrar por ella. El Excel suma decisión, motivo, nota y fecha, más las normas y las fuentes con su fecha. En la demo, las decisiones se guardan solo en el navegador |
| (4) **Los índices por entidad** para compararlas | Seis índices (con vínculo, con antecedentes, compras partidas, cerca del tope, le compra a pocos y dependen de ella) en una tabla que se ordena en el inicio, y una franja en cada entidad que la compara con las demás. Con todas las entidades del país, los de las entidades chicas se ajustarán hacia el promedio; en la demo van sin ajustar |

La marca: Space Grotesk en títulos y cifras, el carbón (#4D4B4C) y el cian (#00E1FF) de UTEC Posgrado, el logo de
llika.pe (una telaraña, que es lo que quiere decir *llika* en quechua), el de UTEC Posgrado, la mascota y las
ilustraciones del pitch. Se mantienen las reglas de la versión anterior: dos familias de letra y tres pesos, tamaños,
espacios y radios de una escala, contraste de 4,5:1 en el texto y toque de 44 px en el celular.

De paso, los números de compra inventados se repetían entre proveedores de una misma entidad; ahora cada entidad
numera en orden de fecha, con una serie para órdenes de compra y otra para órdenes de servicio
(`fuente/renumerar_compras.py`). No cambió ninguna fecha ni monto.

## Cómo se arma y se prueba

| Archivo | Qué es |
|---|---|
| `fuente/plantilla.html` | La página: estructura, estilos y código |
| `fuente/datos_demo.json` | Los datos inventados de las tres entidades |
| `fuente/imagenes/` | Las imágenes en WebP: mascota, ilustraciones y logos |
| `fuente/armar_demo.py` | Junta la plantilla, los datos y las imágenes en `llika_pe_demo.html` |
| `fuente/probar_demo.py` | Abre la demo en Chromium, la mide y comprueba cada cambio |
| `fuente/capturas_prueba.py` y `fuente/capturas_etapa2.py` | Capturas para comparar dos versiones |
| `fuente/preparar_imagenes.py` | Rehace las WebP desde las imágenes originales del curso (solo si cambia una) |
| `fuente/renumerar_compras.py` | La renumeración de las compras, ya aplicada a `datos_demo.json` |

```
cd demo/fuente
python armar_demo.py
python probar_demo.py ../llika_pe_demo.html capturas
```

`armar_demo.py` no escribe nada si un RUC tiene dígito verificador válido, si se repite un número de compra en una
entidad, si aparece una palabra que acusa («sospechoso», «culpable» y parecidas), si un número no está en formato
es-PE, si hay un color fuera de los tokens del estilo o si un par de colores no llega al contraste mínimo (4,5:1 en
texto y 3:1 en bordes y marcas, en claro y en oscuro). Se edita siempre la plantilla o los datos, nunca el HTML final.

`probar_demo.py` necesita Playwright (`pip install playwright` y `python -m playwright install chromium`). Recorre
nueve vistas a 1440 y 375 px, en claro y oscuro, y reporta errores de consola, desborde horizontal, familias y pesos de
letra, tamaños, radios y el área de toque en el celular. Comprueba los cambios de la prueba de uso (porcentajes,
compras a la vista, aviso, coincidencia de apellidos, quién declaró, Excel completo y con filtros) y los del contrato:
norma y fecha en cada señal de las 296 fichas; los seis índices de la página iguales a los que recalcula en Python; que
decidir pida motivo, siga al recargar, se filtre, salga en el Excel y se pueda quitar; y que todas las imágenes carguen
con su texto alternativo. Deja las capturas y los Excel de prueba en la carpeta que se le indique.

```
python capturas_etapa2.py ../llika_pe_demo.html capturas despues
```
