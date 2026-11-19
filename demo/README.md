# Demo de llika.pe

La demo es la pantalla real de LLIKA en un solo archivo HTML, sin servidor ni instalación: basta con abrir
`llika_caso_real_2025.html` en cualquier navegador. Trae tres municipalidades de Dos de Mayo (Huánuco), Ripán, Pachas y
Marías, con sus proveedores de 2025 y lo que se sabía al 31 de diciembre de 2025: registros públicos del Estado. Abre en
la lista de entidades; de ahí se entra a la lista de proveedores de una entidad y a la ficha de cada uno, con el enlace a
la resolución o al informe de cada señal. Desde la ficha se baja la carpeta del proveedor (un zip) y se pide por correo
que se corrija un dato. Es la misma pantalla del video y del pitch.

Se rehace con `python llika.py caso-real`, que necesita las tablas procesadas (unos 4 GB, fuera del repositorio: se las
pasamos a quien las pida). `datos_demo.json` es un juego de datos de prueba de la plantilla: de él salen las cifras del
método que lleva cada pantalla.

## Versiones

Hasta el 19/11/2026 el repositorio publicaba una demo con datos inventados (`llika_pe_demo.html`), armada con `fuente/datos_demo.json`. Desde entonces la demo es la pantalla real. Lo que sigue es el historial de la plantilla.

| Versión | SHA-256 de `llika_pe_demo.html` | Qué es |
|---|---|---|
| 13/11/2026 | `417c1381a1e64d50ad66800ad2df6f52193579baa71c67d9552d3e9e7bf5764f` | La que se usó en la prueba de uso con tres participantes (primer commit de esta carpeta) |
| 14/11/2026, prueba de uso | `81ab358bdd54b28cded3a5fa9c57b15cfa1562f4bfa94927ce04086b50cef8a8` | Los cinco cambios que salieron de esa prueba (segundo commit) |
| 14/11/2026, contrato de alcance | `bb261637ea9b1ada78f0f31db3c86b4bfcb266c135e763d329385e521321b464` | La marca de UTEC Posgrado y lo que faltaba del contrato de alcance |
| 14/11/2026, datos reales | `eeee4d24dea90ce835db87233952092709e5883f8cb2ec345e19f69ddf66c486` | La plantilla también sirve para datos reales: enlaces a la resolución o al informe de cada señal y avisos según el modo. En el inicio, una sola tabla de entidades con sus cifras y sus índices (antes eran dos tablas con las mismas entidades) |
| 15/11/2026, cifras del Tribunal | `f373b9e9966a0205afc91a4bf75b1496578a7e9cf1053d537778596bcd97e143` | «Cuánto cuesta equivocarse» con las cifras medidas de nuevo: S/ 613 millones a 2,477 proveedores hasta que rige su sanción, unos 30 meses del cierre del año a la sanción y la denuncia a los seis meses de los hechos |
| 15/11/2026, norma de las penalidades | `ed8943b3f17e0973bfe04f5c0c15469d6c2e29b11decaf09ed67c3ca9d75af02` | Antes del 22 abr 2025, las sanciones del Tribunal estaban en el art. 50 del TUO de la Ley 30225, y las penalidades, en su reglamento (D.S. 344-2018-EF); la versión anterior las juntaba bajo el art. 50 |
| 15/11/2026, auditoría | `3d707275e8ed881f0d0ed8d87020f40b5f47305744398e3723589fd43bb80357` | «Cómo funciona» usa las cifras del informe: 2.75 veces (antes decía 2.7) y de 42 a 63 horas en vano por acierto, frente a 116 a 174 por monto, a 40 o 60 minutos por revisión según la prueba de uso (antes, de 63 a 87 y de 147 a 198, con otro supuesto). «Datos y fuentes» dice cuántas resoluciones se leyeron (10,967) y cuántos informes se leyeron a fondo (2,847 de 14,915), y la regla del parentesco nombra también el número de personas con esos apellidos, que es lo que muestra la pantalla real |
| 17/11/2026, carpeta y corrección | `6e00697bc62837ba4104a9521b7e7591e190d620852741e7720a5ea86e06037b` | Cada ficha trae el botón «Descargar la carpeta de este proveedor», y el formulario de corrección deja escrito un correo al buzón del equipo en vez de no enviar nada. El detalle está en «Carpeta del proveedor y pedido de corrección» |
| 17/11/2026, índices frente al país | `5fb19c088121a9f3699bd6dd82242fecc8a4c6b22a7d2a34203bdf56e8bca1d3` | Cada índice de una entidad va con su lugar entre las 2 647 entidades de la lista de 2025 (su percentil): «más que el 33%», una barra que llega hasta ese lugar y color si está en el 10 % más alto. Antes se comparaba solo con las entidades de la página. El detalle está en «Índices frente a las 2 647 entidades» (esta versión) |

### Con datos reales

La misma plantilla arma la pantalla de entidades reales, con el JSON que deja `radar/v4/exportar_entidad.py`. Antes
va `radar/v4/indices_entidades.py`, que calcula los índices de las 2 647 entidades (`python llika.py caso-real` corre
los cuatro pasos):

```
python armar_demo.py --real ../../privado/datos_caso_real_2025.json
python probar_real.py ../../privado/llika_caso_real_2025.html ../../privado/datos_caso_real_2025.json ../../privado/capturas
```

Con `--real`, `armar_demo.py` exige lo contrario que en la demo: que todo RUC tenga dígito verificador válido. Además
exige que cada enlace a un documento sea de un sitio del Estado (gob.pe), y cambia los avisos de «datos inventados» por
los de datos reales. También exige que los índices de cada entidad se hayan calculado con la distribución de
`fuente/distribucion_indices_2025.json`, que es la que lleva la pantalla. Las demás reglas (palabras que no pasan,
números es-PE, colores y contraste) son las mismas. El JSON se escribe en `privado/`, que no va a git; con `caso-real`, la pantalla va a `demo/`.

`probar_real.py` abre la pantalla en Chromium a 1440 y 375 px, en claro y oscuro, y comprueba lo siguiente:

- que no haya errores de consola ni desborde;
- que el aviso diga «Datos reales» y que no quede ningún «inventado»;
- que, en todas las fichas, cada señal tenga su norma y su fuente con fecha, y cada documento del JSON tenga su enlace;
- los errores básicos: un enlace a una entidad que no está, un número que no es RUC y una búsqueda sin resultados;
- que el Excel baje;
- que la carpeta de cada proveedor baje con sus tres archivos, que el Excel de dentro abra y que cada documento del
  JSON esté como enlace en la ficha de la carpeta y en la hoja «Señales»;
- que el formulario de corrección deje escrito un correo al buzón del equipo, con el RUC en el asunto y en el cuerpo;
- que los seis índices de cada entidad sean los del JSON, calculados con todos sus proveedores, y que su lugar entre
  las 2 647 entidades (el texto, el largo de la barra y el color) sea el que la prueba recalcula en Python.

Con red, también pide cada enlace a gob.pe y anota su respuesta. Deja en la carpeta de capturas, de muestra, las
carpetas de los tres primeros proveedores de la primera entidad.

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
| (4) **Los índices por entidad** para compararlas | Seis índices (con vínculo, con antecedentes, compras partidas, cerca del tope, le compra a pocos y dependen de ella) en una tabla que se ordena en el inicio, y una franja en cada entidad. Desde el 17/11, cada índice se compara con las 2 647 entidades de la lista de 2025: ver «Índices frente a las 2 647 entidades» |

La marca: Space Grotesk en títulos y cifras, el carbón (#4D4B4C) y el cian (#00E1FF) de UTEC Posgrado, el logo de
llika.pe (una telaraña, que es lo que quiere decir *llika* en quechua), el de UTEC Posgrado, la mascota y las
ilustraciones del pitch. Se mantienen las reglas de la versión anterior: dos familias de letra y tres pesos, tamaños,
espacios y radios de una escala, contraste de 4,5:1 en el texto y toque de 44 px en el celular.

De paso, los números de compra inventados se repetían entre proveedores de una misma entidad; ahora cada entidad
numera en orden de fecha, con una serie para órdenes de compra y otra para órdenes de servicio
(`fuente/renumerar_compras.py`). No cambió ninguna fecha ni monto.

### Carpeta del proveedor y pedido de corrección

Al final de cada ficha, el botón «Descargar la carpeta de este proveedor» baja un zip que se arma en el navegador, sin
librerías ni conexión. Se llama `llika_carpeta_<RUC>_<entidad>_<año>.zip` y trae tres archivos:

| Archivo | Qué trae |
|---|---|
| `LÉEME.txt` | Qué es la carpeta, la fecha de corte de la lista, la fecha de la descarga, las fuentes con la fecha de sus datos y el buzón de correcciones. Con datos reales, también la huella de la lista |
| `ficha.html` | La ficha, en una página que se abre sola en cualquier navegador y se puede imprimir: el aviso de que no constituye sanción, falta ni delito, por qué está en ese puesto, cómo llegó a él, sus señales con norma y fuente, sus compras y, si la hay, la decisión de quien revisó |
| `compras_y_senales.xlsx` | La hoja «Compras», con las mismas columnas que el Excel de la lista, y la hoja «Señales»: una fila por señal y por documento, con su norma, su fuente, la fecha de esos datos y el enlace al documento oficial, que Excel abre con un clic |

Las resoluciones del Tribunal y los informes de control no van dentro del zip: van como enlace a la página del Estado
que los publica, porque un archivo HTML sin servidor no puede bajarlos de otro sitio. En la demo no hay enlaces, porque
los documentos son inventados.

El formulario «Reportar un error» ya no se queda en la página. Valida lo que se escribió y abre el programa de correo
con un mensaje ya escrito para `pedro.casaverde@utec.edu.pe`, el buzón del equipo: lleva el proveedor, su RUC, la
entidad, el puesto, el dato que se pide corregir y lo que escribió la persona. El pedido llega cuando la persona envía
ese correo. Si su navegador no abre ningún programa de correo, el botón «Copiar el pedido» deja el texto listo para
pegarlo en el que use. Va por correo y no por GitHub porque un pedido puede traer un DNI y el repositorio es público.

### Índices frente a las 2 647 entidades

Hasta el 17/11 la página calculaba los seis índices con los 100 primeros proveedores de cada entidad y los comparaba
solo con las entidades que traía. Ahora cada índice lleva su lugar entre las 2 647 entidades de la lista de 2025.

| Qué | Cómo quedó |
|---|---|
| De dónde salen los valores del país | `radar/v4/indices_entidades.py` calcula los seis índices de las 2 647 entidades con todos sus proveedores (1 007 366 casos y 3 378 148 compras de 2025) y las reglas de la ficha. Demora unos 9 segundos y pide 1,7 GB. No corre ningún modelo ni toca la lista congelada |
| Qué lleva la pantalla | `fuente/distribucion_indices_2025.json` (60 KB, huella `73485208c9d6a88a`): los 2 647 valores de cada índice, ordenados, en centésimas de punto porcentual y sin el RUC de ninguna entidad. `armar_demo.py` lo pone en la página, en la demo y con datos reales |
| El lugar | La parte de las entidades con un valor más bajo, que es el percentil. Se escribe «más que el 33%» y se trunca, no se redondea, para no decir de más. Si ninguna entidad tiene menos, dice «ninguna tiene menos»; si ninguna tiene más, «ninguna tiene más» |
| Cómo se ve | Bajo cada valor, una barra que llega hasta ese lugar, con una raya en la mitad. Lleva color solo si la entidad está en el 10 % más alto. En la página de la entidad, además, cuánto tiene la mitad de las entidades |
| Con datos reales | Los índices de la entidad llegan en el JSON, calculados con todos sus proveedores (176 en Ripán), no solo con los 100 que se muestran |
| En la demo | Las tres entidades son inventadas y sus índices se calculan en la página, con los proveedores de su lista. Se comparan con los valores reales de las 2 647 entidades, y la página lo dice: «Esta entidad es inventada [...] se comparan con las 2,647 entidades reales de la lista de 2025». Como se inventaron para mostrar todas las señales, quedan muy arriba en vínculo y antecedentes |

La página ya no anuncia un ajuste de las entidades chicas hacia el promedio. Se midió: en las entidades con más de 50
proveedores ese ajuste movería el lugar menos de 3 puntos (en la mediana), y en «Con vínculo» no aplica, porque 2 190
entidades tienen 0. En su lugar, la página avisa que en las entidades con 20 proveedores o menos (98 de 2 647) un proveedor más o
menos mueve mucho el porcentaje, y que «Le compra a pocos» sale más alto mientras menos proveedores tiene la entidad.

## Cómo se arma y se prueba

| Archivo | Qué es |
|---|---|
| `fuente/plantilla.html` | La página: estructura, estilos y código |
| `fuente/datos_demo.json` | Los datos inventados de las tres entidades |
| `fuente/distribucion_indices_2025.json` | Los valores reales de los seis índices en las 2 647 entidades de la lista de 2025, ordenados y sin RUC (lo arma `radar/v4/indices_entidades.py`) |
| `fuente/imagenes/` | Las imágenes en WebP: mascota, ilustraciones y logos |
| `fuente/armar_demo.py` | Junta la plantilla, los datos y las imágenes en `llika_pe_demo.html` (o, con `--real`, en la pantalla de datos reales) |
| `fuente/probar_demo.py` | Abre la demo en Chromium, la mide y comprueba cada cambio |
| `fuente/probar_real.py` | Lo mismo para la pantalla con datos reales, y los enlaces a los documentos |
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
norma y fecha en cada señal de las 296 fichas; los seis índices de la página iguales a los que recalcula en Python, y
también su lugar entre las 2 647 entidades (el texto, el largo de la barra y el color); que decidir pida motivo, siga al recargar, se filtre, salga en el Excel y se pueda quitar; y que todas las imágenes carguen
con su texto alternativo. De la carpeta del proveedor comprueba, en las 296 fichas, que el zip baje sin daño con sus
tres archivos, que el Excel de dentro abra con las compras y las señales de esa ficha, y que la ficha y el LÉEME lleven
el aviso y la fecha de corte; además, que la ficha de la carpeta abra sola sin errores y que lleve la decisión de quien
revisó. Del formulario comprueba que vacío no abra el correo y que, lleno, el correo vaya al buzón con el RUC, la
entidad, el dato y lo escrito. La prueba captura el enlace del correo: no abre el programa de correo de la
computadora. Deja las capturas, los Excel de prueba y tres carpetas de muestra en la carpeta que se le indique.

```
python capturas_etapa2.py ../llika_pe_demo.html capturas despues
```
