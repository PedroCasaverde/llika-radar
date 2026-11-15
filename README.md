# LLIKA: radar de vínculos e impedimentos en las compras menores

Proyecto del curso Proyecto Integrador I de la Maestría en Ciencia de Datos e Inteligencia Artificial de UTEC Posgrado.
Equipo 05: Pedro Casaverde, Gian Carlo Quiroz y Sherly Salazar.

LLIKA cruza las compras menores del Estado (hasta 8 UIT, sin concurso) con registros públicos del OECE, la Contraloría,
el JNE y SUNAT. Para cada entidad y año sugiere por qué proveedores conviene empezar a revisar: primero los que tienen un
vínculo con una autoridad de la entidad, después los que más se parecen a proveedores que luego fueron sancionados y más
dependen de esa entidad, y al final la compra más grande que parece partida. Cada ficha muestra de dónde sale cada dato,
la norma que aplica y, cuando existe, el enlace a la resolución del Tribunal o al informe de control. Es una propuesta
académica y pública: aparecer en la lista no es una acusación.

## Qué hay aquí

| Carpeta o archivo | Qué es |
|---|---|
| `llika.py` | La entrada: revisa el entorno y corre cada paso, con su tiempo y su memoria |
| `demo/` | La pantalla (`fuente/plantilla.html`), cómo se arma y se prueba, y la demo con datos inventados |
| `radar/v4/` | La cadena vigente: de las tablas a la lista, los puntajes de 2025 y los datos de una entidad para la pantalla |
| `radar/` | Las alertas de vínculo e impedimento y las tablas que usa la cadena |
| `datos/_scripts/`, `datos/crudos/acc8_informes_cgr/`, `ocr_informes/` | Descarga de las fuentes, lectura de documentos (OCR y modelos de lenguaje) y su validación |
| `docs/` | Model Card v0, datasheet de los datos y la lista de archivos del paquete de datos |

Los datos no están en el repositorio: pesan varios GB y traen nombres, RUC y DNI. La pantalla con datos reales se arma en
`privado/`, que tampoco va a git.

## Entorno

Probado con Python 3.13.12 en Windows 11 (12 núcleos, 48 GB de RAM).

```
python -m venv .venv
.venv\Scripts\activate            (en Linux o Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m playwright install chromium
python llika.py revisar
```

`revisar` dice qué versión de Python y qué librerías hay, cuánta memoria y disco, qué datos faltan, qué paso los arma y si
la lista congelada tiene su huella. `requirements.txt` fija las versiones con las que la lista de 2025 se rehízo idéntica;
`requirements-fuentes.txt` suma lo necesario para leer los documentos desde cero (ruta 3).

**Semillas.** Los dos modelos del parecido promedian 10 árboles de LightGBM con las semillas 0 a 9 y 12 hilos
(`radar/v4/m6_proveedor.py`); la lista al azar y los intervalos por remuestreo usan `numpy.random.default_rng(0)`
(`radar/v4/lista_unica.py`). Con eso, `python llika.py puntajes` rehace la lista de 2025 idéntica byte por byte en la PC
del equipo. En otra máquina no está probado: los árboles con varios hilos podrían mover algún empate. Si pasa, el paso se
detiene y lo dice.

## Tres formas de correrlo

### 1. La demo, sin datos (un minuto)

```
python llika.py demo
```

Arma `demo/llika_pe_demo.html` con tres entidades inventadas (ningún RUC tiene un dígito verificador válido) y la prueba en
Chromium. Se abre con cualquier navegador, sin servidor. El detalle está en `demo/README.md`.

### 2. El caso real, con las tablas procesadas (unos 2 minutos y 8 GB de RAM)

Las tablas procesadas no se publican en ningún enlace porque traen nombres y DNI: el equipo las entrega en persona a quien
las pida. Son 39 archivos, unos 4 GB, y van en `datos/` con la misma estructura de carpetas; la lista está en
`docs/paquete_de_datos.md`. Luego:

```
python llika.py revisar          comprueba que estén y que la lista tenga su huella
python llika.py caso-real        Ripán, con Pachas y Marías para comparar: datos, pantalla y prueba
```

Para comprobar además que la lista de 2025 se reproduce, `python llika.py puntajes` la rehace con los mismos modelos y
se detiene si no sale idéntica byte por byte. Ese paso es opcional: pide unos 7 minutos, 24 GB de RAM y los archivos de
ajuste que también lista `docs/paquete_de_datos.md`.

La pantalla queda en `privado/llika_caso_real_2025.html`. Para otra entidad:

```
python llika.py entidad 20144801459 [más RUC]
python llika.py pantalla privado/datos_ripan_2025.json
python llika.py probar privado/llika_ripan_2025.html privado/datos_ripan_2025.json
```

### Probado desde un clon limpio

El 14/11/2026 se clonó el repositorio desde GitHub en otra carpeta (en la misma PC del equipo), se creó un entorno nuevo
solo con `requirements.txt` y se siguieron los pasos de arriba:
- `python llika.py demo` armó la misma demo y pasó sus pruebas;
- con el paquete de datos, `python llika.py caso-real` dio el mismo JSON (huella `e9de3840bf7a252d`) y la misma pantalla,
  y `probar_real.py` pasó entero;
- `python llika.py puntajes` rehízo la lista de 2025 idéntica byte por byte y el mismo archivo de puntajes
  (`79d655736b29bd46`).

### 3. Desde las fuentes (horas, 48 GB de RAM y unos 30 GB de disco)

Todas las fuentes son públicas. Este es el orden en que se armó la data; esta ruta todavía no se repitió completa desde un
clon limpio.

| Etapa | Scripts, en orden | Paso manual o con costo |
|---|---|---|
| 1. Descargas | `datos/_scripts/bajar_fuentes.py`; `bajar_historia_wiki.py` (sanciones, multas, penalidades e inhabilitaciones judiciales, una versión por mes); `bajar_historia_conformacion.py` (socios, un corte por mes); `radar/v4/u1_portal.py bajar` y `parquet`; `perucompras_ocds.py`; `tce_resoluciones.py indice`, `bajar` y `texto`, y `tce_faltantes.py`; en `datos/crudos/acc8_informes_cgr/`, `bajar_fichas.py`, `agregar_informes_E1.py` y `bajar_todos_informes.py` | Las autoridades del JNE (`Autoridades_Electas_20251113.xls`) se bajan a mano de la Plataforma Nacional de Datos Abiertos, a `datos/crudos/jne_autoridades/`. El buscador de informes de la Contraloría pide una sesión del navegador (`cookies.txt`) |
| 2. Lectura de documentos | OCR de los informes con `ocr_informes/` (en Kaggle con GPU, o `ocr_local.py`) y de las resoluciones escaneadas con `ocr_resoluciones.py`; lectura con modelos de lenguaje: `lector_claude.py` (resoluciones), `clasificar_hechos.py` y `extraer_deepseek.py` (informes); `validar_extracciones.py` | Piden claves de API en un `.env` local. Costó unos 31 dólares (16 por las resoluciones y 15 por los informes). Repetirlo da resultados parecidos, no idénticos: las salidas quedan guardadas y la cadena las lee |
| 3. Tablas y alertas | `procesar_conosce.py`, `limpiar_conformacion.py`, `procesar_ocds.py`, `perucompras_ordenes.py`, `entidad_declarante.py`; `radar/00_padron.py`, `14_antecedentes.py`, `19_empleos_previos.py`, `20_alerta_puerta_giratoria.py`, `01b_sumar_otras.py`, `01_base.py`; `padron_ubicacion.py`, `ocds_exposicion.py`; `radar/02_funcionarios.py`, `03_vinculos.py`, `25_tarjeta_territorial.py`, `26_senales_ocds.py`, `06_puntaje.py`, `23_panel_etiqueta.py`, `29_estructura_benchmark.py` | `capa2_padre_hijo_jne.py` corre solo y con el navegador cerrado, porque pide mucha memoria |
| 4. Capas de la v4 | `radar/v4/capa1_base.py`, `capa2_x.py`, `capa2_territorial.py`, `capa2_familia.py`, `capa2_padre_hijo_jne.py`, `capa2_padre_hijo_jne_prob.py`, `capa2_impedidos.py`, `estadisticos_simples.py` | |
| 5. Modelos y evaluación | Los modelos y las corridas de examen de 2022 y 2024 (`corrida_sellada.py`), la lista de 2025 (`lista_unica.py` y `lista_unica_limpia.py`, en sus modos `exploracion`, `segunda`, `congelar` y `expediente`) y lo que pesa en el parecido (`importancia_parecido.py`) | Los exámenes y la lista de 2025 se corren una sola vez: cada script se detiene si su salida ya existe |

## Manejo de errores

- **Entrada inválida.** `exportar_entidad.py` rechaza, con un mensaje y el código 2, un RUC de menos dígitos o con dígito
  verificador inválido, un RUC repetido, un año que no es 2025, más de 100 proveedores y una entidad que no compra o no
  está en el catálogo del OECE. `llika.py` avisa si falta un argumento o si el paso no existe.
- **Dato faltante.** Si falta un archivo, el paso dice cuál y qué script lo arma. Si la lista no tiene su huella, no arma
  la pantalla. Una orden sin descripción muestra «Sin descripción en CONOSCE». Una señal sin documento publicado lo dice
  («El OECE publica la penalidad en su registro, sin el documento»), y si la resolución no está en gob.pe, se da su número
  para buscarla.
- **Antes de escribir la pantalla,** `armar_demo.py` comprueba los RUC, los enlaces (solo de gob.pe), las palabras que no
  deben aparecer, el formato de los números y el contraste de los colores.
- **En el navegador,** un enlace a una entidad que no está muestra «No encontramos esa página»; un número de más de 11
  dígitos, que un RUC tiene 11 y un DNI 8; y una búsqueda sin resultados sugiere las entidades que sí hay. `probar_real.py`
  comprueba todo esto.

## Arquitectura y límites

```
fuentes públicas ──► descargas y lectura de documentos ──► tablas y alertas ──► capas v4 ──► modelos
                                                                                         │
           navegador ◄── HTML estático ◄── armar_demo.py ◄── JSON de la entidad ◄── lista congelada de 2025
                         (sin servidor)                   (exportar_entidad.py)     y puntajes de todos los casos
```

La pantalla es un solo archivo HTML: no tiene servidor ni base de datos, y se puede publicar en cualquier hosting estático.
Las cifras son de la PC del equipo (12 núcleos, 48 GB de RAM), medidas el 14/11/2026:

| Paso | Tiempo | Memoria pico |
|---|---|---|
| Datos de tres entidades (`exportar_entidad.py`) | 1,1 min | 4,4 GB |
| Puntajes de 2025, opcional (`llika.py puntajes`) | 6,5 a 8,9 min | 21 a 23 GB |
| Armar la pantalla | 0,5 s | 0,01 GB |
| Abrir la pantalla en el navegador | 0,44 s | archivo de 0,65 MB |
| Abrir una ficha | 9 ms (mediana); 32 ms el peor caso | |

**Costo.** Publicar la pantalla no cuesta nada. Leer las resoluciones y los informes con modelos de lenguaje costó unos
31 dólares, una sola vez. Actualizar cada año solo pide leer los documentos nuevos.

## Qué tan bien funciona

Se midió con las listas de 2022 y 2024, armadas solo con lo que se sabía al cierre de cada año, frente a las señales
adversas documentadas después: sanciones del Tribunal, penalidades y observaciones de la Contraloría. Entre los cinco
primeros de cada entidad hay 412 proveedores distintos con señal, frente a 150 si se ordena por monto (2,75 veces, con un
intervalo al 90 % de 2,43 a 3,14) y 160 si se ordena por el tamaño del proveedor. La lista de 2025 está congelada (huella
`189af867041d1270`) y se armó sin mirar ninguna señal de 2025: su resultado se medirá cuando se publiquen.

## Datos públicos y privacidad

Todo lo que muestra LLIKA son registros que el Estado ya publica, cruzados y con su fuente y su fecha. La única inferencia
propia es el posible parentesco: se muestra como coincidencia de apellidos, sin confirmar con RENIEC, nunca como un hecho.
Los nombres, RUC y DNI no están en este repositorio, y los scripts solo imprimen cifras agregadas.
