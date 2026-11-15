# Datasheet de los datos de LLIKA

Sigue el esquema de Gebru et al. (2021), *Datasheets for Datasets*. Describe las tablas que arma la cadena del
repositorio a partir de fuentes públicas del Estado peruano.

## Motivación

- **Para qué se armó:** para ordenar, en cada entidad y año, a los proveedores de compras menores (hasta 8 UIT, sin
  concurso) según por cuáles conviene empezar una revisión, y para medir si ese orden encuentra más proveedores con señales
  adversas posteriores que las reglas que se usan hoy.
- **Quién la armó:** el Equipo 05 de Proyecto Integrador I (UTEC Posgrado), sin financiamiento externo. El costo de API
  para leer documentos fue de unos 31 dólares.

## Composición

- **Unidad:** el caso, que es una entidad, un proveedor y un año. Se arma con las órdenes de compra y de servicio de
  CONOSCE y con las «otras contrataciones».
- **Base de órdenes:** 23 329 782 órdenes de 2018 a 2025. Solo en 2025 son 3 509 398, que forman 1 020 200 casos; después
  de sacar los casos que son sobre todo compras por catálogo quedan 1 007 366, de 2 647 entidades.
- **Lista congelada de 2025:** 13 164 filas (cinco por entidad), con 10 875 proveedores; el 27,0 % de las filas son
  personas naturales.

| Fuente | Qué aporta | Quién la publica | Corte usado |
|---|---|---|---|
| Órdenes de compra y servicio (CONOSCE) | Base de casos y montos | OECE | Las de 2025, publicadas el 10/09/2026 |
| Concursos y contratos (OCDS) | Experiencia del proveedor en concursos | OECE | 31/10/2026 |
| Sancionados, multas, penalidades e inhabilitaciones judiciales | Antecedentes y etiquetas | OECE (una versión por mes desde 09/2021) | Lo publicado hasta el 31/12/2025 |
| Socios y representantes (RNP) | Mismo dueño y empresa espejo | OECE (un corte por mes desde 11/2021) | 1/12/2025 |
| Resoluciones del Tribunal de Contrataciones | Hechos, entidad y orden de cada sanción | gob.pe | 10 905 resoluciones de 2018 a 2026 |
| Informes de control | Observaciones de la Contraloría | Buscador de la Contraloría | 14 915 informes, consultado el 31/10/2026 |
| Declaraciones juradas de intereses y empleos previos | Funcionarios y sus empleos anteriores | Contraloría | 15/07/2026 |
| Autoridades electas | Vínculo con una autoridad | JNE | 13/11/2025 |
| Padrón reducido de RUC | Ubicación y frecuencia de apellidos | SUNAT | 3/10/2026 |
| Catálogos electrónicos | Compras por acuerdo marco | Perú Compras | 885 mil órdenes |

- **Datos de personas:** nombres, RUC y DNI de personas naturales que le vendieron al Estado, y nombres y cargos de
  funcionarios y autoridades. Todos son públicos.
- **Etiqueta:** señal adversa documentada. Es una sanción del Tribunal, una penalidad o una observación de la Contraloría
  por hechos del año del caso, conocida dentro de los dos años siguientes. Es incompleta: lo que nadie revisó no aparece.

## Proceso de recolección

- **Descargas:** con los scripts de `datos/_scripts/`, que dejan cada archivo con su URL, tamaño y fecha en un registro de
  descargas.
- **Lectura de documentos:** de los informes y de las resoluciones escaneadas con OCR (en Kaggle con GPU y en la PC).
  Texto para 14 889 de 14 895 informes y para todas las resoluciones distintas.
- **Extracción con modelos de lenguaje:**
  - las resoluciones, con Claude Haiku 5.5 en lotes;
  - los informes de fondo, con DeepSeek;
  - las dos con un esquema JSON fijo.
- **Validación contra fuentes oficiales:**
  - en las resoluciones, RUC válido en el 93,9 % y sancionado presente en el registro del OECE en el 96,4 %;
  - en los informes, entidad identificada en el 90,5 %;
  - contra lo copiado a mano por el equipo, se hallaron 274 de 275 órdenes.

## Preprocesamiento y limpieza

- **Órdenes:**
  - se sacan las anuladas, las repetidas y las de monto imposible (cero o sobre el tope del año: 9 UIT en 2022 por el
    DU 016-2022, 8 UIT en los demás años);
  - salen de la lista los casos con la mitad o más de su monto en compras por catálogo;
  - los pasajes aéreos no cuentan para la bandera de compra partida.
- **Socios:** los de cada orden son los vigentes el día de la orden, con el corte mensual anterior.
- **Sin información del futuro:** todo dato del proveedor se toma con lo que se sabía al cierre del año del caso. Lo que
  el Estado publica como foto, sin historia (declaraciones juradas y padrón de SUNAT), va con su última versión, y la ficha
  dice su fecha.
- **Nombres:** en lo que publica el OECE, la Ñ llega como «¿»; en la pantalla se corrige.

## Usos

- **Previsto:** el orden sugerido para revisar y la evaluación de ese orden.
- **No previsto:** perfilar personas, sancionar, decidir contrataciones o tomar el parecido como probabilidad.
- **Sesgos conocidos:**
  - las observaciones de la Contraloría dependen de qué entidades se auditaron;
  - las sanciones, de quién denuncia;
  - el parecido favorece a los proveedores activos.

## Distribución

- **El código** es público, en este repositorio.
- **Los datos procesados no se publican en ningún enlace,** porque traen nombres y DNI: el equipo los entrega en persona a
  quien los pida para evaluar el proyecto.
- **Para la ruta del caso real** bastan 39 archivos, unos 4 GB: el rastreo de lecturas de `radar/v4/exportar_entidad.py`
  está en `docs/paquete_de_datos.md`.
- **La pantalla publicada** muestra solo registros que el Estado ya publica, con su fuente y su fecha.

## Mantenimiento

- **Actualización:** las fuentes del OECE cambian cada mes; la lista se arma una vez al año, al cierre.
- **Medición:** la lista de 2025 está congelada (huella `189af867041d1270`) y se medirá cuando se publiquen las señales de
  2025.
- **Correcciones:** cualquier persona puede pedir que se corrija un dato. Lo mantiene el Equipo 05 mientras dure el
  proyecto.
