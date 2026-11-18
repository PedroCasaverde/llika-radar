# Paquete de datos para correr el caso real

Los datos procesados no van a git ni a ningún enlace: traen nombres, RUC y DNI. El equipo los entrega en persona y van en
`datos/`, con estas mismas rutas. Las listas salen de rastrear cada archivo que leen los scripts (14/11/2026).

## Para el caso real (`python llika.py caso-real`)

39 archivos, 3,96 GB. Los lee `radar/v4/exportar_entidad.py`. El paso anterior, `radar/v4/indices_entidades.py`
(los índices de las 2 647 entidades), lee siete de esos mismos archivos y no pide ninguno más: la lista, los puntajes,
las órdenes, las sanciones, las dos versiones de penalidades y la de multas.

| Archivo o carpeta | Archivos | MB |
|---|---|---|
| `datos/crudos/acc8_informes_cgr/listas/lista_unica_informes_ampliada.csv` | 1 | 46,4 |
| `datos/crudos/acc9_entidades/entidades_contratantes_20261003.csv` | 1 | 0,4 |
| `datos/crudos/jne_autoridades/Autoridades_Electas_20251113.xls` | 1 | 14,0 |
| `datos/crudos/oece_multas/_historia` | 1 | 1,1 |
| `datos/crudos/oece_penalidades/_historia` | 2 | 2,8 |
| `datos/crudos/tce_resoluciones/indice.csv` | 1 | 13,0 |
| `datos/procesados/acc3_conformacion` | 11 | 1 067,7 |
| `datos/procesados/benchmark/eventos.parquet` | 1 | 0,8 |
| `datos/procesados/radar/alerta_doble_percepcion.parquet` | 1 | 0,1 |
| `datos/procesados/radar/alerta_empresa_espejo.parquet` | 1 | 0,6 |
| `datos/procesados/radar/alerta_puerta_giratoria.parquet` | 1 | 0,4 |
| `datos/procesados/radar/alerta_sancionado.parquet` | 1 | 0,2 |
| `datos/procesados/radar/empleos_previos.parquet` | 1 | 46,0 |
| `datos/procesados/radar/ordenes.parquet` | 1 | 395,7 |
| `datos/procesados/radar/tarjeta_territorial.parquet` | 1 | 0,8 |
| `datos/procesados/v4/base/entidades.parquet` | 1 | 0,1 |
| `datos/procesados/v4/base/hechos_orden.parquet` | 1 | 0,3 |
| `datos/procesados/v4/base/hechos_relacion.parquet` | 1 | 0,4 |
| `datos/procesados/v4/base/informes.parquet` | 1 | 0,6 |
| `datos/procesados/v4/base/ordenes.parquet` | 1 | 851,5 |
| `datos/procesados/v4/base/ordenes_texto.parquet` | 1 | 808,6 |
| `datos/procesados/v4/base/proveedores.parquet` | 1 | 71,9 |
| `datos/procesados/v4/base/resoluciones.parquet` | 1 | 0,7 |
| `datos/procesados/v4/base/sanciones.parquet` | 1 | 0,4 |
| `datos/procesados/v4/resultados/lista_unica_limpia/lista_2025_unica.parquet` | 1 | 0,9 |
| `datos/procesados/v4/resultados/lista_unica_limpia/puntajes_2025.parquet` | 1 | 76,1 |
| `datos/procesados/v4/resultados/lista_unica_limpia/puntajes_2025_proveedor.parquet` | 1 | 13,2 |
| `datos/procesados/v4/x_caso_anio.parquet` | 1 | 546,0 |

## Además, para rehacer los puntajes de 2025 (`python llika.py puntajes`, opcional)

192 archivos más, 0,19 GB. Los lee `radar/v4/puntajes_2025.py`. Ese paso escribe de nuevo
`puntajes_2025.parquet` y se detiene si la lista que rehace no es idéntica a la congelada.

| Archivo o carpeta | Archivos | MB |
|---|---|---|
| `datos/crudos/oece_inhabilitaciones_judiciales/_historia` | 60 | 0,0 |
| `datos/crudos/oece_penalidades/_historia` | 120 | 106,4 |
| `datos/procesados/ocds/adjudicaciones.parquet` | 1 | 32,9 |
| `datos/procesados/ocds/contratos.parquet` | 1 | 19,8 |
| `datos/procesados/perucompras/ordenes_catalogo.parquet` | 1 | 23,7 |
| `datos/procesados/v4/casos/eventos_examen.parquet` | 1 | 0,8 |
| `datos/procesados/v4/diccionario_x.csv` | 1 | 0,0 |
| `datos/procesados/v4/resultados/lista_unica/lista_2025_unica.parquet` | 1 | 0,9 |
| `datos/procesados/v4/x_familia.parquet` | 1 | 0,0 |
| `datos/procesados/v4/x_impedidos.parquet` | 1 | 0,0 |
| `datos/procesados/v4/x_padre_hijo.parquet` | 1 | 0,3 |
| `datos/procesados/v4/x_padre_hijo_jne.parquet` | 1 | 2,0 |
| `datos/procesados/v4/x_territorial.parquet` | 1 | 0,2 |
| `datos/procesados/v4/x_territorial_proveedor.parquet` | 1 | 0,1 |

## Además, para el notebook de la evaluación (`notebooks/evaluacion_v0.ipynb`, opcional)

6 archivos más, unos 18 MB, en `datos/procesados/v4/resultados/lista_unica_limpia/`. Las dos tablas de evaluación las
arma `radar/v4/tabla_evaluacion.py` (7,6 minutos y 20 GB de memoria pico, con los datos de la ruta 3; no se rastreó si
alcanza con el paquete de los puntajes): rehace el universo
limpio de 2022 y 2024 y se detiene si la lista no es idéntica a la guardada o si no repite la segunda medición.

| Archivo | Archivos | MB |
|---|---|---|
| `evaluacion_2022.parquet` y `evaluacion_2024.parquet` | 2 | 14,9 |
| `evaluacion_manifiesto.json` y `segunda_medicion.json` | 2 | 0,0 |
| `lista_2022.parquet` y `lista_2024.parquet` | 2 | 1,8 |

## Huellas para comprobar

| Archivo | SHA-256 (primeros 16) |
|---|---|
| `datos/procesados/v4/resultados/lista_unica_limpia/lista_2025_unica.parquet` | `189af867041d1270` |
| `datos/procesados/v4/resultados/lista_unica_limpia/puntajes_2025.parquet` | `79d655736b29bd46` |
| `datos/procesados/v4/resultados/lista_unica_limpia/evaluacion_2022.parquet` | `96d6e4936f1e8dc9` |
| `datos/procesados/v4/resultados/lista_unica_limpia/evaluacion_2024.parquet` | `0c52002e1bf62336` |

Con estos datos, `python llika.py caso-real` deja dos archivos que se pueden comprobar (17/11/2026):

| Archivo que deja | SHA-256 (primeros 16) |
|---|---|
| `datos/procesados/v4/resultados/lista_unica_limpia/distribucion_indices_2025.json`, igual a `demo/fuente/distribucion_indices_2025.json` | `73485208c9d6a88a` |
| `privado/datos_caso_real_2025.json` | `3b9ed5208ec0f4aa` |

Hasta el 14/11/2026 el JSON llevaba la fecha en que se armó, y su huella (`e9de3840bf7a252d`) solo se repetía ese día.
Desde el 17/11 la fecha va en `datos_caso_real_2025.control.json` y la huella del JSON ya no cambia con el día.
