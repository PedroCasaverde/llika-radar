# LLIKA: radar de vínculos e impedimentos en las compras menores

Proyecto del curso Capstone Project I de la Maestría en Ciencia de Datos e Inteligencia Artificial de UTEC Posgrado.
Equipo 05: Pedro Casaverde, Gian Carlo Quiroz y Sherly Salazar.

El radar cruza las órdenes de compras menores (hasta 8 UIT) de todo el país con registros públicos del OECE, la
Contraloría y SUNAT. Marca las contrataciones con personas o empresas impedidas o vinculadas con la entidad que compra y,
para cada entidad y año, entrega los casos ordenados por puntaje y monto, con la evidencia y la norma de cada alerta. Es
una herramienta para que el órgano de control institucional (OCI) decida qué revisar primero: no acusa a nadie.

## Las seis alertas

| Alerta | Qué detecta | Gravedad |
|---|---|---|
| Sancionado que vende | Proveedor inhabilitado por el Tribunal que vende durante su sanción | 3 (1 si es servicio público) |
| Empresa espejo | Empresa con un socio o representante que está sancionado o integra una empresa sancionada | 3 (1 si el vínculo pasa por un gremio) |
| Fraccionamiento entre RUC del mismo dueño | Una persona y sus empresas que, en una entidad y un año, quedan cada una bajo el tope y juntas lo pasan | 2 |
| Funcionario que se contrata | Funcionario con cargo de decisión que le vende a su propia entidad | 3 |
| Doble percepción | Funcionario con cargo en una entidad que cobra órdenes de otra en las mismas fechas | 2 (1 si paga una entidad educativa) |
| Posible familiar | Persona con los dos apellidos de un funcionario de la entidad que le compra | 2 (3 si el funcionario tiene cargo de decisión) |

El puntaje de un caso (proveedor, entidad y año) suma, por cada tipo, la alerta más alta: gravedad legal por
probabilidad. La probabilidad es 1 en los cruces exactos y la de parentesco en los apellidos. Si dos casos empatan, va
primero el de mayor monto.

## Datos

Todas las fuentes son públicas. No se versionan: pesan varios GB y las salidas del radar traen datos de personas.

| Fuente | Quién la publica | Para qué se usa |
|---|---|---|
| Órdenes de compras menores (CONOSCE), 2018-2026 | OECE | Base de todas las alertas |
| Sanciones del Tribunal, una versión por mes desde 2021 | OECE | Sancionado que vende, empresa espejo y validación |
| Socios y representantes de proveedores (RNP), un corte por mes desde 11/2021 | OECE | Empresa espejo y mismo dueño |
| Declaraciones juradas de intereses | Contraloría | Funcionario que se contrata, doble percepción y posible familiar |
| Padrón reducido de RUC | SUNAT | Frecuencia de nombres y apellidos, y control de identidades |
| Entidades | OECE | Nombre, tipo y departamento de cada entidad |

## Requisitos

- Python 3.13 y `pip install -r requirements.txt`.
- 16 GB de RAM: la base de órdenes tiene unos 20 millones de filas.
- Unos 30 GB libres para los datos crudos y procesados.

## Cómo correrlo

Todo se corre desde la raíz del repositorio. Los scripts solo imprimen cifras agregadas.

1. Bajar las fuentes a `datos/crudos/`:

   ```
   python datos/_scripts/bajar_fuentes.py
   python datos/_scripts/bajar_historia_wiki.py sancionados
   python datos/_scripts/bajar_historia_conformacion.py <primera versión> <última versión>
   ```

   La conformación de proveedores se baja por rango de versiones de la wiki de datos abiertos del OECE, una por mes desde
   noviembre de 2021.

2. Procesar las órdenes y limpiar la conformación:

   ```
   python datos/_scripts/procesar_conosce.py
   python datos/_scripts/limpiar_conformacion.py
   ```

3. Correr el radar en orden:

   | Script | Qué hace |
   |---|---|
   | `radar/00_padron.py` | Cuenta, con el padrón de SUNAT, cuántas personas tienen cada nombre completo y cada par de apellidos |
   | `radar/01_base.py --rehacer` | Arma la base de órdenes y de sanciones con su historia, y la alerta de sancionado que vende |
   | `radar/02_funcionarios.py` | Alertas de funcionario que se contrata, doble percepción y posible familiar, con la probabilidad de parentesco |
   | `radar/03_vinculos.py` | Alertas de empresa espejo y de fraccionamiento entre RUC del mismo dueño, con los socios vigentes el día de cada orden |
   | `radar/04_indices.py` | Índices por entidad, encogidos con Beta-Binomial |
   | `radar/05_control.py` | Control de confiabilidad: cifras, muestra al azar y sanciones posteriores |
   | `radar/06_puntaje.py` | Puntaje por puntos y tabla de casos |
   | `radar/08_lasso.py` | Prueba de los pesos con Lasso, en años separados |
   | `radar/09_confirmacion_2026.py` | Confirmación con las órdenes y sanciones de 2026, con criterios fijados antes |
   | `radar/10_ficha_entidad.py --ejemplo` | Prototipo v0: el Excel de casos y fichas de una entidad y un año |
   | `radar/11_muestra_parentesco.py` | Muestra de alertas de familiar para verificar a mano en las declaraciones juradas |
   | `radar/12_evaluacion_e5.py` | Evaluación del E5: baselines contra el modelo v0, con rangos y error por segmento |

Una corrida completa sobre ocho años toma menos de una hora en una computadora personal, sin contar la limpieza inicial
de los cortes de socios. La ficha de una entidad sale en unos cuatro segundos.

## Qué produce

- `datos/procesados/radar/`: alertas, casos, índices y fichas por entidad. No se versiona porque trae datos de personas.
- `resultados/evaluacion_e5.json`: los agregados de la evaluación del E5, sin datos de personas.

## Evaluación (E5)

Prueba con las alertas de 2022-2023 frente a las sanciones del Tribunal que empiezan desde 2024 (942 164 proveedores,
1 996 con sanción posterior). Métrica: de los 1 000 primeros proveedores de cada lista, el porcentaje con sanción
posterior, con intervalo del 95 % por bootstrap.

| Opción | 1 000 primeros con sanción posterior | Proveedores de la lista por entidad (mediana) |
|---|---|---|
| Al azar | 0,21 % (0,00 a 0,50) | 1 |
| Lo que se hace hoy: lo más caro primero | 3,7 % (2,5 a 4,9) | 7 |
| Modelo simple: regresión logística de tamaño | 5,5 % (4,1 a 7,0) | 8 |
| Modelo v0: puntaje por puntos | 7,1 % (5,5 a 8,6) | 4 |
| Modelo v0 con pesos aprendidos (Lasso) | 7,8 % (6,2 a 9,6) | 7 |

## Privacidad

Las fuentes son públicas, pero las salidas tratan datos personales (Ley 29733). Por eso los scripts solo imprimen
agregados, las salidas se quedan en la computadora del equipo y la probabilidad de parentesco es solo para el usuario
institucional. Compartir apellidos no prueba parentesco: la alerta se vuelve un hecho solo cuando el especialista la
confirma con la declaración jurada del funcionario.
