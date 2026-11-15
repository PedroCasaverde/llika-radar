# Model Card v0 de LLIKA

Sigue el esquema de Mitchell et al. (2019), *Model Cards for Model Reporting*. Todas las cifras salen de los archivos de
resultados de la cadena (`segunda_medicion.json`, `lista_2025_manifiesto.json` e `importancia_parecido.json`), que no
traen datos de personas.

## Detalles del modelo

- **Quién y cuándo.** Equipo 05 del curso Proyecto Integrador I (UTEC Posgrado): Pedro Casaverde, Gian Carlo Quiroz y
  Sherly Salazar. Versión congelada el 13/11/2026, con la huella de la lista `189af867041d1270`.
- **Qué devuelve.** Para cada entidad y año, un orden sugerido de sus proveedores de compras menores (hasta 8 UIT): los
  cinco primeros son la recomendación y el resto sigue el mismo orden. El orden es una regla con tres pasos:
  1. Primero, los proveedores con un vínculo con una autoridad electa de la zona (la autoridad o su empresa) y las
     coincidencias de apellidos poco comunes con ella. Es una regla, sin pesos.
  2. Después, el puntaje R × log(1 + M) × C. R es el **parecido**: qué tanto se parece el proveedor a los que después
     tuvieron una sanción o una penalidad. M es lo que la entidad le compró en el año, y C, la parte de lo que el proveedor
     le vende al Estado que va a esa entidad.
  3. El quinto puesto es para la compra más grande que parece partida (dos o más compras en 30 días que pasan el tope, o
     una que llega al 90 % de él); si no hay ninguna, para la compra más grande.
- **El parecido R** une dos modelos por proveedor y año, uno para sanciones del Tribunal y otro para penalidades:
  R = 1 − (1 − p_sanción)(1 − p_penalidad). Cada uno tiene dos piezas:
  - un margen de exposición, una regresión logística sobre cuánto vende y en cuántas compras;
  - lo que le suma LightGBM con 54 datos del proveedor: 100 árboles de profundidad 3, restricciones monótonas, promedio
    de 10 semillas (0 a 9) y 12 hilos (`radar/v4/m6_proveedor.py`).
- **Ajuste de la lista de 2025:** casos de 2018 a 2023 (3 293 747 proveedores-año; 2 210 con sanción y 1 060 con
  penalidad), con lo que se sabía al 31/12/2025.
- **Cambio frente al contrato de alcance.** El Informe Preliminar pactó como modelo v0 una regresión logística L1. Se
  reemplazó por estos dos modelos durante la mejora del modelo (noviembre de 2026): con la unidad caso y la etiqueta de
  señal adversa, los modelos simples, la logística entre ellos, no superaron al orden por tamaño en los cortes de
  validación. Está declarado en la hoja Cambios del E4 (fila del 5/11/2026).

## Uso previsto

- **Para qué:** decidir por qué compras menores de una entidad conviene empezar una revisión.
- **Usuarios:** el especialista del órgano de control institucional (el usuario de diseño); también el abastecimiento de
  la entidad, periodistas y cualquier ciudadano, porque la web es pública y académica.
- **Fuera de su alcance:**
  - sancionar, acusar o calificar a una persona o empresa;
  - decidir una contratación;
  - medir corrupción o pérdidas: el dinero que muestra no es una pérdida probada;
  - leer R como una probabilidad: R ordena, no estima cuántos van a ser sancionados.

## Factores

Tipo de entidad, tamaño de la entidad, año, motivo por el que el proveedor entra a la lista y tipo de señal adversa
(impedimento, sanción, penalidad u observación de la Contraloría). En la lista de 2025, el 27,0 % de las filas son
personas naturales.

## Métricas

- **Métrica técnica:** proveedores distintos con una señal adversa documentada entre los cinco primeros de todas las
  entidades-año, contando a cada proveedor una vez por año. La señal tiene que ser por hechos del año del caso y conocida
  dentro de los dos años siguientes.
- **Comparadores:** el orden por monto (lo más caro primero), por tamaño del proveedor, alerta y luego monto, y al azar.
- **Incertidumbre:** intervalo al 90 % de la razón frente a cada comparador, con remuestreo de Poisson por proveedor
  (2 000 repeticiones, semilla 0).
- **Contrapeso:** las observaciones de la Contraloría (los hallazgos del control) no deben empeorar; también se miden las
  carpetas revisadas en vano por cada acierto.

## Datos de evaluación

Listas de 2022 y 2024 (26 303 filas), cada una ajustada solo con los años anteriores: 2018 a 2020 para 2022 y 2018 a 2022
para 2024. Es una segunda medición, no ciega: esos años ya se habían usado una vez como examen. La lista de 2025 está
congelada y se armó sin mirar ninguna señal de 2025.

## Resultados

| Verdad | LLIKA | Monto | Tamaño | Alerta y monto | Azar | LLIKA frente al monto (IC 90 %) |
|---|---|---|---|---|---|---|
| Señal adversa (todas) | **412** | 150 | 160 | 180 | 72 | 2,75 (2,43 a 3,14) |
| Impedimento | 61 | 5 | 0 | 64 | 4 | 12,2 (6,7 a 32,0) |
| Sanción del Tribunal | 102 | 19 | 40 | 24 | 21 | 5,37 (3,92 a 8,23) |
| Penalidad | 240 | 68 | 103 | 68 | 39 | 3,53 (2,98 a 4,33) |
| Observación de la Contraloría | 61 | 68 | 24 | 69 | 10 | 0,90 (0,75 a 1,05) |

**Por año:** frente al monto, 2,19 (1,88 a 2,57) en 2022 y 3,71 (3,05 a 4,67) en 2024.

**Por motivo** (de cada 100 filas, cuántas traen un proveedor con señal):

| Motivo | Por cada 100 filas |
|---|---|
| Historial, con antecedentes | 4,4 |
| Vínculo con una autoridad o su empresa | 4,1 |
| Coincidencia de apellidos | 2,2 |
| Historial, sin antecedentes | 1,6 |
| Compra con bandera | 0,6 |
| Peso en la entidad | 0,4 |
| Compra más grande | 0,4 |

**Contrapeso.** Con LLIKA se revisan 62,8 carpetas en vano por cada acierto, frente a 174,4 por monto. En las
observaciones de la Contraloría, LLIKA empata con el monto, sin mejorarlo (0,90; 0,75 a 1,05).

**Alcance.** De los 1 254 proveedores con señal en esos dos años, la lista de cinco por entidad ve 412 (32,9 %).

**Qué pesa en el parecido** (al mezclar cada grupo de datos entre los proveedores de 2025): concursos y contratos,
49,6 %; cuánto vende sin concurso, 20,7 %; antigüedad y crecimiento, 8,4 %; qué vende, 7,2 %; antecedentes, 6,8 %;
compras por catálogo, 6,4 %; cómo le va en los concursos, 0,8 %.

## Consideraciones éticas

- **El parecido mide, sobre todo, cuánto trabaja un proveedor con el Estado.** Favorece mirar a los proveedores activos.
  Por eso la ficha nunca lo presenta como una falta.
- **La coincidencia de apellidos es una inferencia.** Puede ser un homónimo, así que se muestra como «sin confirmar con
  RENIEC», con el número de personas del padrón que tienen esa combinación de apellidos.
- **Personas naturales:** son el 27 % de la lista de 2025. La pantalla muestra su nombre, RUC y DNI, que son datos
  públicos. Cualquier persona puede pedir que se corrija un dato.
- **Datos que no son abiertos:** los parientes declarados y las denuncias no lo son, y el alcance de la regla de vínculos
  depende de lo que el Estado publica.

## Advertencias y recomendaciones

- Usar la lista para ordenar una revisión, nunca como prueba.
- Leer siempre el motivo y los documentos de la ficha antes de pedir información.
- No comparar el R de dos proveedores como si fueran probabilidades.
- Repetir la medición con las señales de 2025 cuando se publiquen, antes de cambiar el modelo.
