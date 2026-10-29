# Taller de evaluación (31/10/2026): el experimento mínimo

Esta carpeta repite la evaluación del E5 sin bajar los datos completos del radar, que pesan unos 28 GB y traen RUC y
DNI. Las tablas de `datos/` tienen una fila por proveedor, con las mismas variables que usa
`radar/12_evaluacion_e5.py`, pero sin RUC: cada proveedor es un número correlativo. No son una muestra, porque están
todos los proveedores de cada ventana.

## Cómo correrlo

**Opción 1, Colab (no hay que instalar nada).**
[Abrir el notebook en Colab](https://colab.research.google.com/github/PedroCasaverde/llika-radar/blob/main/taller/taller_experimento_minimo.ipynb)
y elegir *Entorno de ejecución → Ejecutar todas*. El notebook baja solo las tablas desde este repositorio.

**Opción 2, en una laptop.**

```
cd taller
pip install -r requirements.txt
python correr_taller.py          # o abrir el notebook en Jupyter o VS Code
```

En la PC donde se probó tardó unos 30 segundos. Al final imprime la línea de la plenaria, guarda la corrida en
`bitacora/` y suma una fila a `bitacora/BITACORA.md`.

## De dónde salen las tablas

1. `datos/_scripts/` baja las fuentes públicas: compras de CONOSCE, sanciones del Tribunal, socios y representantes del
   RNP, declaraciones juradas de intereses y padrón de SUNAT.
2. `radar/00` a `radar/17` cruzan las fuentes, generan las alertas y evalúan el modelo. `radar/12_evaluacion_e5.py`
   da las cifras del E5.
3. `taller/exportar_datos_taller.py` resume esa salida en las tres tablas de `datos/`, con la misma función `ventana()`
   de `radar/12`, y guarda su huella sha256 en `datos/huellas.json`.
4. El notebook repite la evaluación sobre esas tablas y comprueba que da las cifras del E5.

## Qué mide

| Pieza | Qué es |
|---|---|
| Referencia | Lo que se hace hoy: los proveedores ordenados por monto, de mayor a menor |
| Candidato | Modelo v0 del E5 (regresión logística L1 con alertas de vínculo y antecedentes), con su receta congelada: C = 0,01 |
| Conjunto congelado | 942 164 proveedores con órdenes en 2022-2023, 1 996 con sanción desde 2024 (huella sha256 en `datos/huellas.json`) |
| Partición | Aprende con las compras de 2018-2021 y las sanciones de 2022-2023; se examina con las compras de 2022-2023 y las sanciones desde 2024 |
| Métrica | Precisión en los 1000 primeros: qué porcentaje tuvo una sanción posterior |
| Incertidumbre | Bootstrap de proveedores con pesos de Poisson, 1000 repeticiones, semilla 11 |
| Contrapeso | Carga por entidad: cuántos de esos proveedores le tocan a cada entidad |
| KPIs del E3 | Aciertos en el top-50 y el top-1000 frente al monto y al azar, cuántos no habrían entrado ordenando por monto y el peso de los proveedores pequeños en la lista |

Resultado esperado (el del E5): referencia 3,7 %, regla por puntos 7,1 %, v0 14,3 %. La diferencia del v0 frente a la
referencia es de +10,6 puntos porcentuales, con rango de +8,4 a +12,9. El notebook compara contra estas cifras y avisa
si algo no coincide.

## Archivos

- `taller_experimento_minimo.ipynb`: el notebook, en diez pasos. Se arma desde `correr_taller.py`, así que los dos
  dicen lo mismo.
- `correr_taller.py`: lo mismo como script, para quien no use notebooks.
- `requirements.txt`: las librerías, con las versiones con las que se probó.
- `datos/`: `aprendizaje.parquet` (compras de 2018-2021), `examen.parquet` (2022-2023), `examen_entidades.parquet`
  (pares proveedor-entidad para la carga) y `huellas.json`.
- `bitacora/`: `BITACORA.md` (lo declarado, los cambios y la tabla de corridas), el pre-registro y cada corrida en JSON.
- `exportar_datos_taller.py`: arma las tablas a partir de los datos completos; solo lo necesita quien los tenga.

Si no se pueden cargar las tablas, `MODO_DEMO = True` usa el 10 % de los proveedores. Los números cambian y hay que
declararlo en la bitácora.
