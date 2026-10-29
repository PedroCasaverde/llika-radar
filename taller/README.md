# Taller de evaluación (31/10/2026): el experimento mínimo

Carpeta para repetir la evaluación del E5 sin bajar los datos completos del radar (pesan varios GB y traen RUC).
Las tablas de `datos/` tienen una fila por proveedor, con las mismas variables que usa `radar/12_evaluacion_e5.py`,
pero sin RUC: cada proveedor es un número correlativo.

## Cómo correrlo

**Opción 1, Colab (no hay que instalar nada).** Abrir `taller_experimento_minimo.ipynb` en Colab y ejecutar todo
(Entorno de ejecución → Ejecutar todas). El notebook baja solo las tablas desde este repositorio.

**Opción 2, en una laptop.**

```
pip install pandas pyarrow scikit-learn
cd taller
python correr_taller.py          # o abrir el notebook en Jupyter / VS Code
```

Tarda entre 2 y 5 minutos. Al final imprime la línea de la plenaria y guarda la corrida en `bitacora/`.

## Qué mide

| Pieza | Qué es |
|---|---|
| Referencia | Lo que se hace hoy: los proveedores ordenados por monto, de mayor a menor |
| Candidato | Modelo v0 del E5 (regresión logística L1 con alertas de vínculo y antecedentes) |
| Conjunto congelado | 942 164 proveedores con órdenes en 2022-2023, 1996 con sanción desde 2024 (huella sha256 en `datos/huellas.json`) |
| Métrica | Precisión en los 1000 primeros: qué porcentaje tuvo una sanción posterior |
| Incertidumbre | Bootstrap de proveedores con pesos de Poisson, 1000 repeticiones, semilla 11 |
| Contrapeso | Carga por entidad: cuántos de esos proveedores le tocan a cada entidad |

Resultado esperado (el del E5): referencia 3,7 %, regla por puntos 7,1 %, v0 14,3 %; diferencia v0 menos referencia
+10,6 puntos porcentuales, con rango de +8,4 a +12,9. El notebook compara solo contra estas cifras y avisa si algo no
coincide.

## Archivos

- `taller_experimento_minimo.ipynb`: el notebook, celda por celda en el orden del taller.
- `correr_taller.py`: lo mismo como script, para quien no use notebooks.
- `datos/`: `aprendizaje.parquet` (2018-2021), `examen.parquet` (2022-2023), `examen_entidades.parquet` (pares
  proveedor-entidad para la carga) y `huellas.json`.
- `bitacora/`: el pre-registro y cada corrida guardada en JSON (fecha, versiones, semillas, huellas y resultados).
- `exportar_datos_taller.py`: arma las tablas a partir de los datos completos; solo lo necesita quien tenga esos datos.

Si no se pueden cargar las tablas, `MODO_DEMO = True` usa el 10 % de los proveedores; los números cambian y hay que
declararlo en la bitácora.
