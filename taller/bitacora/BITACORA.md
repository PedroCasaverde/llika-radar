# Bitácora del experimento mínimo (G5, LLIKA)

Esta bitácora registra lo que el equipo declaró antes de correr, cómo se repite una corrida, los cambios declarados y
cada corrida hecha. Cada corrida deja además un JSON en esta carpeta con la fecha, las versiones, las semillas, las
huellas de los datos y todos los resultados.

## 1. Lo declarado antes de correr

| Pieza | Declaración |
|---|---|
| Referencia | Lo que se hace hoy: los proveedores ordenados por monto total, de mayor a menor |
| Candidato | Modelo v0 del E5: regresión logística L1 con las cinco alertas de vínculo y cuatro variables de antecedentes, con su receta congelada (C = 0,01) |
| Conjunto congelado | 942 164 proveedores con órdenes en 2022-2023 y sin sanción previa a 2024, de los cuales 1 996 tienen una sanción desde 2024 (`examen.parquet`, sha256 `25e19432…`) |
| Partición | Se aprende con las compras de 2018 a 2021 y las sanciones que empiezan en 2022 o 2023; se examina con las compras de 2022 y 2023 frente a las sanciones desde el 1/1/2024 |
| Métrica primaria | Precisión en los 1000 primeros: de los 1000 primeros de la lista, qué porcentaje recibió después una sanción del Tribunal |
| Contrapeso | Carga por entidad: mediana y percentil 90 de los casos que le tocan a cada entidad |
| Incertidumbre | Bootstrap de proveedores con pesos de Poisson, 1000 repeticiones, semilla 11, pareado entre listas |

El detalle está en `preregistro.json`, que el notebook vuelve a escribir, con su fecha, cada vez que corre.

## 2. Cómo se repite una corrida

1. Abrir `taller/taller_experimento_minimo.ipynb` en Colab y ejecutar todas las celdas. En una laptop, desde la carpeta
   `taller/`, correr `pip install -r requirements.txt` y luego `python correr_taller.py`.
2. Comprobar que las tres huellas dicen «OK, es el conjunto congelado» y que el paso 9 dice «coincide» en las tres
   listas.
3. Comparar el JSON nuevo con el de la fila elegida en la tabla de corridas: deben coincidir los resultados, las
   diferencias y la línea de la plenaria.

## 3. Cambios declarados

| Fecha | Qué cambió | Por qué | Efecto en las cifras |
|---|---|---|---|
| 29/10/2026 | La regularización del v0 queda fija en C = 0,01, el valor que eligió la validación cruzada del E5, en vez de buscarse otra vez | El día del taller no se optimizan hiperparámetros | Ninguno: los mismos pesos y el mismo 14,3 % |
| 29/10/2026 | El script fija su salida en UTF-8 | En Git Bash, o al guardar la salida en un archivo, se caía antes de registrar la corrida | Ninguno |
| 29/10/2026 | El pre-registro declara también la partición y la receta del v0, y el notebook comprueba que las tablas son las declaradas | Que la partición sin fuga se vea en pantalla | Ninguno |

Los cambios del proyecto frente al E4 están declarados en la hoja Cambios del E4: el giro del 23/10 hacia vínculos e
impedimentos, el v0 con antecedentes, la corrección de la fuga de etiquetas y la nueva evidencia de la H1′.

## 4. Corridas

La corrida de las 14:27 se hizo con la versión anterior del notebook, que buscaba C con validación cruzada: eligió el
mismo C y dio los mismos números.

| Fecha | Modo | Referencia | Candidato | Diferencia en p.p. [rango 95 %] | Entorno | Archivo |
|---|---|---|---|---|---|---|
| 2026-10-29 14:27 | completo | 3,7 % | 14,3 % (v0) | +10,6 [+8,4 a +12,9] | Python 3.13.12, scikit-learn 1.9.1 | `corrida_20261029_142748.json` |
