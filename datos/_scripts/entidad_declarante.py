"""Mejor identificación de la entidad de cada declarante (PENDIENTES, sección 5: la regla por nombre del 15/10 solo
reconoce el 54,8 % de las declaraciones, y eso afecta a todas las alertas de funcionarios).

  python -X utf8 entidad_declarante.py

Para cada nombre de entidad distinto de las declaraciones juradas:
1. la regla del 15/10 (nombre completo, o la parte antes de « - »), si da un solo RUC;
2. si no, la única entidad del padrón cuyo nombre contiene todas las palabras del nombre buscado, o entre varias, la
   que tiene menos palabras de más (a lo más 2), si es una sola («GOBIERNO REGIONAL HUÁNUCO» -> la sede central);
3. si no, la entidad con más palabras en común (Jaccard de 0,6 o más), si le saca 0,15 a la segunda.
Salida: datos/procesados/radar/entidad_declarante.csv (nombre, ruc, método y puntaje), para revisar una muestra antes de
reemplazar la regla en radar/comun.py (eso va primero a HALLAZGOS y se aplica con aprobación de Pedro).
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "radar"))
from comun import entidades, norm, ruc_por_nombre, ultimo  # noqa: E402
from validar_extracciones import VACIAS  # noqa: E402

CAMBIAN = {"SUB", "METROPOLITANO", "METROPOLITANA", "SUBREGIONAL"}


def main():
    d = pd.read_csv(ultimo("acc4_dji_declaraciones"), dtype=str, usecols=["ENTIDAD"], encoding="utf-8")
    cuenta = d["ENTIDAD"].fillna("").value_counts()
    nombres = pd.Series(cuenta.index)
    ent = entidades()
    exacto = ruc_por_nombre(nombres, ent)
    pal = [(r, set(n.split()) - VACIAS) for r, n in zip(ent["RUC"], ent["n1"])]
    filas = []
    for nombre, ex in zip(nombres, exacto):
        if isinstance(ex, str):
            filas.append((nombre, ex, "regla_15_10", 1.0))
            continue
        w = set(norm(nombre).split()) - VACIAS
        if len(w) < 2:
            filas.append((nombre, None, "sin_palabras", 0.0))
            continue
        # palabras de más que cambian de entidad: «DIRECCIÓN REGIONAL DE SALUD» no es la «DIRECCIÓN SUB REGIONAL», ni
        # «GOBIERNO REGIONAL LIMA» el «METROPOLITANO» (revisión de 46 asignaciones, 1/11)
        sub = [(len(s - w), r) for r, s in pal if w <= s and not ((s - w) & CAMBIAN)]
        if len(sub) == 1:
            filas.append((nombre, sub[0][1], "contiene_todas", 1.0))
            continue
        if sub:   # varias: la de nombre más parecido (menos palabras de más, a lo más 2), si es una sola
            sub.sort()
            if sub[0][0] <= 2 and (len(sub) == 1 or sub[1][0] > sub[0][0]):
                filas.append((nombre, sub[0][1], "contiene_todas_mas_cercana", 1.0))
                continue
        j = sorted(((len(w & s) / len(w | s), r) for r, s in pal), reverse=True)[:2]
        if j and j[0][0] >= 0.6 and j[0][0] - (j[1][0] if len(j) > 1 else 0) >= 0.15:
            filas.append((nombre, j[0][1], "jaccard", round(j[0][0], 3)))
        else:
            filas.append((nombre, None, "ambiguo" if sub else "sin_candidato", round(j[0][0], 3) if j else 0.0))
    m = pd.DataFrame(filas, columns=["nombre", "ruc", "metodo", "puntaje"])
    m["declaraciones"] = m["nombre"].map(cuenta)
    salida = RAIZ / "datos" / "procesados" / "radar" / "entidad_declarante.csv"
    m.to_csv(salida, index=False)
    t = m.groupby("metodo")["declaraciones"].sum()
    print((t / t.sum() * 100).round(1).to_string())
    sin_pnp = ~m["nombre"].map(norm).str.startswith("POLICIA NACIONAL")
    print(f"Sin la PNP (un solo nombre para todas sus unidades): "
          f"{m.loc[sin_pnp & m['ruc'].notna(), 'declaraciones'].sum() / m.loc[sin_pnp, 'declaraciones'].sum():.1%}")
    print(f"Declaraciones con entidad: {m.loc[m['ruc'].notna(), 'declaraciones'].sum() / m['declaraciones'].sum():.1%} "
          f"(antes, solo la regla del 15/10: {t.get('regla_15_10', 0) / t.sum():.1%})")


if __name__ == "__main__":
    main()
