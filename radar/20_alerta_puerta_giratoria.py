"""Paso 20: alerta de «puerta giratoria» (PENDIENTES.md 5; HALLAZGOS 69).

Una orden de la entidad E al proveedor P se marca si un declarante de E:
- tenía el cargo en E en la fecha de la orden (periodo de la declaración jurada), y
- trabajó antes (o trabaja todavía) en P, una empresa privada con RUC válido, y ese empleo previo empezó antes de la
  orden y terminó como mucho 4 años antes de ella (o sigue vigente).

No mira las sanciones: solo construye la alerta y la describe. La evaluación con la regla de siempre va en el paso 21,
con las variantes registradas antes de correrla.

Salida: datos/procesados/radar/alerta_puerta_giratoria.parquet (una fila por orden y declarante).
    E:/UTEC_llika/.venv/Scripts/python.exe -I -X utf8 -W ignore 20_alerta_puerta_giratoria.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import SALIDA  # noqa: E402

ENFRIAMIENTO = pd.DateOffset(years=4)
EDUCATIVO = (r"UNIVERSIDAD|INSTITUTO|COLEGIO|ESCUELA|ACADEMIA|CENTRO DE ESTUDIOS|EDUCA|PONTIFICIA|SCHOOL|COLLEGE"
             r"|CENTRO NAVAL|CLUB")


def main():
    t0 = time.time()
    e = pd.read_parquet(SALIDA / "empleos_previos.parquet")
    e = e[e.ruc_valido & ~e.empleador_publico & e.ruc_entidad.notna() & e.inicio_previo.notna()]
    e = e[e.ruc_empleador != e.ruc_entidad]
    pares = set(zip(e.ruc_entidad, e.ruc_empleador))
    print(f"Empleos previos privados con entidad actual identificada: {len(e):,} "
          f"({e.persona.nunique():,} personas; {len(pares):,} pares entidad-empleador)", flush=True)

    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "anio", "monto"])
    o["ruc"], o["ruc_entidad"] = o.ruc.astype(str), o.ruc_entidad.astype(str)
    clave = o.ruc_entidad + "|" + o.ruc
    o = o[clave.isin({f"{a}|{b}" for a, b in pares})]
    print(f"Órdenes de esas entidades a esos empleadores (sin mirar fechas todavía): {len(o):,}", flush=True)

    m = o.merge(e, left_on=["ruc_entidad", "ruc"], right_on=["ruc_entidad", "ruc_empleador"])
    en_cargo = (m.fecha >= m.inicio) & (m.fecha <= m.cese)
    empezo_antes = m.inicio_previo <= m.fecha
    reciente = m.cese_previo.isna() | (m.cese_previo >= m.fecha - ENFRIAMIENTO)
    a = m[en_cargo & empezo_antes & reciente].copy()
    a["vigente_en_empleador"] = a.cese_previo.isna() | (a.cese_previo >= a.fecha)
    a["meses_desde_salida"] = ((a.fecha - a.cese_previo).dt.days / 30.44).round(1)
    # Puerta giratoria propiamente dicha: dejó el proveedor (a lo más un mes después) y entró a la entidad
    a["salio_antes_de_entrar"] = a.cese_previo.notna() & (a.cese_previo <= a.inicio + pd.Timedelta(days=31))
    # Docencia: universidades, institutos y colegios. Enseñar es común y permitido, y esas instituciones le venden
    # cursos a las entidades: se separa para no confundirlo con un conflicto de interés.
    a["educativo"] = a.nombre_empleador.fillna("").str.upper().str.contains(EDUCATIVO, regex=True)
    cols = ["id", "ruc_entidad", "ruc", "fecha", "anio", "monto", "persona", "cargo", "decide", "locador", "inicio",
            "nombre_empleador", "cargo_previo", "inicio_previo", "cese_previo", "vigente_en_empleador",
            "salio_antes_de_entrar", "educativo", "meses_desde_salida"]
    a = a[cols].drop_duplicates(["id", "persona"])
    a.to_parquet(SALIDA / "alerta_puerta_giratoria.parquet", index=False)

    u = a.drop_duplicates("id")
    print(f"Alerta: {u.id.nunique():,} órdenes por S/ {u.monto.sum() / 1e6:,.1f} millones; "
          f"{u.ruc.nunique():,} proveedores; {u.ruc_entidad.nunique():,} entidades; {a.persona.nunique():,} declarantes "
          f"({time.time() - t0:.0f} s)")
    print("  Por año:", u.groupby("anio").id.size().to_dict())
    print(f"  Con un declarante de cargo de decisión: {a.groupby('id').decide.any().mean():.1%} de las órdenes | "
          f"declarante locador: {a.groupby('id').locador.all().mean():.1%}")
    print(f"  El declarante seguía en el empleador al momento de la orden: {a.groupby('id').vigente_en_empleador.any().mean():.1%}")
    print(f"  Proveedor persona natural (RUC 10): {u.ruc.str.startswith('10').mean():.1%}")
    for nombre, filtro in (("sin docencia", ~a.educativo),
                           ("puerta giratoria estricta, sin docencia", ~a.educativo & a.salio_antes_de_entrar),
                           ("vínculo vigente con el proveedor, sin docencia", ~a.educativo & a.vigente_en_empleador)):
        v = a[filtro].drop_duplicates("id")
        print(f"  {nombre}: {len(v):,} órdenes; {v.ruc.nunique():,} proveedores; {v.ruc_entidad.nunique():,} entidades")


if __name__ == "__main__":
    main()
