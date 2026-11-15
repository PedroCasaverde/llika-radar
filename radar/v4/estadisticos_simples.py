"""Capa 2, estadísticos simples para los hallazgos del control (HALLAZGOS 149; Pedro: «sí, corre la 149 en paralelo»,
2/11). Definiciones fijadas antes de calcular nada.

Por caso (entidad × proveedor × año), con datos del año del caso (contemporáneos):
1. Fraccionamiento por objeto (Ley 30225, art. 20): la entidad suma en el año, para el mismo objeto, más que el tope de
   la compra menor, en dos órdenes o más, a uno o varios proveedores.
     s_obj_frac_parte    parte del monto del caso en objetos así;
     s_obj_frac_veces    cuántas veces el tope suma, en la entidad y el año, el objeto del caso que más suma.
2. Pruebas de dígitos (Nigrini; Baesens et al., Fraud Analytics, cap. 2; OCP R029):
     s_benford_1d, s_benford_2d           desviación absoluta media (MAD) del primer y del segundo dígito de los montos
                                          del proveedor en el año (todas sus entidades, 10 soles o más) frente a la ley
                                          de Benford; solo con 30 órdenes o más, si no queda vacía;
     s_benford_1d_nac, s_benford_2d_nac   lo mismo frente a la distribución nacional del año, que cuida el efecto del
                                          tope (sensibilidad);
     s_redondo_100, s_redondo_1000        parte de las órdenes del caso con monto múltiplo de 100 y de 1000;
     s_monto_repetido                     parte de las órdenes del caso cuyo monto exacto (al céntimo) se repite en él.
3. Precio atípico frente a sus pares (Fraud Analytics, cap. 3): z robusta del logaritmo del monto frente a las órdenes
   del mismo objeto en todo el país ese año (mediana y MAD × 1,4826, piso de 0,05, z acotada en ±5; grupos de 20
   órdenes o más):
     s_precio_z          promedio de la z de las órdenes del caso;
     s_precio_atipico    parte de las órdenes del caso con z de 3 o más.
4. Fechas inusuales: s_fecha_inusual, parte de las órdenes del caso emitidas en sábado, domingo o feriado nacional.

El objeto sale de la descripción de la orden: lo que sigue a la primera marca genérica (adquisición, compra,
contratación, servicio, suministro, prestación), sin números ni palabras vacías, hasta la primera coma, dos puntos,
paréntesis o palabra de propósito (para, por, según...). La principal toma las dos primeras palabras; la de tres
palabras (s_obj3_*, s_precio3_*) es sensibilidad y solo se reporta.

    python -I -X utf8 estadisticos_simples.py [--sin-diccionario | --solo-diccionario]
--sin-diccionario: no toca diccionario_x.csv (para correr mientras otra capa lo reescribe); --solo-diccionario: solo
agrega las filas del diccionario a partir de x_simples.parquet.
Salida: datos/procesados/v4/x_simples.parquet y filas en diccionario_x.csv. Solo imprime agregados.
"""
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dateutil.easter import easter

V4 = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4"
BASE = V4 / "base"
K = ["ruc_entidad", "ruc", "anio"]
ANIOS = range(2018, 2026)
MIN_BENFORD, MIN_PARES, PISO, TOPE_Z = 30, 20, 0.05, 5.0
T0 = time.time()

TABLA = str.maketrans("ÁÉÍÓÚÜÀÈÌÒÙÄËÏÖÑ", "AEIOUUAEIOUAEION")
MARCA = re.compile(r"\b(?:ADQUISICION(?:ES)?|COMPRAS?|CONTRATACION(?:ES)?|SERVICIOS?|SUMINISTROS?|PRESTACION(?:ES)?)\b")
TOKEN = re.compile(r"[A-Z]+|[,;:()\[\]]")         # palabras sin dígitos y signos que cortan; los números se pierden solos
CORTAN = {",", ";", ":", "(", ")", "[", "]", "PARA", "POR", "SEGUN", "DURANTE", "CORRESPONDIENTE", "CORRESPONDIENTES",
          "DESTINADO", "DESTINADA", "DESTINADOS", "DESTINADAS", "QUE", "CONFORME", "MEDIANTE"}
VACIAS = set("""DEL LAS LOS UNA UNO UNOS UNAS CON SIN SUS OTRO OTRA OTROS OTRAS DIVERSO DIVERSA DIVERSOS DIVERSAS VARIOS
VARIAS COMO ENTRE SOBRE HASTA DESDE BAJO ANTE TRAS CONTRA HACIA ESTE ESTA ESTOS ESTAS ESE ESA ESOS ESAS DICHO DICHA CADA
TODO TODOS TODA TODAS MAS MUY SER ASI TAMBIEN MES MESES DIA DIAS ANO ANOS PERIODO ENERO FEBRERO MARZO ABRIL MAYO JUNIO
JULIO AGOSTO SETIEMBRE SEPTIEMBRE OCTUBRE NOVIEMBRE DICIEMBRE SERVICIO SERVICIOS SERV ADQUISICION ADQUISICIONES ADQ
COMPRA COMPRAS CONTRATACION CONTRATACIONES SUMINISTRO SUMINISTROS PRESTACION PRESTACIONES LOCACION PRESTADO PRESTADOS
REQUERIMIENTO ORDEN PEDIDO OFICIO INFORME MEMORANDO SOLICITUD MATERIAL MATERIALES INSUMO INSUMOS ARTICULO ARTICULOS
PRODUCTO PRODUCTOS BIEN BIENES GENERAL GENERALES NRO SUB""".split())
P1 = np.log10(1 + 1 / np.arange(1, 10))                                                     # Benford, primer dígito
P2 = np.array([sum(np.log10(1 + 1 / (10 * k + d)) for k in range(1, 10)) for d in range(10)])  # Benford, segundo dígito


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def objeto(texto):
    """(dos palabras, tres palabras) del objeto de una descripción; vacío si no queda palabra con contenido."""
    u = texto.upper().translate(TABLA)
    m = MARCA.search(u)
    for parte in ((u[m.end():], u) if m else (u,)):
        w = []
        for t in TOKEN.findall(parte):
            if t in CORTAN:
                if w:
                    break
                continue
            if len(t) >= 3 and t not in VACIAS:
                w.append(t)
                if len(w) == 3:
                    break
        if w:
            return " ".join(w[:2]).lower(), " ".join(w).lower()
    return "", ""


def feriados():
    """Feriados nacionales con su año de inicio; no incluye los días no laborables que decreta el Ejecutivo."""
    dias = []
    for a in ANIOS:
        fijos = ["01-01", "05-01", "06-29", "07-28", "07-29", "08-30", "10-08", "11-01", "12-08", "12-25"]
        fijos += ["08-06", "12-09"] if a >= 2022 else []      # Batalla de Junín (Ley 31530) y de Ayacucho
        fijos += ["07-23"] if a >= 2023 else []               # Día de la Fuerza Aérea (Ley 31822, 8/7/2023)
        fijos += ["06-07"] if a >= 2024 else []               # Día de la Bandera (Ley 31788, 15/6/2023)
        dias += [pd.Timestamp(f"{a}-{d}") for d in fijos]
        p = pd.Timestamp(easter(a))
        dias += [p - pd.Timedelta(days=3), p - pd.Timedelta(days=2)]      # Jueves y Viernes Santo
    return pd.DatetimeIndex(dias)


def fraccionamiento(o, col):
    """Por orden: veces que el objeto suma el tope en la entidad y el año, y si es fraccionamiento (2 órdenes o más)."""
    con = (o[col] != "").to_numpy()
    g = o[con].groupby(["ruc_entidad", col])
    s, n, tp = g["monto"].transform("sum"), g["monto"].transform("size"), g["tope"].transform("max")
    veces = np.full(len(o), np.nan)
    veces[con] = (s / tp).to_numpy()
    frac = np.zeros(len(o), dtype=bool)
    frac[con] = ((n >= 2) & (s > tp)).to_numpy()
    return veces, frac


def precio(o, col):
    """z robusta del log del monto frente al mismo objeto en el país y el año (grupos de 20 órdenes o más)."""
    con = (o[col] != "").to_numpy()
    d = pd.DataFrame({"k": o.loc[con, col].to_numpy(), "lm": np.log(np.maximum(o.loc[con, "monto"].to_numpy(float), 1.0))})
    g = d.groupby("k")["lm"]
    n, med = g.transform("size"), g.transform("median")
    mad = (d["lm"] - med).abs().groupby(d["k"]).transform("median")
    z = ((d["lm"] - med) / np.maximum(1.4826 * mad, PISO)).clip(-TOPE_Z, TOPE_Z).where(n >= MIN_PARES)
    out = np.full(len(o), np.nan)
    out[con] = z.to_numpy()
    return out


def benford(o):
    """MAD del primer y segundo dígito por proveedor (30 órdenes o más de 10 soles o más) frente a Benford y al país."""
    m = o["monto"].to_numpy(float)
    ok = m >= 10
    m, r = m[ok], o.loc[ok, "ruc"].to_numpy()
    e = np.floor(np.log10(m) + 1e-9)
    d1 = np.clip(np.floor(m / 10 ** e + 1e-9), 1, 9).astype(int)
    d2 = (np.floor(m / 10 ** (e - 1) + 1e-9) % 10).astype(int)
    n = pd.Series(r).value_counts()
    sel = np.isin(r, n[n >= MIN_BENFORD].index.to_numpy())
    out = []
    for nombre, d, digitos, teorica in (("1d", d1, range(1, 10), P1), ("2d", d2, range(10), P2)):
        nacional = pd.Series(d).value_counts(normalize=True).reindex(digitos, fill_value=0).to_numpy()
        tab = pd.DataFrame({"ruc": r[sel], "d": d[sel]}).groupby(["ruc", "d"]).size().unstack(fill_value=0)
        tab = tab.reindex(columns=list(digitos), fill_value=0)
        prop = tab.div(tab.sum(axis=1), axis=0)
        out.append(pd.DataFrame({f"s_benford_{nombre}": (prop - teorica).abs().mean(axis=1),
                                 f"s_benford_{nombre}_nac": (prop - nacional).abs().mean(axis=1)}))
    return pd.concat(out, axis=1).rename_axis("ruc").reset_index(), float(sel.mean())


def un_anio(a, dias_feriados):
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "fecha", "monto", "tope"],
                        filters=[("anio", "==", a)])
    lo, hi = int(o["id"].min()), int(o["id"].max())
    # los id ya no son contiguos por año (160, C3): se filtra por anio
    filtro = [("anio", "==", a)] if "anio" in pq.read_schema(BASE / "ordenes_texto.parquet").names else [("id", ">=", lo), ("id", "<=", hi)]
    t = pd.read_parquet(BASE / "ordenes_texto.parquet", columns=["id", "descripcion"], filters=filtro)
    o = o.merge(t, on="id", how="left")
    codigos, unicos = pd.factorize(o["descripcion"].fillna(""))
    claves = [objeto(x) for x in unicos]
    o["obj"] = np.array([c[0] for c in claves], dtype=object)[codigos]
    o["obj3"] = np.array([c[1] for c in claves], dtype=object)[codigos]
    o = o.drop(columns="descripcion")
    o["anio"] = a
    for sufijo, col in (("", "obj"), ("3", "obj3")):
        o[f"veces{sufijo}"], frac = fraccionamiento(o, col)
        o[f"m_frac{sufijo}"] = o["monto"] * frac
        o[f"z{sufijo}"] = precio(o, col)
        o[f"z_at{sufijo}"] = (o[f"z{sufijo}"] >= 3).astype(float).where(o[f"z{sufijo}"].notna())
    m = o["monto"].to_numpy(float)
    cent = np.round(m * 100).astype(np.int64)
    o["redondo_100"] = (cent % 10_000 == 0) & (m >= 100)
    o["redondo_1000"] = (cent % 100_000 == 0) & (m >= 1000)
    o["cent"] = cent
    o["repetido"] = o.groupby(K + ["cent"])["id"].transform("size") >= 2
    o["inusual"] = (o["fecha"].dt.dayofweek >= 5) | o["fecha"].dt.normalize().isin(dias_feriados)
    c = o.groupby(K).agg(monto=("monto", "sum"), m_frac=("m_frac", "sum"), m_frac3=("m_frac3", "sum"),
                         s_obj_frac_veces=("veces", "max"), s_obj3_frac_veces=("veces3", "max"),
                         s_redondo_100=("redondo_100", "mean"), s_redondo_1000=("redondo_1000", "mean"),
                         s_monto_repetido=("repetido", "mean"), s_precio_z=("z", "mean"), s_precio_atipico=("z_at", "mean"),
                         s_precio3_z=("z3", "mean"), s_precio3_atipico=("z_at3", "mean"),
                         s_fecha_inusual=("inusual", "mean")).reset_index()
    for sufijo in ("", "3"):
        c[f"s_obj{sufijo}_frac_parte"] = (c[f"m_frac{sufijo}"] / c["monto"].where(c["monto"] > 0)).fillna(0)
        c[f"s_obj{sufijo}_frac_veces"] = c[f"s_obj{sufijo}_frac_veces"].fillna(0)
    b, parte_b = benford(o)
    c = c.drop(columns=["monto", "m_frac", "m_frac3"]).merge(b, on="ruc", how="left")
    top = o.loc[o["obj"] != "", "obj"].value_counts()
    paso(f"   {a}: {len(o):,} órdenes, {len(c):,} casos; con objeto {(o['obj'] != '').mean():.1%}; objetos distintos "
         f"{len(top):,}; órdenes en fraccionamiento por objeto {(o['m_frac'] > 0).mean():.1%}; con pares para el precio "
         f"{o['z'].notna().mean():.1%}; casos con Benford {c['s_benford_1d'].notna().mean():.1%} (órdenes {parte_b:.1%}); "
         f"fecha inusual {o['inusual'].mean():.2%}; redondas a 100 {o['redondo_100'].mean():.1%}")
    if a == 2021:
        paso("   objetos más frecuentes de 2021: " + "; ".join(f"{k} ({v:,})" for k, v in top.head(15).items()))
    return c


DESCRIPCION = {
    "s_obj_frac_parte": "parte del monto del caso en objetos que la entidad fraccionó en el año (objeto de 2 palabras)",
    "s_obj_frac_veces": "veces el tope que suma en la entidad y el año el objeto del caso que más suma (2 palabras)",
    "s_obj3_frac_parte": "lo mismo con objeto de 3 palabras (sensibilidad)",
    "s_obj3_frac_veces": "lo mismo con objeto de 3 palabras (sensibilidad)",
    "s_benford_1d": "MAD del primer dígito del proveedor en el año frente a Benford (30 órdenes o más)",
    "s_benford_2d": "MAD del segundo dígito del proveedor en el año frente a Benford (30 órdenes o más)",
    "s_benford_1d_nac": "MAD del primer dígito frente a la distribución nacional del año (sensibilidad)",
    "s_benford_2d_nac": "MAD del segundo dígito frente a la distribución nacional del año (sensibilidad)",
    "s_redondo_100": "parte de las órdenes del caso con monto múltiplo de 100",
    "s_redondo_1000": "parte de las órdenes del caso con monto múltiplo de 1000",
    "s_monto_repetido": "parte de las órdenes del caso cuyo monto exacto se repite en el caso",
    "s_precio_z": "promedio de la z robusta del monto frente al mismo objeto en el país y el año",
    "s_precio_atipico": "parte de las órdenes del caso con z robusta de 3 o más",
    "s_precio3_z": "lo mismo con objeto de 3 palabras (sensibilidad)",
    "s_precio3_atipico": "lo mismo con objeto de 3 palabras (sensibilidad)",
    "s_fecha_inusual": "parte de las órdenes del caso emitidas en sábado, domingo o feriado nacional",
}


def diccionario(x):
    """Clase propia: `columnas()` no la toma mientras la 149 está en prueba."""
    d = pd.read_csv(V4 / "diccionario_x.csv")
    d = d[~d["variable"].isin(DESCRIPCION)]
    filas = [{"variable": k, "grupo": "estadísticos simples", "tipo": "proporción" if "parte" in k or k.startswith(("s_redondo", "s_monto_rep", "s_fecha", "s_precio_at", "s_precio3_at")) else "conteo o monto",
              "clase_temporal": "contemporánea, en prueba (149)", "fuente": f"órdenes de CONOSCE del año: {v} (HALLAZGOS 149)",
              "lleno_pct": round(100 * float(x[k].notna().mean()), 1)} for k, v in DESCRIPCION.items()]
    pd.concat([d, pd.DataFrame(filas)], ignore_index=True).to_csv(V4 / "diccionario_x.csv", index=False, encoding="utf-8")
    paso(f"diccionario: {len(filas)} filas de la 149")


def main():
    if "--solo-diccionario" in sys.argv:
        diccionario(pd.read_parquet(V4 / "x_simples.parquet"))
        return
    paso("estadísticos simples por caso y año (149)")
    dias = feriados()
    with ProcessPoolExecutor(3) as pool:           # los años son independientes (160, lote 2)
        x = pd.concat(list(pool.map(un_anio, ANIOS, repeat(dias))), ignore_index=True)
    num = [c for c in x.columns if c.startswith("s_")]
    x[num] = x[num].astype("float32")
    x["anio"] = x["anio"].astype("int16")
    if x.duplicated(K).any():
        raise SystemExit("casos repetidos en x_simples")
    x.to_parquet(V4 / "x_simples.parquet", index=False)
    paso(f"guardado: {len(x):,} casos; promedio por variable: " + ", ".join(f"{c} {x[c].mean():.3f}" for c in num))
    if "--sin-diccionario" not in sys.argv:
        diccionario(x)
    paso("listo")


if __name__ == "__main__":
    main()
