"""Rutas y funciones que comparten los scripts del radar de LLIKA."""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
CRUDOS = DATOS / "crudos"
PROC = DATOS / "procesados"
SALIDA = PROC / "radar"
UIT = {2018: 4150, 2019: 4200, 2020: 4300, 2021: 4400, 2022: 4600, 2023: 4950, 2024: 5150, 2025: 5350}
ANIOS = list(UIT)
CORTE_DJI = pd.Timestamp("2026-07-15")  # las declaraciones juradas están actualizadas a esta fecha
DECIDE = (r"\b(?:ALCALD|REGIDOR|GOBERNAD|CONSEJER|GERENT|SUB ?GERENT|DIRECTOR|JEFE|JEFA|LOGIST|ABASTEC|ADMINISTRADOR"
          r"|PRESIDENT|SECRETARI[OA] GENERAL|COMITE|TITULAR|VICEMINISTR|MINISTR)")
# quien declara pero no es empleado: locadores, colaboradores, consultores, árbitros, miembros de juntas de disputas;
# a veces el «cargo» es la descripción del servicio contratado («CONTRATACION DEL SERVICIO DE ...»)
LOCADOR = (r"^(?:SERVICIO|CONTRATACION)|\b(?:COLABORADOR|LOCADOR|LOCACION|OTRO CARGO|CONSULTOR|PRACTICANTE|DAB)\b"
           r"|\bARBITR|\bCONCILIA")


def ultimo(carpeta, patron="*"):
    return sorted((CRUDOS / carpeta).glob(patron))[-1]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", s).split())


def norm_serie(s):
    """Normaliza solo los valores distintos y los vuelve a repartir (rápido con millones de filas repetidas)."""
    u = pd.Series(s.unique())
    return s.map(dict(zip(u, u.map(norm))))


def fecha_dji(s, ref=None):
    f = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    if ref is not None:  # año de dos cifras: si queda después de la presentación, es del siglo anterior
        f = f.where(~(f > ref), f - pd.DateOffset(years=100))
    return f


def entidades():
    ent = pd.read_csv(ultimo("acc9_entidades"), sep="|", dtype=str, encoding="cp1252", encoding_errors="replace")
    ent["n1"] = ent["NOMBRE_DE_ENTIDAD"].map(norm)
    ent["n2"] = ent["NOMBRE_DE_ENTIDAD"].str.split(" - ").str[0].map(norm)
    return ent


def ruc_por_nombre(nombres, ent):
    """Regla del 15/10 (H3′): nombre normalizado completo, y si no, la parte antes de « - »; solo nombres únicos."""
    u1 = ent.drop_duplicates("n1", keep=False).set_index("n1")["RUC"]
    u2 = ent.drop_duplicates("n2", keep=False).set_index("n2")["RUC"]
    return nombres.map(norm).map(u1).fillna(nombres.str.split(" - ").str[0].map(norm).map(u2))


def funcionarios():
    """Una fila por persona, entidad y cargo, con el periodo del cargo (abierto hasta el corte si no tiene cese)."""
    r1 = pd.read_csv(ultimo("acc4_dji_declaraciones"), dtype=str, encoding="utf-8",
                     usecols=["CODIGO_DDJJ", "APELLIDO_PATERNO", "APELLIDO_MATERNO", "NOMBRES", "ENTIDAD", "CARGO",
                              "FECHA_PRESENTACION_DDJJ", "FECHA_INICIO_CARGO", "FECHA_CESE_CARGO"])
    pres = fecha_dji(r1["FECHA_PRESENTACION_DDJJ"])
    r1["inicio"] = fecha_dji(r1["FECHA_INICIO_CARGO"], pres)
    r1["cese"] = fecha_dji(r1["FECHA_CESE_CARGO"], pres).fillna(CORTE_DJI)
    ent = entidades()
    r1["ruc_entidad"] = ruc_por_nombre(r1["ENTIDAD"], ent)
    r1["ap"], r1["am"] = norm_serie(r1["APELLIDO_PATERNO"].fillna("")), norm_serie(r1["APELLIDO_MATERNO"].fillna(""))
    r1["persona"] = (r1["ap"] + " " + r1["am"] + " " + norm_serie(r1["NOMBRES"].fillna(""))).str.strip()
    r1["par"] = r1["ap"] + " " + r1["am"]
    r1["cargo"] = norm_serie(r1["CARGO"].fillna(""))
    r1["decide"] = r1["cargo"].str.contains(DECIDE, regex=True)
    r1["locador"] = r1["cargo"].str.contains(LOCADOR, regex=True) & ~r1["decide"]
    f = r1[r1["ruc_entidad"].notna() & r1["inicio"].notna() & (r1["cese"] >= r1["inicio"])]
    f = f.merge(ent[["RUC", "DEPARTAMENTO"]].rename(columns={"RUC": "ruc_entidad", "DEPARTAMENTO": "depto"}),
                on="ruc_entidad", how="left")
    cols = ["persona", "ap", "am", "par", "ruc_entidad", "depto", "cargo", "decide", "locador", "inicio", "cese"]
    return f[cols].drop_duplicates(["persona", "ruc_entidad", "inicio", "cese"]).reset_index(drop=True)


def ordenes(columnas=None):
    return pd.read_parquet(SALIDA / "ordenes.parquet", columns=columnas)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - m, c + m
