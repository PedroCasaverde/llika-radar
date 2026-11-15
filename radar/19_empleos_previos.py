"""Empleos previos de los declarantes (ACC-5, Reporte3), listos para la alerta de «puerta giratoria» (PENDIENTES.md 5).

Une cada empleo previo con su declaración (ACC-4, Reporte1): quién es, en qué entidad y cargo está y desde cuándo.
Cubre todos los años en que la declaración muestra el cargo vigente (2018-2026); las declaraciones existen desde 2022,
así que quien dejó el cargo antes de 2022 no aparece (cobertura menor en 2018-2020: ver cobertura_por_anio.csv).

Controles de calidad:
- RUC del empleador: 11 dígitos, dígito verificador y presencia en el padrón de SUNAT.
- Público o privado: el RUC está (o no) en el maestro de entidades del Estado.
- Fechas: Reporte3 trae años de 4 cifras con errores de tipeo (0022, 9183, 2105...): se aceptan de 1950 a la fecha de
  presentación de la declaración (más un año); el resto queda vacío y marcado como corregido. Sin fecha de cese, el
  empleo se toma como vigente.
- Una misma persona repite sus empleos en cada declaración anual: se deja una fila por persona, empleador y periodo.

Uso: python -X utf8 19_empleos_previos.py
Salida: datos/procesados/radar/empleos_previos.parquet, padron_rucs.parquet (reutilizable) y cobertura_por_anio.csv
"""
import time
import zipfile

import numpy as np
import pandas as pd

from comun import CORTE_DJI, DECIDE, LOCADOR, SALIDA, entidades, fecha_dji, norm_serie, ruc_por_nombre, ultimo

PESOS = np.array([5, 4, 3, 2, 7, 6, 5, 4, 3, 2])


def ruc_valido(serie):
    """11 dígitos y dígito verificador (módulo 11) correctos."""
    s = serie.fillna("")
    ok = s.str.fullmatch(r"(10|15|17|20)\d{9}")
    d = np.array([list(map(int, x)) for x in s[ok]]) if ok.any() else np.empty((0, 11), int)
    resultado = pd.Series(False, index=s.index)
    if len(d):
        r = 11 - (d[:, :10] @ PESOS) % 11
        r = np.where(r == 10, 0, np.where(r == 11, 1, r))
        resultado[ok] = r == d[:, 10]
    return resultado


def padron_rucs():
    """RUC, estado y condición de domicilio del padrón reducido de SUNAT, en Parquet para no releer el zip."""
    destino = SALIDA / "padron_rucs.parquet"
    zip_ = ultimo("acc6_padron_sunat", "*.zip")
    if destino.exists() and destino.stat().st_mtime > zip_.stat().st_mtime:
        return pd.read_parquet(destino)
    partes = []
    with zipfile.ZipFile(zip_) as z, z.open("padron_reducido_ruc.txt") as fh:
        for t in pd.read_csv(fh, sep="|", dtype=str, encoding="latin-1", usecols=[0, 2, 3], header=0,
                             names=["ruc", "estado", "condicion"], chunksize=4_000_000, on_bad_lines="skip"):
            partes.append(t)
    p = pd.concat(partes, ignore_index=True).drop_duplicates("ruc")
    p.to_parquet(destino, index=False)
    return p


def main():
    t0 = time.time()
    # 1) Declaraciones: persona, entidad (RUC por nombre, regla del 15/10), cargo y su periodo
    r1 = pd.read_csv(ultimo("acc4_dji_declaraciones"), dtype=str, engine="pyarrow",
                     usecols=["CODIGO_DDJJ", "APELLIDO_PATERNO", "APELLIDO_MATERNO", "NOMBRES", "ENTIDAD", "CARGO",
                              "FECHA_PRESENTACION_DDJJ", "FECHA_INICIO_CARGO", "FECHA_CESE_CARGO"])
    r1["presentacion"] = fecha_dji(r1["FECHA_PRESENTACION_DDJJ"])
    r1["inicio"] = fecha_dji(r1["FECHA_INICIO_CARGO"], r1["presentacion"])
    r1["cese"] = fecha_dji(r1["FECHA_CESE_CARGO"], r1["presentacion"]).fillna(CORTE_DJI)
    ent = entidades()
    r1["ruc_entidad"] = ruc_por_nombre(r1["ENTIDAD"], ent)
    r1["persona"] = (norm_serie(r1["APELLIDO_PATERNO"].fillna("")) + " " + norm_serie(r1["APELLIDO_MATERNO"].fillna(""))
                     + " " + norm_serie(r1["NOMBRES"].fillna(""))).str.strip()
    r1["cargo"] = norm_serie(r1["CARGO"].fillna(""))
    r1["decide"] = r1["cargo"].str.contains(DECIDE, regex=True)
    r1["locador"] = r1["cargo"].str.contains(LOCADOR, regex=True) & ~r1["decide"]
    print(f"Declaraciones: {len(r1):,}; con entidad identificada: {r1.ruc_entidad.notna().mean():.1%} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # Cobertura: declaraciones con el cargo vigente en cada año
    cob = []
    for y in range(2018, 2027):
        v = r1.inicio.notna() & (r1.inicio <= f"{y}-12-31") & (r1.cese >= f"{y}-01-01")
        cob.append({"anio": y, "declaraciones_vigentes": int(v.sum()),
                    "personas_vigentes": int(r1.loc[v, "persona"].nunique())})
    pd.DataFrame(cob).to_csv(SALIDA / "cobertura_por_anio.csv", index=False)

    # 2) Empleos previos
    r3 = pd.read_csv(ultimo("acc5_dji_empleos"), dtype=str, engine="pyarrow")
    r3["ruc_empleador"] = r3["RUC_ENTIDAD_LABORO"].fillna("").str.replace(r"\D", "", regex=True)
    r3 = r3.merge(r1[["CODIGO_DDJJ", "persona", "ruc_entidad", "cargo", "decide", "locador", "inicio", "cese",
                      "presentacion"]], on="CODIGO_DDJJ", how="inner")
    r3["inicio_previo"] = pd.to_datetime(r3["FECHA_INICIO_CARGO"], format="%d/%m/%Y", errors="coerce")
    r3["cese_previo"] = pd.to_datetime(r3["FECHA_CESE_CARGO"], format="%d/%m/%Y", errors="coerce")
    for c in ("inicio_previo", "cese_previo"):
        malo = (r3[c] < "1950-01-01") | (r3[c] > r3["presentacion"] + pd.DateOffset(years=1))
        r3[f"{c}_corregido"] = malo
        r3.loc[malo, c] = pd.NaT
    r3["vigente"] = r3["FECHA_CESE_CARGO"].isna() | r3["cese_previo"].isna() & ~r3["cese_previo_corregido"]
    r3["ruc_valido"] = ruc_valido(r3["ruc_empleador"])
    pad = padron_rucs()
    r3["en_padron"] = r3["ruc_empleador"].isin(set(pad.ruc))
    publicos = set(ent["RUC"].dropna())
    r3["empleador_publico"] = r3["ruc_empleador"].isin(publicos)
    r3["nombre_empleador"] = r3["NOMBRE_ENTIDAD_LABORO"]
    r3["cargo_previo"] = norm_serie(r3["CARGO_ENTIDAD_LABORO"].fillna(""))
    out = r3[["persona", "ruc_entidad", "cargo", "decide", "locador", "inicio", "cese", "ruc_empleador", "ruc_valido",
              "en_padron", "empleador_publico", "nombre_empleador", "cargo_previo", "inicio_previo", "cese_previo",
              "vigente"]].drop_duplicates(["persona", "ruc_entidad", "ruc_empleador", "inicio_previo", "cese_previo"])
    out.to_parquet(SALIDA / "empleos_previos.parquet", index=False)

    # Controles
    print(f"Empleos previos: {len(r3):,} filas -> {len(out):,} sin repetir ({time.time() - t0:.0f} s)")
    print(f"  RUC válido: {out.ruc_valido.mean():.1%} | en el padrón: {out.en_padron.mean():.1%} | "
          f"empleador público: {out.empleador_publico.mean():.1%} | "
          f"privado con RUC válido: {(out.ruc_valido & ~out.empleador_publico).mean():.1%}")
    print(f"  Fechas corregidas: inicio {r3.inicio_previo_corregido.mean():.2%}, cese {r3.cese_previo_corregido.mean():.2%}")
    print(f"  Personas: {out.persona.nunique():,}; con al menos un empleador privado: "
          f"{out.loc[out.ruc_valido & ~out.empleador_publico, 'persona'].nunique():,}")
    print("Cobertura por año:", {c['anio']: c['personas_vigentes'] for c in cob})


if __name__ == "__main__":
    main()
