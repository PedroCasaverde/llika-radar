# Limpia la conformación jurídica del RNP (ACC-3) y la valida contra el padrón de SUNAT (ACC-6).
# Deja un Parquet por corte en procesados/acc3_conformacion/ y un reporte de calidad para decidir si la fuente sirve.
# Uso: python limpiar_conformacion.py                 (limpia todos los cortes que haya en crudos/)
#      python limpiar_conformacion.py conformacion_juridica_20261013.csv
import json
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path(__file__).resolve().parents[1]
ENTRADA = DATOS / "crudos" / "acc3_conformacion"
SALIDA = DATOS / "procesados" / "acc3_conformacion"
PADRON = sorted((DATOS / "crudos" / "acc6_padron_sunat").glob("*.zip"))[-1]

# Criterio para seguir con la fuente, fijado antes de ver el resultado (13/10/2026)
MIN_USABLE = 0.95          # filas con una llave de identidad confiable
MIN_NOMBRE_SUNAT = 0.90    # personas con RUC 10 cuyo nombre coincide con SUNAT

ROLES = {"SOCIO_ACCIONISTA_TITULAR": "socio_accionista_titular", "REPRESENTANTE_LEGAL": "representante_legal",
         "ORGANO_ADMINISTRACION": "organo_administracion"}
EXTRANJEROS = ("CARNET DE EXTRANJERIA", "PASAPORTE", "DOC. IDENT. EQUIVALENTE AL DNI (solo extranjeros)")


def digito_ruc(base10: pd.Series) -> pd.Series:
    """Dígito verificador del RUC (módulo 11 de SUNAT) para una serie de 10 dígitos."""
    d = np.array([list(map(int, s)) for s in base10], dtype=int).reshape(-1, 10)
    r = 11 - (d @ np.array([5, 4, 3, 2, 7, 6, 5, 4, 3, 2])) % 11
    return pd.Series(np.where(r >= 10, r - 10, r).astype(str), index=base10.index)


def ruc_valido(s: pd.Series) -> pd.Series:
    ok = s.str.fullmatch(r"\d{11}").fillna(False)
    out = pd.Series(False, index=s.index)
    if ok.any():
        out[ok] = digito_ruc(s[ok].str[:10]) == s[ok].str[10]
    return out


def normalizar(s: pd.Series) -> pd.Series:
    """Mayúsculas, sin tildes ni signos, espacios simples. Sirve para comparar nombres entre fuentes."""
    s = s.fillna("").map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode())
    # «S.A.C.» y «SAC» deben quedar iguales: primero se quitan los puntos, después el resto de signos
    return s.str.upper().str.replace(".", "", regex=False).str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.split().str.join(" ")


def leer_padron(rucs: set) -> pd.DataFrame:
    partes = []
    with zipfile.ZipFile(PADRON) as z, z.open("padron_reducido_ruc.txt") as fh:
        for t in pd.read_csv(fh, sep="|", dtype=str, encoding="latin-1", usecols=[0, 1, 2, 3], chunksize=3_000_000,
                             on_bad_lines="skip"):
            t.columns = ["ruc", "nombre_sunat", "estado_sunat", "condicion_sunat"]
            partes.append(t[t["ruc"].isin(rucs)])
    return pd.concat(partes).drop_duplicates("ruc").set_index("ruc")


def misma_persona(x: set, y: set) -> bool:
    return x <= y or y <= x or len(x & y) / len(x | y) >= 0.6


def comparar_nombres(a: pd.Series, b: pd.Series) -> pd.Series:
    """igual / mismo_orden_distinto / parecido / distinto / sin_dato, por coincidencia de palabras.
    «parecido» = un nombre contiene al otro (falta un segundo nombre) o comparten 60 % de las palabras.
    Dos hermanos con los mismos apellidos («PEREZ GARCIA JUAN» y «PEREZ GARCIA MARIA») salen «distinto»."""
    na, nb = normalizar(a), normalizar(b)
    out = pd.Series("sin_dato", index=a.index)
    hay = (na != "") & (nb != "") & b.notna()
    ta, tb = na[hay].str.split().map(set), nb[hay].str.split().map(set)
    parecido = pd.Series([misma_persona(x, y) for x, y in zip(ta, tb)], index=ta.index)
    out[hay] = np.select([na[hay] == nb[hay], ta == tb, parecido], ["igual", "mismo_orden_distinto", "parecido"], "distinto")
    return out


def dni_ambiguo(dni: pd.Series, nombre: pd.Series) -> pd.Series:
    """True si el mismo DNI aparece con nombres que no son de la misma persona."""
    d = pd.DataFrame({"dni": dni, "nn": normalizar(nombre)})[dni.notna()]
    nombres = d.groupby("dni")["nn"].agg(lambda s: s.value_counts().index.tolist())
    multi = nombres[nombres.map(len) > 1]
    amb = multi.map(lambda ns: any(not misma_persona(set(ns[0].split()), set(x.split())) for x in ns[1:]))
    return dni.map(amb).fillna(False).astype(bool)


def leer_crudo(crudo: Path) -> tuple[pd.DataFrame, int, int]:
    """Lee un corte. Si alguna fila trae un separador vacío de más (pasa en el corte de junio de 2024: 30 filas con «||»),
    se quita el primer campo vacío entre los cinco primeros; si aun así no cuadra, la fila se descarta. Devuelve
    (tabla, filas reparadas, filas descartadas)."""
    try:
        return pd.read_csv(crudo, sep="|", dtype=str, encoding="cp1252", encoding_errors="replace"), 0, 0
    except pd.errors.ParserError:
        pass
    lineas, reparadas, descartadas = [], 0, 0
    with open(crudo, encoding="cp1252", errors="replace", newline="") as fh:
        for linea in fh:
            campos = linea.rstrip("\r\n").split("|")
            if len(campos) == 10:
                vacio = next((i for i in range(1, 6) if campos[i].strip() == ""), None)
                if vacio is not None:
                    del campos[vacio]
                    reparadas += 1
            if len(campos) != 9:
                descartadas += 1
                continue
            lineas.append(campos)
    return pd.DataFrame(lineas[1:], columns=lineas[0]), reparadas, descartadas


def rucs_necesarios(crudo: Path) -> set:
    """RUC que hay que buscar en el padrón para un corte: proveedores, integrantes con RUC y el RUC 10 de cada DNI."""
    c = leer_crudo(crudo)[0]
    c.columns = [x.strip() for x in c.columns]
    num = c["NUMERO_DOCUMENTO"].fillna("").str.replace(r"\D", "", regex=True)
    dni = num[c["TIPO_DOCUMENTO"].str.strip().str.startswith("DOC. NACIONAL", na=False) & num.str.len().eq(8)]
    dni = pd.concat([dni, num[num.str.len().eq(11) & num.str.startswith("10")].str[2:10]]).drop_duplicates()
    return set(c["RUC"].str.strip()) | set(num[num.str.len().eq(11)]) | set("10" + dni + digito_ruc("10" + dni))


def limpiar(crudo: Path, pad: pd.DataFrame | None = None) -> dict:
    c, reparadas, descartadas = leer_crudo(crudo)
    c.columns = [x.strip() for x in c.columns]
    c = c.apply(lambda s: s.str.strip())  # quita el tabulador que trae NUMERO_DOCUMENTO
    corte = c["FECHA_CORTE"].iloc[0]

    df = pd.DataFrame({
        "fecha_corte": pd.to_datetime(c["FECHA_CORTE"], format="%Y%m%d"),
        "ruc_proveedor": c["RUC"],
        "rol": c["TIPO_CONF_JURIDICA"].map(ROLES),
        "tipo_doc_declarado": c["TIPO_DOCUMENTO"],
        "doc_original": c["NUMERO_DOCUMENTO"],
        "nombre_integrante": c["NOMBRE_RAZONODENOMINACIONSOCIAL"].str.upper().str.split().str.join(" "),
        # no es la fecha del cargo: es una por proveedor y cambia con cada trámite en el RNP
        "fecha_ultimo_tramite_rnp": pd.to_datetime(c["FECHA_INICIO_VIGENCIA"], format="%Y%m%d", errors="coerce"),
        "forma_societaria": c["DE_FORMA_SOCIETARIA"],
    })
    num = df["doc_original"].fillna("").str.replace(r"\D", "", regex=True)
    declarado_dni = df["tipo_doc_declarado"].str.startswith("DOC. NACIONAL", na=False)
    extranjero = df["tipo_doc_declarado"].isin(EXTRANJEROS)

    # el tipo de integrante se decide por el número, no por el tipo declarado
    once = num.str.len().eq(11)
    df["tipo_integrante"] = np.select(
        [once & num.str.startswith("20"), once & num.str.startswith("10"), once & num.str[:2].isin(["15", "17"]),
         declarado_dni & num.str.len().eq(8), extranjero],
        ["empresa_ruc20", "persona_ruc10", "persona_ruc15_17", "persona_dni", "persona_extranjera"], "documento_invalido")
    df["dni_integrante"] = np.where(df["tipo_integrante"] == "persona_dni", num,
                                    np.where(df["tipo_integrante"] == "persona_ruc10", num.str[2:10], None))
    df["ruc_integrante"] = np.where(once, num, None)
    df["doc_extranjero"] = np.where(df["tipo_integrante"] == "persona_extranjera", df["doc_original"].str.upper(), None)

    # banderas de calidad
    df["tipo_inconsistente"] = (declarado_dni & once) | (df["tipo_doc_declarado"].str.startswith("REG. UNICO", na=False) & ~once)
    df["autorreferencia"] = num == df["ruc_proveedor"]
    df["ruc_proveedor_valido"] = ruc_valido(df["ruc_proveedor"])
    df["ruc_integrante_valido"] = ruc_valido(df["ruc_integrante"].fillna(""))
    df["duplicado"] = df.duplicated(["ruc_proveedor", "tipo_integrante", "doc_original", "rol"], keep="first")
    df["texto_danado"] = df["nombre_integrante"].str.contains("�|Ã", na=False, regex=True)

    # validación externa con SUNAT: proveedor, integrante empresa y persona con RUC 10 (10 + DNI + dígito)
    dni_ok = df["dni_integrante"].notna()
    ruc10 = pd.Series(None, index=df.index, dtype=object)
    ruc10[dni_ok] = "10" + df.loc[dni_ok, "dni_integrante"] + digito_ruc("10" + df.loc[dni_ok, "dni_integrante"])
    df["ruc10_de_la_persona"] = ruc10
    if pad is None:
        pad = leer_padron(set(df["ruc_proveedor"]) | set(df["ruc_integrante"].dropna()) | set(ruc10.dropna()))
    df["razon_social_proveedor"] = df["ruc_proveedor"].map(pad["nombre_sunat"])
    df["estado_sunat_proveedor"] = df["ruc_proveedor"].map(pad["estado_sunat"])
    df["condicion_sunat_proveedor"] = df["ruc_proveedor"].map(pad["condicion_sunat"])
    nombre_ref = df["ruc_integrante"].map(pad["nombre_sunat"]).where(df["ruc_integrante"].notna(), ruc10.map(pad["nombre_sunat"]))
    df["nombre_vs_sunat"] = comparar_nombres(df["nombre_integrante"], nombre_ref)

    df["dni_ambiguo"] = dni_ambiguo(df["dni_integrante"], df["nombre_integrante"])

    # una fila es usable para cruzar si su llave de identidad es confiable. En empresas integrantes el nombre no se
    # exige (cambian de razón social: «Bolsa de Valores de Lima» hoy es «Grupo BVL»); basta el RUC con dígito válido.
    persona = df["tipo_integrante"].isin(["persona_dni", "persona_ruc10"])
    base = (~df["autorreferencia"] & ~df["duplicado"] & df["ruc_proveedor_valido"]
            & (df["ruc_integrante"].isna() | df["ruc_integrante_valido"]))
    df["usable_para_cruce"] = (
        base & df["tipo_integrante"].isin(["persona_dni", "persona_ruc10", "empresa_ruc20", "persona_ruc15_17"])
        & ~(persona & df["nombre_vs_sunat"].eq("distinto")) & ~df["dni_ambiguo"])
    # para unir empresas entre sí también sirven los carnés de extranjería y pasaportes (no cruzan con RUC 10)
    df["usable_red_empresas"] = df["usable_para_cruce"] | (base & df["tipo_integrante"].eq("persona_extranjera"))

    SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA / f"conformacion_corte{corte}.parquet", index=False)

    # reporte
    n = len(df)
    pct = lambda m: round(float(m.sum()) / n * 100, 2)  # noqa: E731
    personas_ruc10 = df["dni_integrante"].notna() & ruc10.map(pad["nombre_sunat"]).notna() & df["ruc_integrante"].isna()
    nom = df.loc[personas_ruc10, "nombre_vs_sunat"].value_counts(normalize=True)
    coincide_ruc10 = float(nom.reindex(["igual", "mismo_orden_distinto", "parecido"]).fillna(0).sum())
    nombres_por_dni = df[dni_ok].assign(nn=normalizar(df.loc[dni_ok, "nombre_integrante"])).groupby("dni_integrante")["nn"].nunique()
    r = {
        "archivo": crudo.name, "fecha_corte": corte, "filas": n, "proveedores": int(df["ruc_proveedor"].nunique()),
        "filas_reparadas_separador_de_mas": reparadas, "filas_descartadas_por_formato": descartadas,
        "tipo_integrante": df["tipo_integrante"].value_counts().to_dict(),
        "problemas_pct_filas": {
            "tipo_declarado_inconsistente": pct(df["tipo_inconsistente"]), "autorreferencia": pct(df["autorreferencia"]),
            "documento_invalido": pct(df["tipo_integrante"].eq("documento_invalido")), "duplicado": pct(df["duplicado"]),
            "ruc_proveedor_con_digito_invalido": pct(~df["ruc_proveedor_valido"]),
            "ruc_integrante_con_digito_invalido": pct(df["ruc_integrante"].notna() & ~df["ruc_integrante_valido"]),
            "nombre_distinto_a_sunat": pct(df["nombre_vs_sunat"].eq("distinto")), "texto_danado": pct(df["texto_danado"])},
        "conteos": {"tipo_declarado_inconsistente": int(df["tipo_inconsistente"].sum()), "autorreferencia": int(df["autorreferencia"].sum()),
                    "documento_invalido": int(df["tipo_integrante"].eq("documento_invalido").sum()), "duplicado": int(df["duplicado"].sum()),
                    "nombre_distinto_a_sunat": int(df["nombre_vs_sunat"].eq("distinto").sum()), "texto_danado": int(df["texto_danado"].sum())},
        "proveedor_en_sunat_pct": round(float(df.drop_duplicates("ruc_proveedor")["razon_social_proveedor"].notna().mean() * 100), 2),
        "proveedor_estado_sunat": df.drop_duplicates("ruc_proveedor")["estado_sunat_proveedor"].value_counts().head(4).to_dict(),
        "personas_con_ruc10_en_sunat": int(personas_ruc10.sum()),
        "nombre_vs_sunat_en_personas_con_ruc10": {k: round(float(v) * 100, 2) for k, v in nom.items()},
        "empresas_integrantes_nombre_vs_sunat": df.loc[df["tipo_integrante"].eq("empresa_ruc20"), "nombre_vs_sunat"].value_counts().to_dict(),
        "dni_con_mas_de_un_nombre_pct": round(float((nombres_por_dni > 1).mean() * 100), 2),
        "dni_ambiguos": int(df.loc[df["dni_ambiguo"], "dni_integrante"].nunique()),
        "filas_con_dni_ambiguo_pct": pct(df["dni_ambiguo"]),
        "personas_dni_nombre_distinto_a_sunat_pct": pct(persona & df["nombre_vs_sunat"].eq("distinto")),
        "usable_para_cruce_pct": pct(df["usable_para_cruce"]),
        "usable_red_empresas_pct": pct(df["usable_red_empresas"]),
        "criterio": {"min_usable_pct": MIN_USABLE * 100, "min_nombre_sunat_pct": MIN_NOMBRE_SUNAT * 100},
    }
    r["veredicto"] = ("seguir" if r["usable_para_cruce_pct"] >= MIN_USABLE * 100 and coincide_ruc10 >= MIN_NOMBRE_SUNAT
                      else "revisar antes de usar")
    json.dump(r, open(SALIDA / f"calidad_corte{corte}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return r


if __name__ == "__main__":
    # para la historia: python limpiar_conformacion.py _historia/conformacion_juridica_v28_corte20240101.csv ...
    archivos = [ENTRADA / a for a in sys.argv[1:]] or sorted(ENTRADA.glob("conformacion_juridica_*.csv"))
    # con varios cortes, el padrón (1,6 GB) se lee una sola vez para todos
    pad = leer_padron(set().union(*(rucs_necesarios(a) for a in archivos))) if len(archivos) > 1 else None
    for a in archivos:
        print(json.dumps(limpiar(a, pad), ensure_ascii=False, indent=1), flush=True)
