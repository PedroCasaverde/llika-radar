"""Paso 25: tarjeta territorial de autoridades electas (HALLAZGOS 70.11 y 71; Pedro aprobó construirla el 1/11 como dato
para el benchmark, no como alerta del radar todavía).

Con la lista del JNE (autoridades regionales y municipales electas en 2014, 2018 y 2022, con DNI) marca cada orden de
CONOSCE en que:
1. **autoridad_misma**: el proveedor (RUC 10, que lleva el DNI) es una autoridad y la entidad está en su territorio,
   mientras ejerce o hasta 12 meses después; o es alcalde, gobernador o vicegobernador y le vende a cualquier entidad
   mientras ejerce (DL 1444, desde el 30/01/2019; el texto de la ley está transcrito en las resoluciones);
2. **empresa_de_autoridad**: el proveedor es una empresa donde una autoridad es socia, representante u órgano de
   administración (conformación jurídica del OECE, foto de octubre de 2026), en el mismo territorio y tiempo;
3. **pariente_posible**: el proveedor persona natural tiene los mismos dos apellidos que una autoridad del territorio,
   en el mismo tiempo. Lleva la frecuencia nacional de ese par de apellidos (padrón de SUNAT): cuanto más raro, más
   probable el parentesco. La probabilidad calibrada queda para el benchmark.
Territorio: distrito (alcalde y regidores distritales), provincia (provinciales) o región (consejeros, gobernador).

    python -I -X utf8 25_tarjeta_territorial.py
Trae DNI de autoridades: queda en datos/procesados/radar/ (fuera de git). En pantalla, solo agregados.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comun import CRUDOS, SALIDA, entidades, norm_serie  # noqa: E402

LOCALES = ["ALCALDE DISTRITAL", "ALCALDE PROVINCIAL", "REGIDOR DISTRITAL", "REGIDOR PROVINCIAL", "CONSEJERO REGIONAL",
           "GOBERNADOR REGIONAL", "VICEGOBERNADOR REGIONAL", "PRESIDENTE REGIONAL", "VICEPRESIDENTE REGIONAL"]
DL1444 = pd.Timestamp("2019-01-30")
PART = {"DE", "DEL", "LA", "LAS", "LOS", "Y", "SAN", "SANTA", "DA", "DI", "VON", "VAN", "MC"}


def apellidos(nombre):
    t, out, i = nombre.split(), [], 0
    while len(out) < 2 and i < len(t):
        u = [t[i]]
        i += 1
        while u[-1] in PART and i < len(t):
            u.append(t[i])
            i += 1
        out.append(" ".join(u))
    return (out + ["", ""])[:2]


def autoridades():
    a = pd.read_excel(CRUDOS / "jne_autoridades" / "Autoridades_Electas_20251113.xls", dtype=str)
    a = a[a["CARGO"].isin(LOCALES)].rename(columns={"DOCUMENTOIDENTIDAD": "dni_aut"}).copy()
    a["ini"] = pd.to_datetime(a["FEINICIOVIGENCIA"], errors="coerce")
    a["fin"] = pd.to_datetime(a["FEFINVIGENCIA"], errors="coerce").dt.normalize()
    a["hasta"] = a["fin"] + pd.DateOffset(months=12)
    a["nivel"] = np.select([a["CARGO"].str.contains("DISTRITAL"), a["CARGO"].str.contains("PROVINCIAL")],
                           ["distrito", "provincia"], "region")
    a["eR"], a["eP"], a["eD"] = (norm_serie(a[c].fillna("")) for c in ("REGION", "PROVINCIA", "DISTRITO"))
    a["ap"], a["am"] = norm_serie(a["APELLIDOPATERNO"].fillna("")), norm_serie(a["APELLIDOMATERNO"].fillna(""))
    a["todo_el_pais"] = a["CARGO"].str.startswith("ALCALDE") | a["CARGO"].str.contains("GOBERNADOR|PRESIDENTE")
    return a.dropna(subset=["ini", "fin"])


def en_territorio(m):
    return (m["eR"] == m["R"]) & ((m["nivel"] == "region") |
                                  ((m["eP"] == m["P"]) & ((m["nivel"] == "provincia") | (m["eD"] == m["D"]))))


def ventana(m):
    mand = (m["fecha"] >= m["ini"]) & (m["fecha"] <= m["fin"])
    post = (m["fecha"] > m["fin"]) & (m["fecha"] <= m["hasta"])
    terr = en_territorio(m)
    fuera = mand & m["todo_el_pais"] & ~terr & (m["fecha"] >= DL1444)
    m["momento"] = np.select([terr & mand, terr & post, fuera],
                             ["en_su_territorio_durante", "en_su_territorio_12_meses_despues",
                              "fuera_de_su_territorio_durante"], "")
    return m[m["momento"] != ""]


def main():
    a = autoridades()
    aa = a.rename(columns={"eR": "R", "eP": "P", "eD": "D"})
    e = entidades()[["RUC", "DEPARTAMENTO", "PROVINCIA", "DISTRITO"]].rename(columns={"RUC": "ruc_entidad"})
    e["eR"], e["eP"], e["eD"] = (norm_serie(e[c].fillna("")) for c in ("DEPARTAMENTO", "PROVINCIA", "DISTRITO"))
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "nombre", "fecha", "monto"])
    o = o.merge(e[["ruc_entidad", "eR", "eP", "eD"]], on="ruc_entidad", how="left")
    cols = ["dni_aut", "CARGO", "nivel", "R", "P", "D", "ini", "fin", "hasta", "todo_el_pais"]
    salidas = []

    # 1) la autoridad misma
    pn = o[o["ruc"].str.startswith("10")].assign(dni=lambda x: x["ruc"].str[2:10])
    m = ventana(pn.merge(aa[cols], left_on="dni", right_on="dni_aut"))
    salidas.append(m.assign(tipo="autoridad_misma", par="", freq_par=np.nan))

    # 2) empresa de la autoridad (socio, representante u órgano)
    conf = pd.read_csv(CRUDOS / "acc3_conformacion" / "conformacion_juridica_20261013.csv", sep="|", dtype=str,
                       encoding="cp1252", encoding_errors="replace", usecols=["NUMERO_DOCUMENTO", "RUC"])
    conf = conf[conf["NUMERO_DOCUMENTO"].str.fullmatch(r"\d{8}", na=False)].rename(columns={"RUC": "ruc"})
    emp = o[~o["ruc"].str.startswith("10")].merge(conf.drop_duplicates(), on="ruc")
    m = ventana(emp.merge(aa[cols], left_on="NUMERO_DOCUMENTO", right_on="dni_aut"))
    salidas.append(m.assign(tipo="empresa_de_autoridad", par="", freq_par=np.nan))

    # 3) posible pariente: mismos dos apellidos que una autoridad del territorio
    nom = pn[["ruc", "nombre"]].drop_duplicates("ruc")
    nom["n"] = norm_serie(nom["nombre"].fillna(""))
    p12 = np.array([apellidos(x) for x in nom["n"]], dtype=object)
    nom["par"] = pd.Series(p12[:, 0], index=nom.index) + " " + pd.Series(p12[:, 1], index=nom.index)
    aa["par"] = aa["ap"] + " " + aa["am"]
    pp = pn.merge(nom[["ruc", "par"]], on="ruc")
    filas = []
    for nivel, llave in (("distrito", ["R", "P", "D"]), ("provincia", ["R", "P"]), ("region", ["R"])):
        q = aa[aa["nivel"] == nivel][cols + ["par"]]
        izq = pp.rename(columns={"eR": "R", "eP": "P", "eD": "D"})
        k = izq.merge(q, on=llave + ["par"], suffixes=("", "_a"))
        k = k[k["dni"] != k["dni_aut"]]
        k["eR"], k["eP"], k["eD"] = k["R"], k["P"], k["D"]
        for c in ("P", "D"):
            if c not in llave:
                k[c] = k[f"{c}_a"]
        filas.append(ventana(k))
    m = pd.concat(filas)
    m = m[m["momento"] != "fuera_de_su_territorio_durante"]        # el pariente solo está impedido en el territorio
    freq = pd.read_parquet(SALIDA / "padron_pares.parquet").groupby("par")["n"].sum()
    salidas.append(m.assign(tipo="pariente_posible", freq_par=m["par"].map(freq)))

    t = pd.concat(salidas, ignore_index=True)
    t = t[["id", "ruc_entidad", "ruc", "fecha", "monto", "tipo", "momento", "CARGO", "nivel", "dni_aut", "par",
           "freq_par"]].rename(columns={"CARGO": "cargo"})
    t = t.sort_values(["tipo", "freq_par"]).drop_duplicates(["id", "tipo"])
    t.to_parquet(SALIDA / "tarjeta_territorial.parquet", index=False)
    g = t.groupby(["tipo", "momento"]).agg(ordenes=("id", "size"), proveedores=("ruc", "nunique"),
                                           monto_millones=("monto", lambda x: round(x.sum() / 1e6, 1)))
    print(g.to_string())
    pp_ = t[t["tipo"] == "pariente_posible"]
    print(f"pariente posible con par de apellidos raro (100 personas o menos en el padrón): "
          f"{(pp_['freq_par'] <= 100).sum():,} órdenes, {pp_.loc[pp_['freq_par'] <= 100, 'ruc'].nunique():,} proveedores")


if __name__ == "__main__":
    main()
