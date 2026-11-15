"""Validación cruzada de lo extraído de las resoluciones del Tribunal (Claude) y de los informes de control (DeepSeek)
(HALLAZGOS 71). Deja una tabla por tipo de dato, con banderas de qué se pudo comprobar contra otra fuente.

  python -X utf8 validar_extracciones.py

Comprobaciones:
- RUC con su dígito verificador; DNI de 8 dígitos.
- Resoluciones: el administrado en el registro de sanciones (ACC-7) y su orden en CONOSCE (por proveedor, tipo, año y
  número); la fecha de los hechos no puede ser posterior a la resolución.
- Informes: la entidad del informe por su nombre en la ficha; cada orden en CONOSCE (por entidad, tipo, año y número) y,
  si calza, si el monto coincide; el proveedor como proveedor conocido del Estado; la persona con DNI que también es
  proveedor con RUC 10.
Se puede volver a correr cuando lleguen más resultados: rehace todo, en minutos.
Salida en datos/procesados/validado/ (fuera de git: trae nombres y DNI).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "radar"))
from comun import entidades, norm, ruc_por_nombre  # noqa: E402
from revisar_pilotos import doc, ruc_ok  # noqa: E402
from tce_resoluciones import normalizar  # noqa: E402

DATOS = RAIZ / "datos"
OUT = DATOS / "procesados" / "validado"
# CONOSCE: «O/S-258-2022-UNIDAD» (número-año); algunas unidades pegan el año al número: «O/S-2018000001-2018-...»
ORDEN = re.compile(r"^(O/[CS])-(\d+)-(\d{4})")


def tipo_orden(t, numero=""):
    t = f"{t or ''} {numero or ''}".upper()
    if re.search(r"O\s*/?\s*C|COMPRA", t):
        return "O/C"
    if re.search(r"O\s*/?\s*S|SERVICIO", t):
        return "O/S"
    return None


def serial_anio(numero, fecha):
    """«109-2019» -> (109, 2019); «0001392» con fecha 2023-11-27 -> (1392, 2023)."""
    s = str(numero or "")
    grupos = re.findall(r"\d+", s)
    if not grupos:
        return None, None
    anio = next((int(g) for g in grupos[1:] if re.fullmatch(r"20[12]\d", g)), None)
    if anio is None and re.match(r"\d{4}", str(fecha or "")):
        anio = int(str(fecha)[:4])
    serial = int(grupos[0]) if len(grupos[0]) <= 9 else None
    if serial is not None and len(grupos[0]) >= 10 and re.match(r"20[12]\d", grupos[0]):   # formato CONOSCE pegado
        anio, serial = int(grupos[0][:4]), int(grupos[0][4:])
    return serial, anio


VACIAS = {"DE", "DEL", "LA", "LAS", "LOS", "EL", "Y", "O", "SA", "S", "A", "EN", "PARA", "AL"}


def candidatos_entidad(nombres, ent, tope=30):
    """Nombre -> lista de RUC posibles. Primero la regla exacta del 15/10; si no calza, todas las entidades cuyo nombre
    contiene todas las palabras del nombre buscado (por ejemplo «GOBIERNO REGIONAL AREQUIPA» trae la sede y sus unidades
    ejecutoras). La orden misma (tipo, año y número) decide después cuál era."""
    exacto = ruc_por_nombre(nombres.fillna(""), ent)
    pal = [(r, set(n.split()) - VACIAS) for r, n in zip(ent["RUC"], ent["n1"])]
    memo, out = {}, []
    for nombre, ex in zip(nombres.fillna(""), exacto):
        if isinstance(ex, str):
            out.append([ex])
            continue
        if nombre not in memo:
            w = set(norm(nombre).split()) - VACIAS
            c = [r for r, s in pal if w and w <= s] if len(w) >= 2 else []
            memo[nombre] = c if 0 < len(c) <= tope else []
        out.append(memo[nombre])
    return out


def candidatos_por_lugar(nombres, region, provincia, ent, tope=5):
    """Para los nombres que no se resolvieron: las entidades de la misma provincia (o, si no hay, del mismo departamento)
    con más palabras en común (índice de Jaccard de 0,4 o más), hasta 5. La orden decide después cuál era."""
    ent = ent.assign(pal=[set(n.split()) - VACIAS for n in ent["n1"]], dep=ent["DEPARTAMENTO"].map(norm),
                     prov=ent["PROVINCIA"].map(norm))
    por_prov = {k: g for k, g in ent.groupby(["dep", "prov"])}
    por_dep = {k: g for k, g in ent.groupby("dep")}
    out = []
    for nombre, r, pv in zip(nombres.fillna(""), region.fillna(""), provincia.fillna("")):
        w = set(norm(nombre).split()) - VACIAS
        res = []
        for g in (por_prov.get((norm(r), norm(pv))), por_dep.get(norm(r))):
            if g is None or not w:
                continue
            j = [(len(w & s) / len(w | s), ruc) for s, ruc in zip(g["pal"], g["RUC"])]
            res = [ruc for sc, ruc in sorted(j, reverse=True)[:tope] if sc >= 0.4]
            if res:
                break
        out.append(res)
    return out


OCAM = re.compile(r"OCAM-\d{4}-\d+-\d+-\d+", re.IGNORECASE)
OC_ELEC = re.compile(r"electr[oó]nica\s*(?:N\s*[°º.o]*)?\s*0*(\d+)\s*-\s*(20\d\d)", re.IGNORECASE)


def catalogo():
    """Órdenes de catálogo de Perú Compras por su código: «OCAM-2024-1342-148-0» y, hasta 2019, «ORDEN_DE_COMPRA-n-año»."""
    ruta = DATOS / "procesados" / "perucompras" / "ordenes_catalogo.parquet"
    if not ruta.exists():
        return {}
    c = pd.read_parquet(ruta, columns=["ocid", "orden"])
    k = {}
    for oc, orden in zip(c["ocid"], c["orden"].fillna("")):
        m = re.match(r"ORDEN_DE_COMPRA-(\d+)-(\d{4})", orden)
        k[f"OC-{int(m.group(1))}-{m.group(2)}" if m else orden.upper()] = oc
    return k


def en_catalogo(texto, cat):
    t = str(texto or "")
    m = OCAM.search(t)
    if m and m.group(0).upper() in cat:
        return cat[m.group(0).upper()]
    m = OC_ELEC.search(t)
    if m:
        return cat.get(f"OC-{int(m.group(1))}-{m.group(2)}")
    return None


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def conosce():
    f = DATOS / "procesados" / "radar" / "ordenes.parquet"
    o = pd.read_parquet(f, columns=["id", "ruc_entidad", "orden", "ruc", "fecha", "monto"])
    # órdenes «otras» (HALLAZGOS 160, C3): en un empate gana la orden del universo original
    es_otra = "otra_categoria" in pq.read_schema(f).names
    o["otra_categoria"] = pd.read_parquet(f, columns=["otra_categoria"])["otra_categoria"].to_numpy() if es_otra else False
    p = o["orden"].str.extract(ORDEN)
    o["tipo"], o["anio_o"] = p[0], pd.to_numeric(p[2], errors="coerce")
    s = p[1].fillna("")
    pegado = (s.str.len() >= 10) & (s.str[:4] == p[2])
    o["serial"] = pd.to_numeric(s.where(~pegado, s.str[4:]), errors="coerce")
    o = o.dropna(subset=["serial", "anio_o"]).astype({"serial": "int64", "anio_o": "int64"})
    # otras unidades pegan los dos últimos dígitos del año: «O/C-2300418-2023» es la 418. Se agrega esa lectura como
    # segunda fila (la orden es la misma: mismo id)
    corto = (s.str.len() >= 6) & (s.str[:2] == p[2].str[2:])
    alt = o.loc[o.index.isin(corto[corto].index)].copy()
    alt["serial"] = pd.to_numeric(s[alt.index].str[2:], errors="coerce")
    return pd.concat([o, alt.dropna(subset=["serial"]).astype({"serial": "int64"})])


def ocds():
    """Adjudicaciones del OCDS con su entidad, y el número y año del procedimiento sacados de la nomenclatura."""
    a = pd.read_parquet(DATOS / "procesados" / "ocds" / "adjudicaciones.parquet", columns=["ocid", "ruc", "fecha", "monto"])
    pr = pd.read_parquet(DATOS / "procesados" / "ocds" / "procesos.parquet", columns=["ocid", "ruc_entidad", "nomenclatura"])
    a = a.merge(pr.drop_duplicates("ocid"), on="ocid")
    k = a["nomenclatura"].str.extract(r"-0*(\d+)-(20\d\d)-")
    a["n_proc"], a["anio_proc"] = pd.to_numeric(k[0], errors="coerce"), pd.to_numeric(k[1], errors="coerce")
    return a.dropna(subset=["n_proc", "anio_proc"]).astype({"n_proc": "int64", "anio_proc": "int64"})


def resoluciones(o, ent):
    sanc = pd.read_parquet(DATOS / "procesados" / "radar" / "sanciones.parquet")
    registro = set(sanc["ruc"])
    filas = []
    for linea in (DATOS / "procesados" / "claude" / "resultados_resoluciones.jsonl").open(encoding="utf-8"):
        r = json.loads(linea)
        if "salida" not in r:
            continue
        s = r["salida"]
        for k, e in enumerate(s["expedientes"], 1):
            c, sf, imp = e["contratacion"], e["sancion_final"], e["impedimento"] or {}
            fechas = sorted(i["fecha_hechos"] for i in e["infracciones"] if re.match(r"\d{4}-\d\d-\d\d", i["fecha_hechos"]))
            serial, anio = serial_anio(c["numero"], c["fecha"])
            filas.append({
                "resolucion": s["numero"] or r["custom_id"], "expediente": k, "tipo_resolucion": s["tipo"],
                "fecha_resolucion": s["fecha_resolucion"], "administrado": e["administrado"]["nombre"],
                "ruc": doc(e["administrado"]["ruc"], 11), "tipo_administrado": e["administrado"]["tipo"],
                "integrantes": [doc(i["ruc"], 11) for i in e["integrantes_consorcio"]],
                "integrantes_meses": [str(i.get("meses_sancion") or "") for i in e["integrantes_consorcio"]],
                "entidad": e["entidad"]["nombre"], "ruc_entidad_extraido": doc(e["entidad"]["ruc"], 11),
                "contratacion_tipo": c["tipo"], "contratacion_numero": c["numero"],
                "contratacion_procedimiento": c["procedimiento"], "contratacion_fecha": c["fecha"],
                "contratacion_monto": num(c["monto_soles"]), "orden_tipo": tipo_orden(c["tipo"], c["numero"]),
                "orden_serial": serial, "orden_anio": anio,
                "literales": ",".join(i["literal"] for i in e["infracciones"]),
                "decisiones": ",".join(i["decision"] for i in e["infracciones"]),
                "fecha_hechos_min": fechas[0] if fechas else None, "fecha_hechos_max": fechas[-1] if fechas else None,
                "fecha_denuncia": e["fecha_denuncia"] or None, "sancion": sf["tipo"], "meses": num(sf["meses"]),
                "impedimento_literal": imp.get("literal"), "autoridad": (imp.get("autoridad") or {}).get("nombre"),
                "autoridad_cargo": (imp.get("autoridad") or {}).get("cargo"),
                "autoridad_entidad": (imp.get("autoridad") or {}).get("entidad"),
                "relacion": imp.get("relacion_con_administrado"), "documentos": len(e["documentos_cuestionados"])})
    d = pd.DataFrame(filas)
    d["ruc_valido"] = d["ruc"].map(lambda x: ruc_ok(x) if x else False)
    d["en_registro"] = d["ruc"].isin(registro) | d["integrantes"].map(lambda xs: bool(set(xs) & registro))
    d["ruc_entidad"] = d["ruc_entidad_extraido"].where(d["ruc_entidad_extraido"].map(lambda x: bool(x) and ruc_ok(x)))
    d["ruc_entidad"] = d["ruc_entidad"].fillna(ruc_por_nombre(d["entidad"].fillna(""), ent))
    d["fecha_coherente"] = ~(pd.to_datetime(d["fecha_hechos_min"], errors="coerce") >
                             pd.to_datetime(d["fecha_resolucion"], errors="coerce"))
    # la orden en CONOSCE: por proveedor, tipo, año y número (el proveedor es el dato más firme)
    q = d.dropna(subset=["orden_tipo", "orden_serial", "orden_anio", "ruc"]).reset_index()
    q = q.astype({"orden_serial": "int64", "orden_anio": "int64"})
    m = q.merge(o, left_on=["ruc", "orden_tipo", "orden_anio", "orden_serial"],
                right_on=["ruc", "tipo", "anio_o", "serial"], how="inner")
    # si no, el mismo proveedor con una orden del mismo tipo en la misma fecha
    q2 = d[d["orden_tipo"].notna() & d["ruc"].notna() & d["contratacion_fecha"].astype(str).str.match(r"\d{4}-\d\d-\d\d")]
    q2 = q2.reset_index().assign(f=lambda x: pd.to_datetime(x["contratacion_fecha"], errors="coerce"))
    m2 = q2.merge(o.assign(f=o["fecha"].dt.normalize()), left_on=["ruc", "orden_tipo", "f"], right_on=["ruc", "tipo", "f"])
    m = pd.concat([m.assign(via="numero"), m2.assign(via="fecha")])
    m["dif"] = (m["contratacion_monto"] - m["monto"]).abs().fillna(1e12)
    # si la resolución nombra la entidad, primero una orden de esa entidad (HALLAZGOS 160, C9)
    m["ent_mal"] = m["ruc_entidad_x"].notna() & (m["ruc_entidad_x"].astype(str) != m["ruc_entidad_y"].astype(str))
    m = m.sort_values(["ent_mal", "via", "dif", "otra_categoria"], ascending=[True, False, True, True]).drop_duplicates("index")
    d["id_orden_conosce"] = d.index.map(dict(zip(m["index"], m["id"])))
    d["orden_hallada_por"] = d.index.map(dict(zip(m["index"], m["via"])))
    # contratos y procedimientos en el OCDS: por proveedor (o entidad), número y año del procedimiento
    a = ocds()
    txt = d["contratacion_procedimiento"].fillna("") + " " + d["contratacion_numero"].fillna("")
    k = txt.str.extract(r"0*(\d+)\s*-\s*(20[0-2]\d)")
    d["n_proc"], d["anio_proc"] = pd.to_numeric(k[0], errors="coerce"), pd.to_numeric(k[1], errors="coerce")
    es_proc = d["contratacion_tipo"].isin(["procedimiento_de_seleccion", "contrato"])
    q3 = d[es_proc].dropna(subset=["n_proc", "anio_proc"]).reset_index().astype({"n_proc": "int64", "anio_proc": "int64"})
    m3 = pd.concat([q3.merge(a, on=["ruc", "n_proc", "anio_proc"]),
                    q3.dropna(subset=["ruc_entidad"]).drop(columns="ruc").merge(a, on=["ruc_entidad", "n_proc", "anio_proc"])])
    m3 = m3.drop_duplicates("index")
    d["ocid_ocds"] = d.index.map(dict(zip(m3["index"], m3["ocid"])))
    cat = catalogo()
    d["ocid_catalogo"] = [en_catalogo(f"{a} {b}", cat) for a, b in zip(d["contratacion_numero"], d["contratacion_procedimiento"])]
    d["entidad_coincide"] = d.index.map(dict(zip(m["index"], m["ruc_entidad_y"] if "ruc_entidad_y" in m else
                                                 m["ruc_entidad"])))
    d["entidad_coincide"] = d["entidad_coincide"].notna() & (d["entidad_coincide"] == d["ruc_entidad"])
    # integrantes de consorcios con su sanción, para pasarles la S (HALLAZGOS 160, C10)
    integ = d[["resolucion", "expediente", "integrantes", "integrantes_meses"]].explode(["integrantes", "integrantes_meses"]).dropna(subset=["integrantes"])
    integ.rename(columns={"integrantes": "ruc", "integrantes_meses": "meses_sancion"}).to_parquet(OUT / "resoluciones_integrantes.parquet", index=False)
    d = d.drop(columns=["integrantes", "integrantes_meses"])
    d.to_parquet(OUT / "resoluciones_expedientes.parquet", index=False)
    es_orden = d["orden_tipo"].notna()
    rango = d["contratacion_tipo"].isin(["orden_de_compra", "orden_de_servicio"]) & d["orden_anio"].between(2018, 2025)
    return {"expedientes": len(d), "ruc_valido": round(d["ruc_valido"].mean(), 4),
            "en_registro": round(d["en_registro"].mean(), 4), "entidad_con_ruc": round(d["ruc_entidad"].notna().mean(), 4),
            "con_fecha_hechos": round(d["fecha_hechos_min"].notna().mean(), 4),
            "fecha_coherente": round(d["fecha_coherente"].mean(), 4),
            "expedientes_por_orden": int(es_orden.sum()),
            "orden_hallada_en_conosce": round(d.loc[es_orden, "id_orden_conosce"].notna().mean(), 4),
            "ordenes_2018_2025": int(rango.sum()),
            "orden_2018_2025_hallada": round(d.loc[rango, "id_orden_conosce"].notna().mean(), 4),
            "por_numero": int((d["orden_hallada_por"] == "numero").sum()),
            "por_fecha": int((d["orden_hallada_por"] == "fecha").sum()),
            "contratos_o_procedimientos": int(es_proc.sum()),
            "hallados_en_ocds": round(d.loc[es_proc, "ocid_ocds"].notna().mean(), 4),
            "de_2018_en_adelante_hallados": round(d.loc[es_proc & (d["anio_proc"] >= 2018), "ocid_ocds"].notna().mean(), 4),
            "hallados_en_catalogo": int(d["ocid_catalogo"].notna().sum()),
            "orden_2018_2025_hallada_conosce_o_catalogo": round(
                (d.loc[rango, "id_orden_conosce"].notna() | d.loc[rango, "ocid_catalogo"].notna()).mean(), 4)}


def informes(o, ent):
    fi = pd.read_csv(DATOS / "crudos" / "acc8_informes_cgr" / "fichas" / "fichas_informes.csv", dtype=str,
                     usecols=["CresCodigoFormato", "entidad", "Periodo", "region", "provincia"])
    fi = fi.drop_duplicates("CresCodigoFormato").reset_index(drop=True)
    fi["candidatos"] = candidatos_entidad(fi["entidad"], ent)
    sin = fi["candidatos"].map(len) == 0      # los que no salieron por nombre: por lugar y palabras en común
    fi.loc[sin, "candidatos"] = pd.Series(candidatos_por_lugar(fi.loc[sin, "entidad"], fi.loc[sin, "region"],
                                                               fi.loc[sin, "provincia"], ent), index=fi.index[sin])
    fi["ruc_entidad"] = fi["candidatos"].map(lambda c: c[0] if len(c) == 1 else None)
    ficha = fi.set_index("CresCodigoFormato")
    proveedores_estado = set(o["ruc"])
    dni_proveedor = set(o.loc[o["ruc"].str.startswith("10"), "ruc"].str[2:10])
    H, O, P, Q = [], [], [], []
    for linea in (DATOS / "procesados" / "deepseek" / "informes_fondo.jsonl").open(encoding="utf-8"):
        r = json.loads(linea)
        if "salida" not in r:
            continue
        c, s = r["clave"], r["salida"]
        re_ent = ficha["ruc_entidad"].get(c)
        cands = ficha["candidatos"].get(c, [])
        for h in s.get("hechos") or []:
            v = h.get("vinculo") or {}
            H.append({"informe": c, "hecho": h.get("n"), "sumilla": h.get("sumilla"), "tipos": ",".join(h.get("tipos") or []),
                      "desde": (h.get("fechas_hechos") or {}).get("desde"), "hasta": (h.get("fechas_hechos") or {}).get("hasta"),
                      "perjuicio": num(h.get("perjuicio_soles")), "vinculo": v.get("tipo"),
                      "descargos_desvirtuan": (h.get("descargos") or {}).get("desvirtuan")})
            for x in h.get("ordenes") or []:
                serial, anio = serial_anio(x.get("numero"), x.get("fecha"))
                O.append({"informe": c, "hecho": h.get("n"), "ruc_entidad": re_ent, "candidatos": cands, "tipo": tipo_orden(x.get("tipo"), x.get("numero")),
                          "numero": x.get("numero"), "serial": serial, "anio": anio, "fecha": x.get("fecha"),
                          "monto": num(x.get("monto_soles")), "proveedor": x.get("proveedor"),
                          "ruc_proveedor": doc(x.get("ruc_proveedor"), 11)})
            for p in h.get("proveedores") or []:
                rr = doc(p.get("ruc"), 11)
                P.append({"informe": c, "hecho": h.get("n"), "nombre": p.get("nombre"), "ruc": rr, "rol": p.get("rol"),
                          "ruc_valido": bool(rr) and ruc_ok(rr), "proveedor_del_estado": rr in proveedores_estado})
            for p in h.get("personas") or []:
                dn = doc(p.get("dni"), 8)
                Q.append({"informe": c, "hecho": h.get("n"), "nombre": p.get("nombre"), "dni": dn, "cargo": p.get("cargo"),
                          "rol": p.get("rol"), "responsabilidad": p.get("responsabilidad"),
                          "dni_valido": bool(dn) and len(dn) == 8, "tambien_es_proveedor": dn in dni_proveedor})
    H, O, P, Q = map(pd.DataFrame, (H, O, P, Q))
    base = O.dropna(subset=["tipo", "serial", "anio"]).reset_index().astype({"serial": "int64", "anio": "int64"})
    # por entidad (cada candidata), y si no, por el RUC del proveedor que el informe da
    q = base.explode("candidatos").dropna(subset=["candidatos"]).drop(columns="ruc_entidad")
    m1 = q.merge(o, left_on=["candidatos", "tipo", "anio", "serial"], right_on=["ruc_entidad", "tipo", "anio_o", "serial"])
    q2 = base.dropna(subset=["ruc_proveedor"]).drop(columns="ruc_entidad")
    m2 = q2.merge(o, left_on=["ruc_proveedor", "tipo", "anio", "serial"], right_on=["ruc", "tipo", "anio_o", "serial"])
    # si no, una orden de la entidad del mismo tipo, en la misma fecha y por el mismo monto (±1 sol)
    q4 = O.reset_index().assign(f=lambda x: pd.to_datetime(x["fecha"], errors="coerce")).dropna(subset=["f", "monto", "tipo"])
    q4 = q4.explode("candidatos").dropna(subset=["candidatos"]).drop(columns="ruc_entidad")
    m4 = q4.merge(o.assign(f=o["fecha"].dt.normalize()), left_on=["candidatos", "tipo", "f"],
                  right_on=["ruc_entidad", "tipo", "f"])
    m4 = m4[(m4["monto_x"] - m4["monto_y"]).abs() <= 1]
    # por último, la única orden de la entidad de ese tipo, año y monto exacto (al céntimo)
    q5 = O.reset_index().dropna(subset=["monto", "tipo", "anio"]).explode("candidatos").dropna(subset=["candidatos"])
    q5 = q5.drop(columns="ruc_entidad").assign(m=lambda x: x["monto"].round(2))
    o5 = o.drop_duplicates("id").assign(m=lambda x: x["monto"].round(2))
    m5 = q5.merge(o5, left_on=["candidatos", "tipo", "anio", "m"], right_on=["ruc_entidad", "tipo", "anio_o", "m"])
    # «única» se mide primero en el universo original: una «otra» con el mismo monto no le quita el cruce a una orden vieja
    m5u = m5[~m5["otra_categoria"]]
    m5u = m5u[m5u.groupby("index")["id"].transform("nunique") == 1]
    m5o = m5[m5["otra_categoria"] & ~m5["index"].isin(m5.loc[~m5["otra_categoria"], "index"])]
    m5o = m5o[m5o.groupby("index")["id"].transform("nunique") == 1]
    m5 = pd.concat([m5u, m5o])
    m = pd.concat([m1.assign(via="numero"), m2.assign(via="numero"), m4.assign(via="fecha_y_monto"),
                   m5.assign(via="monto_unico")])
    m["dif"] = (m["monto_x"] - m["monto_y"]).abs().fillna(1e12)
    m["orden_via"] = m["via"].map({"numero": 0, "fecha_y_monto": 1, "monto_unico": 2})
    m = m.sort_values(["orden_via", "dif", "otra_categoria"]).drop_duplicates("index")
    O["ruc_entidad"] = O["ruc_entidad"].fillna(O.index.to_series().map(dict(zip(m["index"], m["ruc_entidad"]))))
    O["id_orden_conosce"] = O.index.map(dict(zip(m["index"], m["id"])))
    O["monto_conosce"] = O.index.map(dict(zip(m["index"], m["monto_y"])))
    O["hallada_por"] = O.index.map(dict(zip(m["index"], m["via"])))
    O["monto_coincide"] = (O["monto"] - O["monto_conosce"]).abs() <= 1
    O["ruc_proveedor_conosce"] = O.index.map(dict(zip(m["index"], m["ruc"])))
    cat = catalogo()
    O["ocid_catalogo"] = [en_catalogo(x, cat) for x in O["numero"]]
    for nombre, t in (("informes_hechos", H), ("informes_ordenes", O), ("informes_proveedores", P),
                      ("informes_personas", Q)):
        t.to_parquet(OUT / f"{nombre}.parquet", index=False)
    O = O.drop(columns="candidatos")
    con_clave = O[["tipo", "serial", "anio"]].notna().all(axis=1)
    hallada = O["id_orden_conosce"].notna()
    return {"informes": int(H["informe"].nunique()), "hechos": len(H), "ordenes": len(O),
            "entidad_del_informe_unica": round(ficha.loc[ficha.index.isin(H["informe"]), "ruc_entidad"].notna().mean(), 4),
            "entidad_del_informe_con_candidatas": round(ficha.loc[ficha.index.isin(H["informe"]), "candidatos"].map(bool).mean(), 4),
            "ordenes_con_clave_completa": round(con_clave.mean(), 4),
            "ordenes_halladas_en_conosce": round(hallada.mean(), 4),
            "de_las_que_tienen_clave": round(hallada[con_clave].mean(), 4),
            "de_2018_2025": round(hallada[O["anio"].between(2018, 2025)].mean(), 4),
            "halladas_en_catalogo": int(O["ocid_catalogo"].notna().sum()),
            "de_2018_2025_conosce_o_catalogo": round((hallada | O["ocid_catalogo"].notna())[O["anio"].between(2018, 2025)].mean(), 4),
            "por_fecha_y_monto": int((O["hallada_por"] == "fecha_y_monto").sum()),
            "por_monto_unico": int((O["hallada_por"] == "monto_unico").sum()),
            "monto_coincide_entre_halladas": round(O.loc[hallada, "monto_coincide"].mean(), 4),
            "proveedores": len(P), "proveedores_con_ruc": int(P["ruc"].notna().sum()),
            "ruc_valido": round(P.loc[P["ruc"].notna(), "ruc_valido"].mean(), 4),
            "proveedor_del_estado": round(P.loc[P["ruc"].notna(), "proveedor_del_estado"].mean(), 4),
            "personas": len(Q), "personas_con_dni": int(Q["dni"].notna().sum()),
            "personas_que_tambien_son_proveedores": int(Q["tambien_es_proveedor"].sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    o = conosce()
    ent = entidades()
    res = {"resoluciones": resoluciones(o, ent), "informes": informes(o, ent)}
    (OUT / "resumen_validacion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
