"""Data v4, capa 1: base canónica (HALLAZGOS 118). Tablas con llave y fechas, sin etiquetas ni filtros de universo.

Sale en datos/procesados/v4/base/:
- ordenes.parquet: una fila por orden (la base de 01_base.py: soles, sin anuladas, sin repetidas, monto posible).
- ordenes_texto.parquet: descripción de cada orden (del crudo de CONOSCE), aparte para no cargarla siempre.
- entidades.parquet y proveedores.parquet: canónicos, con ubicación del padrón (foto de 2026, marcada).
- informes.parquet: los 14 970 informes (lista ampliada y complementaria), con fecha de emisión Y de publicación,
  entidad resuelta, modalidad y alcance; es también la tabla de cobertura de las auditorías (con o sin hallazgo).
- resoluciones.parquet: resolución × administrado, con fechas, decisión, infracción y la marca de C núcleo (orden hallada).
- hechos_orden.parquet: hechos ligados a una orden concreta: A (orden citada en un informe; la recuperación exige que la
  orden sea de una entidad candidata del informe) y C núcleo (resolución con la orden hallada).
- hechos_relacion.parquet: hechos ligados a una relación sin orden concreta: B (informe que nombra al proveedor como
  beneficiado, con su periodo, sin expandir) y C sin orden (antecedente).
- sanciones.parquet: registro de inhabilitaciones con su historia.
Al final corre las pruebas de la capa. Solo imprime agregados.

    python -I -X utf8 capa1_base.py
"""
import glob
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from comun import CRUDOS, PROC, SALIDA, UIT, entidades  # noqa: E402

sys.path.insert(0, str(AQUI.parents[1] / "datos" / "_scripts"))       # validar_extracciones importa scripts de su carpeta
spec = importlib.util.spec_from_file_location("valid", AQUI.parents[1] / "datos" / "_scripts" / "validar_extracciones.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

OUT = PROC / "v4" / "base"
CGR = CRUDOS / "acc8_informes_cgr"
T0 = time.time()
# hechos de deudas o de reconocimiento de créditos: el proveedor es acreedor, no beneficiado (HALLAZGOS 160, C8; se marca)
DEUDA = (r"deuda|reconocimiento de (?:cr[eé]dito|deuda)|cr[eé]ditos? devengad|obligaciones? (?:pendientes|de ejercicios anteriores)|devengado")
LISTA_BASICOS = (r"SEDAPAL|SEDAPAR|SEDALIB|SEDACUSCO|SEDALORETO|SEDAJULIACA|SEDAM HUANCAYO|SEDACHIMBOTE|SEDAHUANUCO|SEDACAJ|"
                 r"SEDA AYACUCHO|SEDA HUANUCO|EMAPA|EMAPICA|EPS\.? (GRAU|TACNA|MOQUEGUA|ILO|CHAVIN|SEMAPACH|AGUAS DEL ALTIPLANO|MARA.ON|"
                 r"SELVA CENTRAL|SIERRA CENTRAL|BARRANCA|MANTARO|RIOJA|NOR PUNO|SEDA)|AGUA POTABLE Y ALCANTARILLADO|"
                 r"PRESTADORA DE SERVICIOS DE SANEAMIENTO|ELECTROCENTRO|ELECTRONOROESTE|ELECTRONORTE|ELECTRO SUR ESTE|ELECTROSUR|"
                 r"ELECTRO ORIENTE|ELECTRO UCAYALI|ELECTRO PUNO|ELECTRO TOCACHE|ELECTRO DUNAS|HIDRANDINA|ENEL DISTRIBUCION|LUZ DEL SUR|"
                 r"SOCIEDAD ELECTRICA DEL SUR|COELVISAC|SERVICIO PUBLICO DE ELECTRICIDAD|TELEFONICA DEL PERU|AMERICA MOVIL|ENTEL PERU|"
                 r"VIETTEL|CALIDDA|GAS NATURAL DE LIMA|CONTUGAS|SERVICIOS POSTALES DEL PERU|EDITORA PERU|"
                 r"EMPRESA PERUANA DE SERVICIOS EDITORIALES|BANCO DE LA NACION")


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def tope(anio, fecha=None):
    """Tope legal de la compra menor: 8 UIT; 9 UIT solo del 19/07/2022 al 31/12/2022 (DU 016-2022; HALLAZGOS 148).
    Antes se usaban 9 UIT todo 2022; en la base no hay órdenes de enero a julio de 2022 entre 8 y 9 UIT, pero «cerca
    del tope» de esos meses quedaba mal medido (32 360 órdenes en vez de 51 986)."""
    a = np.asarray(anio)
    nueve = a == 2022
    if fecha is not None:
        f = pd.to_datetime(np.asarray(fecha))
        nueve = nueve & (f >= pd.Timestamp("2022-07-19")) & (f <= pd.Timestamp("2022-12-31"))
    return pd.Series(a).map(UIT).to_numpy() * np.where(nueve, 9, 8)


def norm_res(s):
    return pd.Series(s).astype(str).str.strip().str.upper().str.replace(r"^0+", "", regex=True).to_numpy()


# ------------------------------------------------------------------------------------------------ órdenes
def ordenes():
    paso("órdenes y su texto")
    o = pd.read_parquet(SALIDA / "ordenes.parquet")
    o["ruc_entidad"], o["ruc"], o["orden"] = o["ruc_entidad"].astype(str), o["ruc"].astype(str), o["orden"].astype(str)
    partes = []
    for f in sorted(glob.glob(str(PROC / "acc1_conosce" / "ordenes_20*.parquet"))):
        cols = ["ruc_entidad", "orden", "descripcion_orden", "fecha_registro"]
        d = pd.read_parquet(f, columns=cols)
        partes.append(d.astype({"ruc_entidad": str, "orden": str}))
    t = pd.concat(partes, ignore_index=True)
    t = t.sort_values("fecha_registro").drop_duplicates(["ruc_entidad", "orden"], keep="last")      # igual que 01_base
    t = o[["id", "ruc_entidad", "orden"]].merge(t[["ruc_entidad", "orden", "descripcion_orden"]], on=["ruc_entidad", "orden"], how="left")
    texto = t["descripcion_orden"].fillna("").astype(str).str.strip()
    ot = pd.DataFrame({"id": t["id"], "descripcion": texto, "cruce_texto": t["descripcion_orden"].notna(),
                       "texto_vacio": t["descripcion_orden"].notna() & texto.eq(""),
                       "anio": o["anio"].to_numpy()})         # los id ya no son contiguos por año (160, C3): se filtra por anio
    o["tope"] = tope(o["anio"], o["fecha"])
    o["parte_del_tope"] = (o["monto"] / o["tope"]).astype("float32")
    o.to_parquet(OUT / "ordenes.parquet", index=False)
    ot.to_parquet(OUT / "ordenes_texto.parquet", index=False)
    print(f"   órdenes {len(o):,}; con texto cruzado {ot['cruce_texto'].mean():.1%}; vacíos {ot['texto_vacio'].mean():.2%}", flush=True)
    return o


# ------------------------------------------------------------------------------------------------ entidades y proveedores
TIPOS = (("MUNICIPALIDAD DISTRITAL", "municipalidad distrital"), ("MUNICIPALIDAD PROVINCIAL", "municipalidad provincial"),
         ("GOBIERNO REGIONAL", "gobierno regional"), ("MINISTERIO", "ministerio"), ("UNIVERSIDAD", "universidad"),
         ("HOSPITAL", "salud"), ("SALUD", "salud"), ("UGEL", "educación"), ("UNIDAD DE GESTION EDUCATIVA", "educación"),
         ("PROGRAMA", "programa o proyecto"), ("PROYECTO", "programa o proyecto"))


def entidades_y_proveedores(o):
    paso("entidades y proveedores canónicos")
    ent = entidades().drop_duplicates("RUC")
    u = pd.read_parquet(SALIDA / "padron_ubicacion.parquet").set_index("ruc")
    e = pd.DataFrame({"ruc_entidad": ent["RUC"].astype(str), "nombre": ent["NOMBRE_DE_ENTIDAD"], "departamento": ent["DEPARTAMENTO"],
                      "provincia": ent["PROVINCIA"], "distrito": ent["DISTRITO"], "estado": ent["ESTADO"]})
    e["tipo"] = e["nombre"].map(lambda n: next((v for k, v in TIPOS if k in str(n).upper()), "otra"))
    e["ubigeo_padron_2026"] = e["ruc_entidad"].map(u["ubigeo"])
    en_ordenes = set(o["ruc_entidad"])
    e["con_ordenes"] = e["ruc_entidad"].isin(en_ordenes)
    faltan = sorted(en_ordenes - set(e["ruc_entidad"]))          # entidades con órdenes que no están en la lista oficial
    if faltan:
        e = pd.concat([e, pd.DataFrame({"ruc_entidad": faltan, "tipo": "otra", "con_ordenes": True,
                                        "ubigeo_padron_2026": [u["ubigeo"].get(r) for r in faltan]})], ignore_index=True)
    e.to_parquet(OUT / "entidades.parquet", index=False)
    g = o.groupby("ruc")
    ultimo = o.sort_values("fecha").drop_duplicates("ruc", keep="last").set_index("ruc")["nombre"]   # nombre de la orden más reciente
    p = pd.DataFrame({"nombre": ultimo, "primera_orden": g["fecha"].min(), "ultima_orden": g["fecha"].max()}).reset_index()
    p = p.rename(columns={"index": "ruc"})
    p["tipo_ruc"] = np.select([p["ruc"].str.startswith("10"), p["ruc"].str.startswith("20")], ["persona_natural", "empresa"], "otro")
    p["es_entidad_publica"] = p["ruc"].isin(set(e["ruc_entidad"]))
    p["servicio_basico_lista"] = (p["tipo_ruc"] == "empresa") & p["nombre"].astype(str).str.upper().str.contains(LISTA_BASICOS, regex=True)
    p["ubigeo_padron_2026"] = p["ruc"].map(u["ubigeo"])
    p["direccion_padron_2026"] = p["ruc"].map(u["direccion"])
    p.to_parquet(OUT / "proveedores.parquet", index=False)
    print(f"   entidades {len(e):,} ({len(faltan)} con órdenes fuera de la lista oficial); proveedores {len(p):,}", flush=True)
    return e, p


# ------------------------------------------------------------------------------------------------ informes
def informes():
    paso("informes: emisión, publicación, entidad y alcance")
    l = pd.read_csv(CGR / "listas" / "lista_unica_informes_ampliada.csv", dtype=str)
    c = pd.read_csv(CGR / "listas" / "lista_complementaria_E1.csv", dtype=str)
    c = c[~c["CresCodigoFormato"].isin(set(l["CresCodigoFormato"]))]
    l = pd.concat([l, c], ignore_index=True).drop_duplicates("CresCodigoFormato")
    f = lambda s: pd.to_datetime(s, format="%Y/%m/%d", errors="coerce")  # noqa: E731
    i = pd.DataFrame({"informe": l["CresCodigoFormato"], "numero": l.get("NumeroInforme"), "periodo": l.get("Periodo"),
                      "fecha_emision": f(l["FechaEmision"]), "fecha_publicacion": f(l.get("FechaPublicacion")),
                      "entidad_texto": l["Entidad"], "departamento": l.get("Departamento"), "provincia": l.get("Provincia"),
                      "modalidad": l.get("ModalidadServicio"), "servicio": l.get("ServicioControl"),
                      "con_responsabilidad": l.get("EsConResponsabilidad").eq("S") if "EsConResponsabilidad" in l else np.nan,
                      "nivel_gobierno": l.get("NivelGobierno"), "sector": l.get("Sector")})
    em2 = pd.to_datetime(l["FechaEmision"], format="%d/%m/%Y", errors="coerce")   # la complementaria usa otro formato
    i["fecha_emision"] = i["fecha_emision"].fillna(pd.Series(em2.to_numpy(), index=i.index))
    i["publicacion_imputada"] = i["fecha_publicacion"].isna()
    ent = entidades()
    i["candidatos"] = V.candidatos_entidad(i["entidad_texto"], ent)
    sin = i["candidatos"].map(len) == 0
    i.loc[sin, "candidatos"] = pd.Series(V.candidatos_por_lugar(i.loc[sin, "entidad_texto"], i.loc[sin, "departamento"],
                                                                i.loc[sin, "provincia"], ent), index=i.index[sin])
    i["candidatos"] = i["candidatos"].map(lambda x: [str(r) for r in x])
    i["ruc_entidad"] = i["candidatos"].map(lambda x: x[0] if len(x) == 1 else None)
    h = pd.read_parquet(PROC / "validado/informes_hechos.parquet", columns=["informe", "hecho", "desde", "hasta"])
    h["d"], h["h"] = pd.to_datetime(h["desde"], errors="coerce"), pd.to_datetime(h["hasta"], errors="coerce")
    al = h.groupby("informe").agg(n_hechos=("hecho", "nunique"), alcance_desde=("d", "min"), alcance_hasta=("h", "max"))
    io = pd.read_parquet(PROC / "validado/informes_ordenes.parquet", columns=["informe"]).groupby("informe").size().rename("n_ordenes_citadas")
    ip = pd.read_parquet(PROC / "validado/informes_proveedores.parquet", columns=["informe", "rol"])
    ip = ip[ip["rol"].eq("beneficiado")].groupby("informe").size().rename("n_beneficiados")
    i = i.merge(al, left_on="informe", right_index=True, how="left").merge(io, left_on="informe", right_index=True, how="left") \
         .merge(ip, left_on="informe", right_index=True, how="left")
    i["leido_con_hechos"] = i["n_hechos"].notna()
    i[["n_hechos", "n_ordenes_citadas", "n_beneficiados"]] = i[["n_hechos", "n_ordenes_citadas", "n_beneficiados"]].fillna(0)
    i["demora_publicacion_dias"] = (i["fecha_publicacion"] - i["fecha_emision"]).dt.days
    i.to_parquet(OUT / "informes.parquet", index=False)
    print(f"   informes {len(i):,}; entidad única {i['ruc_entidad'].notna().mean():.1%}; con hechos leídos "
          f"{i['leido_con_hechos'].sum():,}; publicación imputada {int(i['publicacion_imputada'].sum())}", flush=True)
    return i


# ------------------------------------------------------------------------------------------------ resoluciones
def resoluciones(o):
    paso("resoluciones: fechas, decisión e infracción")
    r = pd.read_parquet(PROC / "validado/resoluciones_expedientes.parquet")
    idx = pd.read_csv(CRUDOS / "tce_resoluciones" / "indice.csv", dtype=str)
    idx["k"] = norm_res(idx["numero"])
    pub = idx.drop_duplicates("k").set_index("k")["fecha_publicacion"]
    r["k"] = norm_res(r["resolucion"])
    x = pd.DataFrame({"resolucion": r["resolucion"], "expediente": r["expediente"], "ruc": r["ruc"].astype(str), "tipo_administrado": r["tipo_administrado"],
                      "fecha_resolucion": pd.to_datetime(r["fecha_resolucion"], errors="coerce"),
                      "fecha_publicacion": pd.to_datetime(r["k"].map(pub), errors="coerce"),
                      "tipo_resolucion": r["tipo_resolucion"], "decisiones": r["decisiones"].astype(str),
                      "sanciona": r["decisiones"].astype(str).str.contains("sanciona"),
                      "ruc_valido": r["ruc"].astype(str).str.fullmatch(r"\d{11}", na=False),
                      "literales": r["literales"].astype(str), "impedimento_literal": r["impedimento_literal"],
                      "fecha_hechos_min": pd.to_datetime(r["fecha_hechos_min"], errors="coerce"),
                      "fecha_hechos_max": pd.to_datetime(r["fecha_hechos_max"], errors="coerce"),
                      "entidad_texto": r["entidad"], "ruc_entidad_extraido": r["ruc_entidad"],
                      "procedimiento": r["contratacion_procedimiento"], "monto_contratacion": pd.to_numeric(r["contratacion_monto"], errors="coerce"),
                      "id_orden": pd.to_numeric(r["id_orden_conosce"], errors="coerce").astype("Int64"),
                      "orden_hallada_por": r["orden_hallada_por"], "entidad_coincide": r["entidad_coincide"]})
    x["fecha_publicacion"] = x["fecha_publicacion"].fillna(x["fecha_resolucion"])            # coinciden en el 99 % (2/11)
    x["es_impedimento"] = x["impedimento_literal"].notna() | x["literales"].str.contains(r"\bc\b", regex=True)
    # si la resolución nombra la entidad, la orden debe ser de esa entidad; si no, queda como S con su entidad (160, C9)
    entidad_ok = x["ruc_entidad_extraido"].isna() | x["entidad_coincide"].astype(bool)
    x["c_nucleo"] = x["sanciona"] & x["ruc_valido"] & x["id_orden"].notna() & entidad_ok      # HALLAZGOS 115 y 116
    x.to_parquet(OUT / "resoluciones.parquet", index=False)
    print(f"   expedientes {len(x):,}; sancionan {int(x['sanciona'].sum()):,}; C núcleo {int(x['c_nucleo'].sum()):,} "
          f"(impedimento {int((x['c_nucleo'] & x['es_impedimento']).sum()):,})", flush=True)
    return x


# ------------------------------------------------------------------------------------------------ hechos
def unico_por(df, grupo, col):
    n = df.groupby(grupo)[col].transform("nunique")
    return df[n == 1].drop_duplicates(grupo)


def hechos(o, inf, res):
    paso("hechos a nivel de orden (A y C núcleo) y de relación (B y C sin orden)")
    ids = o[["id", "ruc_entidad", "ruc", "anio", "monto", "fecha"]]
    cand_inf = inf.set_index("informe")["candidatos"]
    io = pd.read_parquet(PROC / "validado/informes_ordenes.parquet").reset_index(names="fila")
    io["cands"] = [list(map(str, c)) if isinstance(c, (list, np.ndarray)) and len(c) else ([str(r)] if isinstance(r, str) else [])
                   for c, r in zip(io["candidatos"], io["ruc_entidad"])]
    io["cands"] = [c if c else list(cand_inf.get(i, [])) for c, i in zip(io["cands"], io["informe"])]
    # A1: ya hallada en la validación
    a1 = io.dropna(subset=["id_orden_conosce"]).assign(id=lambda d: d["id_orden_conosce"].astype("int64"), metodo="hallada")
    nh = io[io["id_orden_conosce"].isna() & io["anio"].between(2018, 2025)]
    # A3: número de orden dentro de una entidad candidata, mismo año, monto compatible, una sola coincidencia
    oc = V.conosce()[["id", "ruc_entidad", "anio_o", "serial", "tipo"]].drop_duplicates().astype({"ruc_entidad": str}).rename(columns={"tipo": "tipo_o"})
    q = nh.dropna(subset=["serial"]).explode("cands").dropna(subset=["cands"]).astype({"serial": "int64"})
    v3 = q.merge(oc, left_on=["cands", "anio", "serial"], right_on=["ruc_entidad", "anio_o", "serial"], suffixes=("", "_o"))
    v3 = v3.merge(ids[["id", "monto"]].rename(columns={"monto": "monto_o"}), on="id")
    v3 = v3[v3["monto"].isna() | ((v3["monto_o"] - v3["monto"]).abs() <= 0.02 * v3["monto"].abs() + 1)]
    v3 = v3[v3["tipo"].isna() | (v3["tipo"] == v3["tipo_o"])]      # si contradice el tipo citado, no es la orden citada (160, C7)
    v3 = unico_por(v3, "fila", "id").assign(metodo="numero_en_candidata")
    # A2: proveedor, año y monto (±1 %), SOLO dentro de las entidades candidatas (corrige el defecto 2 de HALLAZGOS 113)
    q2 = nh[~nh["fila"].isin(v3["fila"])].dropna(subset=["ruc_proveedor", "monto"]).explode("cands").dropna(subset=["cands"])
    v2 = q2.merge(ids.rename(columns={"monto": "monto_o", "ruc_entidad": "ent_o"}), left_on=["ruc_proveedor", "anio", "cands"],
                  right_on=["ruc", "anio", "ent_o"])
    v2 = v2[(v2["monto_o"] - v2["monto"]).abs() <= 0.01 * v2["monto"].abs() + 1]
    v2 = unico_por(v2, "fila", "id").assign(metodo="proveedor_anio_monto_en_candidata")
    A = pd.concat([a1[["fila", "informe", "hecho", "id", "metodo", "cands"]], v3[["fila", "informe", "hecho", "id", "metodo", "cands"]],
                   v2[["fila", "informe", "hecho", "id", "metodo", "cands"]]]).drop_duplicates("fila")
    A = A.merge(ids, on="id")
    # cruces dudosos: se marcan y se miden con y sin ellos, no se borran (HALLAZGOS 160, C7)
    inf_o = io[["fila", "tipo", "monto", "ruc_proveedor"]].rename(columns={"tipo": "tipo_inf", "monto": "monto_inf", "ruc_proveedor": "ruc_inf"})
    A = A.merge(inf_o, on="fila", how="left")
    A["dudosa"] = ((A["ruc_inf"].notna() & (A["ruc_inf"] != A["ruc"]))
                   | (A["monto_inf"].notna() & ((A["monto"] - A["monto_inf"]).abs() > 0.10 * A["monto_inf"].abs() + 1))
                   | ((A["metodo"] == "numero_en_candidata") & (A["tipo_inf"].isna() | A["monto_inf"].isna())))
    A = A.drop(columns=["tipo_inf", "monto_inf", "ruc_inf"])
    A["entidad_coincide"] = [e in c for e, c in zip(A["ruc_entidad"], A["cands"])]
    A = A.assign(fuente="A", documento=A["informe"]).drop(columns=["cands", "fila", "informe"])
    C = res[res["c_nucleo"]][["resolucion", "ruc", "id_orden", "es_impedimento", "literales", "entidad_coincide"]].rename(
        columns={"resolucion": "documento", "ruc": "ruc_administrado", "id_orden": "id"}).astype({"id": "int64"})
    C = C.merge(ids, on="id").assign(fuente="C", metodo="orden_hallada")
    C["administrado_es_proveedor"] = C["ruc_administrado"] == C["ruc"]
    ho = pd.concat([A, C], ignore_index=True)
    ho = ho.drop_duplicates(["fuente", "documento", "hecho", "id", "ruc_administrado"])   # la misma orden citada dos veces en un hecho
    ho.to_parquet(OUT / "hechos_orden.parquet", index=False)
    print(f"   hechos de orden: A {int((ho['fuente'] == 'A').sum()):,} ({ho[ho['fuente'] == 'A']['metodo'].value_counts().to_dict()}); "
          f"A con entidad que no coincide {int(((ho['fuente'] == 'A') & ~ho['entidad_coincide'].astype(bool)).sum()):,}; "
          f"C núcleo {int((ho['fuente'] == 'C').sum()):,}", flush=True)
    # B: el informe nombra al proveedor como beneficiado; se guarda el periodo, sin expandir por años
    ip = pd.read_parquet(PROC / "validado/informes_proveedores.parquet", columns=["informe", "hecho", "ruc", "ruc_valido", "rol"])
    ip = ip[ip["ruc_valido"].fillna(False).astype(bool) & ip["rol"].eq("beneficiado")]
    ih = pd.read_parquet(PROC / "validado/informes_hechos.parquet", columns=["informe", "hecho", "desde", "hasta", "sumilla"])
    b = ip.merge(ih, on=["informe", "hecho"], how="left")
    b["acreedor"] = b["sumilla"].fillna("").str.contains(DEUDA, case=False, regex=True)
    b["desde"], b["hasta"] = pd.to_datetime(b["desde"], errors="coerce"), pd.to_datetime(b["hasta"], errors="coerce")
    b["ruc_entidad"] = b["informe"].map(inf.set_index("informe")["ruc_entidad"])
    b["metodo_entidad"] = np.where(b["ruc_entidad"].notna(), "entidad_unica", "")
    anios_par = ids.groupby(["ruc_entidad", "ruc"])["anio"].agg(set)
    amb = b["ruc_entidad"].isna()

    def resolver(ruc, cands, d, h):
        if not isinstance(cands, (list, np.ndarray)) or len(cands) < 2 or pd.isna(d):
            return None
        hh = h.year if pd.notna(h) else d.year
        con = [e for e in cands if anios_par.get((e, ruc)) and any(d.year <= a <= hh + 1 for a in anios_par.get((e, ruc)))]
        return con[0] if len(con) == 1 else None
    b.loc[amb, "ruc_entidad"] = [resolver(r, cand_inf.get(i), d, h) for r, i, d, h in
                                 zip(b.loc[amb, "ruc"], b.loc[amb, "informe"], b.loc[amb, "desde"], b.loc[amb, "hasta"])]
    b.loc[amb & b["ruc_entidad"].notna(), "metodo_entidad"] = "resuelta_por_casos"
    B = b.dropna(subset=["ruc_entidad"]).rename(columns={"informe": "documento"}).assign(fuente="B")
    CS_COLS = ["resolucion", "ruc", "ruc_entidad_extraido", "fecha_hechos_min", "fecha_hechos_max", "es_impedimento", "procedimiento", "monto_contratacion"]
    Cs = res[res["sanciona"] & res["ruc_valido"] & ~res["c_nucleo"]]
    # la reconsideración de una sanción que ya está en S no es otra sanción (HALLAZGOS 160, C11)
    es_rec = Cs["tipo_resolucion"].astype(str).str.contains("reconsider", case=False)
    bases = set(zip(Cs.loc[~es_rec, "ruc"], Cs.loc[~es_rec, "fecha_hechos_min"]))
    repetida = es_rec & pd.Series([k in bases for k in zip(Cs["ruc"], Cs["fecha_hechos_min"])], index=Cs.index)
    Cs = Cs.loc[~repetida, CS_COLS]
    # la sanción de un consorcio llega a sus integrantes, salvo a los que el Tribunal dejó con 0 meses (HALLAZGOS 160, C10)
    ig = pd.read_parquet(PROC / "validado/resoluciones_integrantes.parquet")
    ig = ig[ig["ruc"].astype(str).str.fullmatch(r"\d{11}") & ~ig["meses_sancion"].astype(str).str.strip().isin(["0", "0.0", "0.00"])]
    ig = ig[["resolucion", "expediente", "ruc"]].astype({"ruc": str}).merge(res[res["sanciona"]][[c for c in CS_COLS if c != "ruc"] + ["expediente"]], on=["resolucion", "expediente"])
    ya = set(zip(Cs["resolucion"], Cs["ruc"])) | set(zip(res.loc[res["c_nucleo"], "resolucion"], res.loc[res["c_nucleo"], "ruc"]))
    ig = ig[[k not in ya for k in zip(ig["resolucion"], ig["ruc"])]].drop_duplicates(["resolucion", "ruc"])
    print(f"   S de consorcios a integrantes: {len(ig):,} de {ig['ruc'].nunique():,} proveedores; reconsideraciones repetidas fuera: {int(repetida.sum()):,}", flush=True)
    Cs = pd.concat([Cs.assign(metodo_entidad="extraida"), ig[CS_COLS].assign(metodo_entidad="integrante_consorcio")], ignore_index=True)
    Cs = Cs.rename(columns={"resolucion": "documento", "ruc_entidad_extraido": "ruc_entidad", "fecha_hechos_min": "desde",
                            "fecha_hechos_max": "hasta"}).assign(fuente="C_sin_orden")
    hr = pd.concat([B[["documento", "hecho", "fuente", "ruc_entidad", "ruc", "desde", "hasta", "metodo_entidad", "acreedor"]],
                    Cs], ignore_index=True)
    hr["ruc_entidad"] = hr["ruc_entidad"].astype("string")
    hr["es_impedimento"] = hr["es_impedimento"].astype("boolean")
    hr["acreedor"] = hr["acreedor"].fillna(False).astype(bool)
    hr.to_parquet(OUT / "hechos_relacion.parquet", index=False)
    print(f"   hechos de relación: B {int((hr['fuente'] == 'B').sum()):,} ({B['metodo_entidad'].value_counts().to_dict()}); "
          f"C sin orden {int((hr['fuente'] == 'C_sin_orden').sum()):,}", flush=True)
    return ho, hr


# ------------------------------------------------------------------------------------------------ pruebas
def pruebas(o, inf, res, ho, hr):
    paso("pruebas de la capa 1")
    fallas = []
    if o["id"].duplicated().any():
        fallas.append("ids de orden repetidos")
    if o.duplicated(["ruc_entidad", "orden"]).any():
        fallas.append("(entidad, orden) repetido")
    if (o["monto"] > o["tope"] + 1e-6).any():
        fallas.append(f"órdenes sobre el tope: {int((o['monto'] > o['tope'] + 1e-6).sum())}")
    if not ho["id"].isin(set(o["id"])).all():
        fallas.append("hechos de orden con id inexistente")
    if inf["informe"].duplicated().any():
        fallas.append("informes repetidos")
    neg = (inf["demora_publicacion_dias"] < 0).sum()
    if neg:
        fallas.append(f"informes publicados antes de emitirse: {int(neg)}")
    rep = int(ho.duplicated(["fuente", "documento", "hecho", "id", "ruc_administrado"]).sum())
    if rep:
        fallas.append(f"hechos de orden repetidos (mismo documento, hecho, orden y administrado): {rep}")
    print("   pruebas:", "todas bien" if not fallas else fallas, flush=True)
    return fallas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    o = ordenes()
    entidades_y_proveedores(o)
    inf = informes()
    res = resoluciones(o)
    ho, hr = hechos(o, inf, res)
    pd.read_parquet(SALIDA / "sanciones.parquet").to_parquet(OUT / "sanciones.parquet", index=False)
    pruebas(o, inf, res, ho, hr)
    paso("listo")


if __name__ == "__main__":
    main()
