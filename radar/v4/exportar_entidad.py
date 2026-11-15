"""Los datos reales de una o más entidades para la pantalla de llika.pe (HALLAZGOS, sección 94).

Arma el mismo JSON que usa la demo (demo/fuente/datos_demo.json), pero con datos reales: hasta 100 proveedores por
entidad en el orden de LLIKA, la ficha de cada uno con sus compras, su vínculo, sus antecedentes y las señales del
expediente, y el enlace al documento oficial de cada señal que lo tiene (la resolución del Tribunal en gob.pe o el informe
de control en la Contraloría). Lo que no tiene un documento por registro (penalidades, órdenes, declaraciones juradas)
dice de qué fuente sale.

    python -I -X utf8 exportar_entidad.py --entidades 20144801459 [otro RUC ...] [--anio 2025] [--max 100] [--salida ruta.json]

Necesita la lista congelada de 2025 y puntajes_2025.py ya corrido. La salida trae nombres, RUC y DNI: va a privado/,
fuera de git. No toca la lista ni sus huellas.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import expediente as X  # noqa: E402
from etiquetas import BASE  # noqa: E402

sys.argv = _a

RAIZ = Path(__file__).resolve().parents[2]
CRUDOS, PROC = RAIZ / "datos" / "crudos", RAIZ / "datos" / "procesados"
SAL = BASE.parent / "resultados" / "lista_unica_limpia"
PRIVADO = RAIZ / "privado"
ANIO, CORTE, TOPE_2025, MOCHILA_ALTA = 2025, pd.Timestamp("2025-12-31"), 42800.0, 0.98
HUELLA_LISTA = "189af867041d1270bf132ab6ca25041e9a3e782713171758dadf22a761a4aff2"
NB = " "
# Versiones de las fuentes publicadas justo después del corte del 31/12/2025 (lo conocido al cierre del año)
PENALIDADES = ["oece_penalidades/_historia/penalidades_v260_2026-01-03.csv", "oece_penalidades/_historia/penalidades2018_v249_2026-01-03.csv"]
MULTAS = "oece_multas/_historia/sancionados_multa_v265_2026-01-03.csv"
JNE = "jne_autoridades/Autoridades_Electas_20251113.xls"
RUBROS = {"c_rub_obras": "Obras y materiales de construcción", "c_rub_consultoria": "Consultoría", "c_rub_alimentos": "Alimentos",
          "c_rub_combustible": "Combustible", "c_rub_salud": "Salud", "c_rub_limpieza_seguridad": "Limpieza y seguridad",
          "c_rub_alquiler_transporte": "Alquiler y transporte", "c_rub_servicios_personales": "Servicios personales",
          "c_rub_tecnologia": "Tecnología", "c_rub_utiles": "Útiles y materiales de oficina", "c_rub_publicidad_eventos": "Publicidad y eventos"}
SENAL = {"md": "mismodueno", "sf": "funcionario", "vi": "inhabilitado", "ee": "espejo", "dp": "doble", "pg": "puerta",
         "dir": "direccion", "nu": "nuevo", "ic": "informes", "oc": "procesos"}
MOTIVO_LISTA = {"vínculo: autoridad o su empresa": "autoridad", "vínculo: pariente, es una coincidencia": "pariente", "compra con bandera": "bandera",
                "sin señal fuerte: compra más grande": "mayor", "sin señal fuerte: peso en la entidad": "peso"}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "setiembre", "octubre", "noviembre", "diciembre"]
T0 = time.time()


class ErrorDeEntrada(Exception):
    """Un error que la persona puede corregir: se muestra sin traza."""


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


# ---------- Formatos ----------
def soles(x):
    return f"S/{NB}{float(x):,.0f}"


def plural(n, uno, varios):
    return f"{int(n):,} {uno if int(n) == 1 else varios}"


def mes_anio(f):
    f = pd.Timestamp(f)
    return f"{MESES[f.month - 1]} de {f.year}"


def dia(f):
    f = pd.Timestamp(f)
    return f"{f.day} {MESES[f.month - 1][:3] if f.month != 9 else 'set'} {f.year}"


MINUSCULAS = {"de", "del", "la", "las", "los", "el", "y", "e", "en", "a", "por", "para", "con"}
SIGLAS = {"SAC", "S.A.C.", "SA", "S.A.", "SRL", "S.R.L.", "EIRL", "E.I.R.L.", "SAA", "S.A.A.", "SCRL", "S.C.R.L.", "UGEL", "OCI", "SAC.", "II", "III", "IV"}


def titulo(s):
    """Nombres como los publica el Estado (en mayúsculas), con mayúscula inicial; siglas y razón social intactas."""
    if not isinstance(s, str) or not s.strip():
        return ""
    # En los nombres que publica el OECE, la Ñ llega como «¿» (PE¿A): entre dos letras, siempre es una Ñ perdida
    s = re.sub(r"(?<=[A-Za-zÁÉÍÓÚÜáéíóúü])¿(?=[A-Za-zÁÉÍÓÚÜáéíóúü])", "Ñ", s)
    out = []
    for i, w in enumerate(s.split()):
        if w.upper() in SIGLAS or re.fullmatch(r"([A-Z]\.){2,}", w.upper()):
            out.append(w.upper())
        elif i > 0 and w.lower() in MINUSCULAS:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)


def slug(s):
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def miles_es(texto):
    """Los textos del expediente usan espacio de miles («S/ 12 400»); la pantalla, el formato es-PE con coma y espacio que no corta."""
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", ",", texto)
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", ",", t)
    return t.replace("S/ ", f"S/{NB}")


def ruc_valido(r):
    if not re.fullmatch(r"\d{11}", str(r)):
        return False
    dv = 11 - sum(int(a) * b for a, b in zip(r[:10], [5, 4, 3, 2, 7, 6, 5, 4, 3, 2])) % 11
    dv = 0 if dv == 10 else 1 if dv == 11 else dv
    return int(r[10]) == dv


# ---------- Documentos oficiales ----------
def num_resolucion(s):
    s = str(s).upper().replace(" ", "")
    m = re.search(r"0*(\d+)-(\d{4})-(TC[EP])-?(S\d+)?", s)
    return f"{int(m.group(1))}-{m.group(2)}-{m.group(3)}{('-' + m.group(4)) if m.group(4) else ''}" if m else None


def indice_resoluciones():
    i = pd.read_csv(CRUDOS / "tce_resoluciones" / "indice.csv", dtype=str, usecols=["numero", "url"])
    i["k"] = i["numero"].map(num_resolucion)
    return i.dropna(subset=["k"]).drop_duplicates("k").set_index("k")["url"].to_dict()


def indice_informes():
    r = pd.read_csv(CRUDOS / "acc8_informes_cgr" / "listas" / "lista_unica_informes_ampliada.csv", dtype=str,
                    usecols=["CresCodigoFormato", "NumeroInforme", "RutaCloudInforme"]).dropna(subset=["RutaCloudInforme"])
    r["url"] = r["RutaCloudInforme"].str.replace("http://", "https://", regex=False)
    d = r.drop_duplicates("NumeroInforme").set_index("NumeroInforme")["url"].to_dict()
    d.update(r.drop_duplicates("CresCodigoFormato").set_index("CresCodigoFormato")["url"].to_dict())
    return d


# ---------- Comprobaciones de entrada (manejo de errores) ----------
def requeridos():
    faltan = []
    for ruta, paso_ in ((SAL / "lista_2025_unica.parquet", "la lista congelada de 2025 (lista_unica_limpia.py congelar)"),
                        (SAL / "puntajes_2025.parquet", "los puntajes de 2025 (puntajes_2025.py)"),
                        (SAL / "puntajes_2025_proveedor.parquet", "los puntajes de 2025 (puntajes_2025.py)"),
                        (BASE / "ordenes.parquet", "la base de órdenes (capa1_base.py)"),
                        (BASE / "ordenes_texto.parquet", "la base de órdenes (capa1_base.py)"),
                        (PROC / "radar" / "tarjeta_territorial.parquet", "la tarjeta territorial (radar/25_tarjeta_territorial.py)"),
                        (CRUDOS / "tce_resoluciones" / "indice.csv", "el índice de resoluciones (datos/_scripts/tce_resoluciones.py)"),
                        (CRUDOS / "acc8_informes_cgr" / "listas" / "lista_unica_informes_ampliada.csv", "la lista de informes de control (datos/crudos/acc8_informes_cgr)")):
        if not ruta.exists():
            faltan.append(f"  - falta {ruta.relative_to(RAIZ)}: se arma con {paso_}")
    if faltan:
        raise ErrorDeEntrada("No están los datos que necesita este paso:\n" + "\n".join(dict.fromkeys(faltan)))


def validar_argumentos(rucs, anio, maximo):
    """Lo que se puede revisar sin leer datos: el año, el número de proveedores y cada RUC."""
    if anio != ANIO:
        raise ErrorDeEntrada(f"El año {anio} no tiene lista congelada para mostrar. La pantalla usa la lista de {ANIO}; las de 2022 y 2024 son de evaluación.")
    if not 1 <= maximo <= 100:
        raise ErrorDeEntrada("--max va de 1 a 100.")
    if len(set(rucs)) != len(rucs):
        raise ErrorDeEntrada("Hay un RUC repetido en --entidades.")
    for r in rucs:
        if not re.fullmatch(r"\d{11}", r):
            raise ErrorDeEntrada(f"«{r}» no es un RUC: tiene que tener 11 dígitos.")
        if not ruc_valido(r):
            raise ErrorDeEntrada(f"El RUC {r} no tiene un dígito verificador válido. Revisa si está bien escrito.")


def validar_entidades(rucs, ents, puntajes):
    for r in rucs:
        if r not in ents.index:
            raise ErrorDeEntrada(f"El RUC {r} no está en el catálogo de entidades del OECE: no es una entidad pública que compre.")
        if r not in puntajes:
            raise ErrorDeEntrada(f"{titulo(ents.loc[r, 'nombre'])} (RUC {r}) no tiene compras menores en {ANIO} dentro del universo de LLIKA.")


# ---------- Piezas de la ficha ----------
def autoridades_jne():
    a = pd.read_excel(CRUDOS / JNE, dtype=str)
    a["nombre_aut"] = (a["NOMBRES"].fillna("") + " " + a["APELLIDOPATERNO"].fillna("") + " " + a["APELLIDOMATERNO"].fillna("")).map(titulo)
    a["periodo"] = pd.to_datetime(a["FEINICIOVIGENCIA"], errors="coerce").dt.year.astype("Int64").astype(str) + "-" + pd.to_datetime(a["FEFINVIGENCIA"], errors="coerce").dt.year.astype("Int64").astype(str)
    return a.drop_duplicates("DOCUMENTOIDENTIDAD").set_index("DOCUMENTOIDENTIDAD")[["nombre_aut", "CARGO", "periodo"]]


def cargo_txt(c):
    return str(c or "autoridad").lower()


def vinculo(caso, ids, tarjeta, jne, padron_par):
    """El vínculo de los grados 3 (autoridad o su empresa) y 2 (pariente por apellidos), con su autoridad del JNE."""
    t = tarjeta[tarjeta["id"].isin(ids)]
    g = int(caso["grado"])
    fuente_jne = "Autoridades electas (JNE)"
    if g == 3:
        for tipo, txt in (("autoridad_misma", "Es una autoridad electa de esta zona: {nombre} (cargo: {cargo}, {periodo})."), ("empresa_de_autoridad", "Uno de sus socios o representantes es {nombre} (cargo: {cargo}, {periodo}).")):
            s = t[t["tipo"] == tipo]
            if len(s):
                a = jne.reindex([s["dni_aut"].iloc[0]]).iloc[0]
                x = txt.format(cargo=cargo_txt(a["CARGO"]), periodo=a["periodo"], nombre=a["nombre_aut"])
                return {"tipo": "autoridad", "x": x, "fuente": fuente_jne + (" y socios y representantes de empresas (OECE)" if tipo != "autoridad_misma" else "")}, cargo_txt(a["CARGO"])
        return {"tipo": "autoridad", "x": "Es una autoridad electa de esta zona o una empresa suya.", "fuente": fuente_jne}, "autoridad"
    if g == 2:
        s = t[t["tipo"] == "pariente_posible"].sort_values("freq_par")
        if len(s):
            f = s.iloc[0]
            a = jne.reindex([f["dni_aut"]]).iloc[0]
            n = f["freq_par"]
            x = (f"Mismos dos apellidos que {a['nombre_aut']} (cargo: {cargo_txt(a['CARGO'])}, {a['periodo']})."
                 + (f" En el padrón de SUNAT, {plural(n, 'persona tiene', 'personas tienen')} esa combinación de apellidos." if pd.notna(n) else ""))
            return {"tipo": "pariente", "prob": None, "x": x, "fuente": fuente_jne + " y padrón de SUNAT"}, cargo_txt(a["CARGO"])
        return {"tipo": "pariente", "prob": None, "x": "Tiene los mismos dos apellidos que una autoridad de esta zona.", "fuente": fuente_jne}, "autoridad"
    return None, None


def antecedentes(rucs, idx_res):
    """Penalidades, multas e inhabilitaciones publicadas hasta el cierre de 2025, cada una con su documento si lo hay."""
    pen = pd.concat([pd.read_csv(CRUDOS / f, sep="|", dtype=str, encoding="cp1252", encoding_errors="replace") for f in PENALIDADES], ignore_index=True)
    pen.columns = [c.strip().upper() for c in pen.columns]
    pen = pen[pen["RUC CONTRATISTA"].isin(rucs)].copy()
    f = pen["FECHA PENALIDAD"].str.strip()
    pen["f"] = pd.to_datetime(f, errors="coerce", format="%d/%m/%y").fillna(pd.to_datetime(f, errors="coerce", format="%d/%m/%Y"))
    pen = pen[pen["f"] <= CORTE].drop_duplicates(["ID CONTRATO", "RUC CONTRATISTA", "FECHA PENALIDAD", "MONTO"])
    mul = pd.read_csv(CRUDOS / MULTAS, sep="|", dtype=str, encoding="cp1252", encoding_errors="replace")
    mul = mul[mul["RUC"].isin(rucs)].rename(columns={"NUMERO_RESOLUCION": "Resolucion", "MONTO": "Monto"}).copy()
    mul["f"] = pd.to_datetime(mul["FECHA_INICIO"], errors="coerce", format="%Y%m%d")
    mul = mul[mul["f"] <= CORTE]
    san = pd.read_parquet(BASE / "sanciones.parquet").astype({"ruc": str})
    san = san[san["ruc"].isin(rucs) & (pd.to_datetime(san["inicio"]) <= CORTE)]
    out = {r: {"pen": [], "inh": [], "mul": []} for r in rucs}
    for _, p in pen.sort_values("f").iterrows():
        tipo = str(p.get("TIPO PENALIDAD") or "").strip().lower()
        tipo = "" if tipo in ("", "nan") or "otra" in tipo else tipo if tipo.startswith("por ") else "por " + tipo
        ent = titulo(str(p.get("ENTIDAD CONTRATANTE") or "otra entidad"))
        monto = pd.to_numeric(str(p.get("MONTO") or "").replace(",", ""), errors="coerce")
        x = f"Penalidad{(' ' + tipo) if tipo else ''}, aplicada por {ent} ({mes_anio(p['f'])}{', ' + soles(monto) if pd.notna(monto) else ''}). Contrato {p['ID CONTRATO']}."
        out[p["RUC CONTRATISTA"]]["pen"].append({"x": x, "nota": "El OECE publica la penalidad en su registro, sin el documento."})
    for _, s in san.sort_values("inicio").iterrows():
        k = num_resolucion(s["resolucion"])
        fin = pd.to_datetime(s["fin"], errors="coerce")
        cuando = f"del {dia(s['inicio'])} al {dia(fin)}" if pd.notna(fin) and not s.get("definitiva") else f"desde el {dia(s['inicio'])}"
        x = f"{'Inhabilitación definitiva' if s.get('definitiva') else 'Inhabilitación temporal'} del Tribunal de Contrataciones (resolución {s['resolucion']}), {cuando}."
        out[s["ruc"]]["inh"].append({"x": x, **({"url": idx_res[k], "doc": "Ver resolución"} if k in idx_res else {"nota": "Resolución no ubicada en gob.pe: búsquela por su número."})})
    for _, m in mul.sort_values("f").iterrows():
        k = num_resolucion(m["Resolucion"])
        monto = pd.to_numeric(str(m.get("Monto") or "").replace(",", ""), errors="coerce")
        x = f"Multa del Tribunal de Contrataciones (resolución {m['Resolucion']}, {mes_anio(m['f'])}{', ' + soles(monto) if pd.notna(monto) else ''})."
        out[m["RUC"]]["mul"].append({"x": x, **({"url": idx_res[k], "doc": "Ver resolución"} if k in idx_res else {"nota": "Resolución no ubicada en gob.pe: búsquela por su número."})})
    return out


def senales_expediente(keys):
    """Las diez señales del expediente (expediente.py) para los casos pedidos, no solo los cinco de la lista."""
    K3 = ["ruc_entidad", "ruc", "anio"]
    x = keys[K3].copy()
    md = X.mismo_dueno(ANIO)
    sf, _ = X.sigue_funcionario(ANIO)
    x = x.merge(md, on=K3, how="left").merge(sf, on=K3, how="left")
    for f in (X.vende_inhabilitado, X.empresa_espejo, X.doble_percepcion, X.puerta_giratoria):
        x = x.merge(f(ANIO), on=K3, how="left")
    x = x.merge(X.de_la_x(ANIO), on=K3, how="left")
    x = x.merge(X.informes_previos(ANIO), on="ruc", how="left")
    return x


def documentos_informes(rucs):
    """Los informes de control en que aparece cada proveedor, conocidos al cierre de 2025 (como expediente.informes_previos)."""
    H = X.Hechos(conocimiento="publicacion")
    e = H.ev[H.ev["fuente"].isin(["A", "B"]) & (H.ev["fecha_conocido"] <= CORTE)].astype({"ruc": str})
    e = e[e["ruc"].isin(rucs)].sort_values("fecha_conocido", ascending=False).drop_duplicates(["ruc", "documento"])
    return e.groupby("ruc")["documento"].apply(list).to_dict()


def exp_items(fila, ruc, docs_inf, idx_res, idx_inf):
    items = []
    for pre, t in SENAL.items():
        col = [c for c in fila.index if c.startswith(pre + "_") and c.endswith("_texto")]
        if not col or not isinstance(fila[col[0]], str) or not fila[col[0]]:
            continue
        it = {"t": t, "x": miles_es(fila[col[0]])}
        if t in ("inhabilitado", "espejo"):
            k = num_resolucion(fila[col[0]])
            if k in idx_res:
                it.update(url=idx_res[k], doc="Ver resolución")
            else:
                it["nota"] = "Resolución no ubicada en gob.pe: búsquela por su número."
        if t == "informes":
            ds = docs_inf.get(ruc, [])
            con = [(d, idx_inf[d]) for d in ds if d in idx_inf]
            if con:
                it.update(url=con[0][1], doc="Ver informe (PDF)")
                if len(con) > 1:
                    it["otros"] = [{"n": d, "url": u} for d, u in con[1:5]]
            else:
                it["nota"] = "Informe no ubicado en el buscador de la Contraloría: búsquelo por su número."
        items.append(it)
    return items


def ordenes_del_caso(o):
    filas = []
    for _, r in o.sort_values(["fecha", "id"]).iterrows():
        m = re.match(r"(O/[SC])-(\d+)-", str(r["orden"]))
        num = f"{m.group(1)} {int(m.group(2)):04d}" if m else str(r["orden"]).split("-20")[0]
        filas.append([pd.Timestamp(r["fecha"]).strftime("%Y-%m-%d"), num, (r["descripcion"] or "Sin descripción en CONOSCE").strip(), round(float(r["monto"]), 2)])
    return filas


def motivo_de(c):
    con_bandera = bool(c["f30"]) or bool(c["f_cerca"])
    g = int(c["grado"])
    if g == 3:
        return "autoridad"
    if g == 2:
        return "pariente"
    if c["es_quinto"]:
        return "bandera" if con_bandera else "mayor"
    if c["percentil_historial"] >= MOCHILA_ALTA:
        hechos = sum(float(c[k] or 0) for k in ("h_prov_penalidades", "h_prov_sanciones_otras", "h_prov_multas", "h_prov_inhabilitaciones"))
        return "antecedentes" if hechos > 0 else "perfil"
    return "peso"


def por_que(motivo, c, ant, cargo, ords):
    if motivo == "autoridad":
        return f"Es una autoridad electa de esta zona ({cargo}) o una empresa suya." if cargo else "Es una autoridad electa de esta zona o una empresa suya."
    if motivo == "pariente":
        return f"Tiene los mismos dos apellidos que una autoridad electa de esta zona ({cargo}): puede ser su pariente." if cargo else "Tiene los mismos dos apellidos que una autoridad electa de esta zona: puede ser su pariente."
    if motivo == "antecedentes":
        p = []
        for k, u, v in (("pen", "penalidad", "penalidades"), ("inh", "inhabilitación", "inhabilitaciones"), ("mul", "multa", "multas")):
            if ant.get(k):
                p.append(plural(len(ant[k]), u, v))
        txt = " y ".join([", ".join(p[:-1]), p[-1]] if len(p) > 1 else p) if p else "antecedentes"
        return f"Tiene {txt}, y vende de forma muy parecida a proveedores que después fueron sancionados."
    if motivo == "perfil":
        return "Vende de forma muy parecida a proveedores que después fueron sancionados. No tiene antecedentes."
    if motivo == "bandera":
        if c["f_cerca"]:
            return f"Una sola compra suya llega al {round(100 * max(o[3] for o in ords) / TOPE_2025)}% del tope para comprar sin concurso."
        return "Varias de sus compras, en 30 días o menos, suman más que el tope para comprar sin concurso."
    if motivo == "mayor":
        return "Es la compra más grande entre las que quedaban: ninguna otra parecía partida."
    cl = 100 * float(c["c_local"])
    if cl >= 99.5:
        return "Depende de esta entidad: le vende todo lo que vende al Estado."
    if cl >= 50:
        return f"Le vende a esta entidad el {round(cl)}% de lo que vende al Estado."
    return f"Le vendió {soles(c['c_monto'])} a esta entidad y se parece en parte a proveedores que después fueron sancionados."


def rubro(x):
    if x is None or not len(x):
        return "Otros bienes y servicios"
    v = {RUBROS[k]: float(x[k] or 0) for k in RUBROS if k in x.index}
    k = max(v, key=v.get) if v else None
    return k if k and v[k] > 0 else "Otros bienes y servicios"


def tipo_ent(t):
    t = str(t or "").lower()
    sector = ("Gobierno local" if "municipalidad" in t else "Gobierno regional" if "regional" in t else "Salud" if "salud" in t or "hospital" in t
              else "Educación" if "universidad" in t or "ugel" in t else "Gobierno nacional")
    return (t[:1].upper() + t[1:]) if t else "Entidad pública", sector


# ---------- Armado ----------
def main():
    ap = argparse.ArgumentParser(description="Datos reales de una o más entidades para la pantalla de llika.pe")
    ap.add_argument("--entidades", nargs="+", required=True, help="RUC de las entidades (11 dígitos)")
    ap.add_argument("--anio", type=int, default=ANIO)
    ap.add_argument("--max", type=int, default=100, help="proveedores por entidad (1 a 100)")
    ap.add_argument("--salida", type=Path, default=None)
    a = ap.parse_args()
    validar_argumentos(a.entidades, a.anio, a.max)
    requeridos()
    lista = SAL / "lista_2025_unica.parquet"
    if hashlib.sha256(lista.read_bytes()).hexdigest() != HUELLA_LISTA:
        raise ErrorDeEntrada("La lista de 2025 no tiene la huella de la lista congelada (189af867…): no se arma la pantalla con otra lista.")
    ents = pd.read_parquet(BASE / "entidades.parquet").astype({"ruc_entidad": str}).drop_duplicates("ruc_entidad").set_index("ruc_entidad")
    Q = pd.read_parquet(SAL / "puntajes_2025.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
    validar_entidades(a.entidades, ents, set(Q["ruc_entidad"]))
    Q = Q[Q["ruc_entidad"].isin(a.entidades) & (Q["puesto"] <= a.max)].copy()
    G = pd.read_parquet(lista, columns=["ruc_entidad", "ruc", "motivo"]).astype({"ruc_entidad": str, "ruc": str})
    paso(f"ETAPA casos: {len(Q):,} de {len(a.entidades)} entidades")

    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "orden", "fecha", "monto"], filters=[("anio", "==", ANIO), ("ruc_entidad", "in", a.entidades)]).astype({"ruc_entidad": str, "ruc": str})
    txt = pd.read_parquet(BASE / "ordenes_texto.parquet", columns=["id", "descripcion"], filters=[("anio", "==", ANIO), ("id", "in", o["id"].tolist())])
    o = o.merge(txt, on="id", how="left")
    antes = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc_entidad", "ruc", "anio", "monto"], filters=[("anio", "<", ANIO), ("ruc_entidad", "in", a.entidades)]).astype({"ruc_entidad": str, "ruc": str})
    previa = antes.groupby(["ruc_entidad", "ruc"]).agg(monto=("monto", "sum"), desde=("anio", "min"), anios=("anio", "nunique"))
    nombres = pd.read_parquet(BASE / "proveedores.parquet", columns=["ruc", "nombre"]).astype({"ruc": str}).drop_duplicates("ruc").set_index("ruc")["nombre"]
    xr = pd.read_parquet(BASE.parent / "x_caso_anio.parquet", columns=["ruc_entidad", "ruc", "anio", "p_exp_ocds_procesos", "p_exp_ocds_ganados"] + list(RUBROS), filters=[("anio", "==", ANIO), ("ruc_entidad", "in", a.entidades)]).astype({"ruc_entidad": str, "ruc": str}).set_index(["ruc_entidad", "ruc"])
    pv = pd.read_parquet(SAL / "puntajes_2025_proveedor.parquet").astype({"ruc": str}).set_index("ruc")
    tarjeta = pd.read_parquet(PROC / "radar" / "tarjeta_territorial.parquet", columns=["id", "tipo", "momento", "dni_aut", "freq_par"])
    tarjeta = tarjeta[tarjeta["id"].isin(set(o["id"])) & tarjeta["momento"].str.startswith("en_su_territorio")]
    jne = autoridades_jne()
    paso("ETAPA órdenes, nombres y autoridades")

    idx_res, idx_inf = indice_resoluciones(), indice_informes()
    rucs = sorted(set(Q["ruc"]))
    ANT = antecedentes(rucs, idx_res)
    EXP = senales_expediente(Q).set_index(["ruc_entidad", "ruc"])
    docs_inf = documentos_informes(rucs)
    paso("ETAPA antecedentes, expediente e informes")

    # Los motivos y sus tasas son las cifras del método (2022 y 2024), las mismas de la demo: en el repo, demo/fuente; en el curso, la fuente del prototipo
    demo = next(p for p in (RAIZ / "demo" / "fuente" / "datos_demo.json", RAIZ / "entregables" / "P4 - Prototipo" / "fuente" / "datos_demo.json") if p.exists())
    motivos = json.loads(demo.read_text(encoding="utf-8"))["motivos"]
    entidades, control = [], {"enlaces": {"con_documento": 0, "sin_documento": 0}, "monto_distinto": 0, "motivos_distintos_de_la_lista": 0}
    for re_ in a.entidades:
        e = ents.loc[re_]
        oe = o[o["ruc_entidad"] == re_]
        tipo, sector = tipo_ent(e["tipo"])
        casos = []
        for _, c in Q[Q["ruc_entidad"] == re_].sort_values("puesto").iterrows():
            oc = oe[oe["ruc"] == c["ruc"]]
            ords = ordenes_del_caso(oc)
            if abs(sum(x[3] for x in ords) - float(c["c_monto"])) > 1:
                control["monto_distinto"] += 1
            vin, cargo = vinculo(c, set(oc["id"]), tarjeta, jne, None)
            ant = {k: v for k, v in ANT[c["ruc"]].items() if v}
            mot = motivo_de(c)
            if c["puesto"] <= 5:
                g = G[(G["ruc_entidad"] == re_) & (G["ruc"] == c["ruc"])]["motivo"]
                esperado = MOTIVO_LISTA.get(g.iloc[0]) if len(g) else None
                if esperado and esperado != mot or (g.iloc[0] == "historial del proveedor" and mot not in ("antecedentes", "perfil")):
                    control["motivos_distintos_de_la_lista"] += 1
            fe = EXP.loc[(re_, c["ruc"])] if (re_, c["ruc"]) in EXP.index else None
            if isinstance(fe, pd.DataFrame):
                fe = fe.iloc[0]
            exp = exp_items(fe, c["ruc"], docs_inf, idx_res, idx_inf) if fe is not None else []
            for it in exp + [y for v in ant.values() for y in v]:
                control["enlaces"]["con_documento" if it.get("url") else "sin_documento"] += 1
            nat = c["ruc"].startswith("10")
            pr = previa.loc[(re_, c["ruc"])] if (re_, c["ruc"]) in previa.index else None
            p_ = pv.loc[c["ruc"]] if c["ruc"] in pv.index else None
            xc = xr.loc[(re_, c["ruc"])] if (re_, c["ruc"]) in xr.index else None
            num = lambda s, k: int(float(s[k])) if s is not None and k in s.index and pd.notna(s[k]) else 0  # noqa: E731
            casos.append({
                "rk": int(c["puesto"]), "motivo": mot, "por": por_que(mot, c, ant, cargo, ords),
                "prov": {"tipo": "natural" if nat else "empresa", "nombre": titulo(nombres.get(c["ruc"], "")), **({"dni": c["ruc"][2:10]} if nat else {}), "ruc": c["ruc"]},
                "rubro": rubro(xc),
                "clocal": int(round(100 * float(c["c_local"]))), "riesgo": "sancion" if c["pS"] >= c["pP"] else "penalidad",
                "vin": vin, "ant": ant,
                "previa": {"desde": int(pr["desde"]), "anios": int(pr["anios"]), "monto": round(float(pr["monto"]), 2)} if pr is not None else None,
                "ord": ords, "exp": exp,
                "con": {"proc": num(xc, "p_exp_ocds_procesos"), "gan": num(xc, "p_exp_ocds_ganados"), "ctr": num(p_, "p_ejec_contratos")},
            })
        nombre = titulo(e["nombre"])
        corto = titulo(str(e["distrito"] if "municipalidad distrital" in str(e["tipo"]).lower() else e["nombre"]))
        entidades.append({"id": slug(corto), "art": "la", "nombre": nombre, "corto": corto, "tipo": tipo, "sector": sector, "ruc": re_,
                          "departamento": titulo(e["departamento"]), "provincia": titulo(e["provincia"]),
                          "proveedores": int(oe["ruc"].nunique()), "ordenes": int(len(oe)), "monto": round(float(oe["monto"].sum()), 2), "casos": casos})
        paso(f"ETAPA {nombre}: {len(casos)} casos; {int(oe['ruc'].nunique())} proveedores; {len(oe)} órdenes")

    datos = {"modo": "real", "anio": str(ANIO), "tope": TOPE_2025, "azar": 0.3, "corte": "2025-12-31", "generado": time.strftime("%Y-%m-%d"),
             "lista": HUELLA_LISTA[:16], "motivos": motivos, "entidades": entidades}
    sal = a.salida or PRIVADO / f"datos_{'_'.join(x['id'] for x in entidades)}_{ANIO}.json"
    sal.parent.mkdir(parents=True, exist_ok=True)
    sal.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    control["huella_json"] = hashlib.sha256(sal.read_bytes()).hexdigest()
    control["minutos"] = round((time.time() - T0) / 60, 1)
    sal.with_suffix(".control.json").write_text(json.dumps(control, ensure_ascii=False, indent=1), encoding="utf-8")
    paso(f"FIN: {sal} | enlaces {control['enlaces']} | montos distintos {control['monto_distinto']} | motivos distintos {control['motivos_distintos_de_la_lista']} | huella {control['huella_json'][:16]}")


if __name__ == "__main__":
    try:
        main()
    except ErrorDeEntrada as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(2)
