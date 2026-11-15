"""OCDS del OECE (fuente C1; PENDIENTES.md, sección 5): de los JSONL comprimidos por año a tablas Parquet.

Uso:  python -X utf8 procesar_ocds.py [--procesos 6] [--muestra 2000]
  --muestra N  lee solo N registros por archivo (para verificar el esquema antes de procesar todo).

Cada línea es un «compiled release» (el estado más reciente de un proceso). Los archivos están partidos por la fecha
de esa última versión, no por la del proceso: un proceso de 2020 actualizado en 2024 está en ocds_2024. Por eso se
leen todos los años 2018-2026 y, si un ocid aparece en más de un archivo, queda la versión con la fecha más reciente.

Tablas (datos/procesados/ocds/):
  procesos        un proceso por ocid: entidad (RUC), método, categoría, valor referencial, fechas, n.° de postores
  postores        quién se presentó a cada proceso (RUC y nombre): sirve para ver quiénes compiten juntos
  adjudicaciones  a quién se adjudicó y por cuánto (una fila por proveedor ganador)
  contratos       contratos firmados: fecha, monto y plazo
  items           ítems del proceso con su código CUBSO y descripción: sirve para comparar el objeto
Usa un proceso del sistema por archivo (la lectura de JSON es trabajo de CPU; la GPU no ayuda aquí).
"""
import argparse
import gzip
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

DATOS = Path(__file__).resolve().parents[1]
ORIGEN = DATOS / "crudos" / "ocds_oece"
DESTINO = DATOS / "procesados" / "ocds"
TABLAS = ("procesos", "postores", "adjudicaciones", "contratos", "items")


def arreglar(s):
    """Tildes mal codificadas («PERÃš» -> «PERÚ»): UTF-8 leído como Latin-1. Solo si el arreglo funciona."""
    if isinstance(s, str) and ("Ã" in s or "Â" in s):
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return s
    return s


def ruc_de(id_):
    """'PE-RUC-20605147217' -> '20605147217'."""
    m = re.search(r"(\d{11})$", str(id_ or ""))
    return m.group(1) if m else None


def monto(v):
    v = v or {}
    return v.get("amount"), v.get("currency")


def leer(args):
    """Lee un archivo anual y deja sus tablas en Parquet (no las devuelve al proceso principal: ahorra memoria)."""
    archivo, muestra, partes = args
    filas = {t: [] for t in TABLAS}
    n = 0
    with gzip.open(archivo, "rt", encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            n += 1
            ocid, fecha = d.get("ocid"), d.get("date")
            t = d.get("tender") or {}
            # entidad: el RUC está en los identificadores adicionales del «buyer»
            ruc_ent = None
            for p in d.get("parties") or []:
                if "buyer" in (p.get("roles") or []):
                    for ai in p.get("additionalIdentifiers") or []:
                        if ai.get("scheme") == "PE-RUC":
                            ruc_ent = ai.get("id")
                    depto = (p.get("address") or {}).get("department")
                    break
            else:
                depto = None
            val, mon = monto(t.get("value"))
            per = t.get("tenderPeriod") or {}
            filas["procesos"].append({
                "ocid": ocid, "fecha_version": fecha, "archivo": archivo.name, "fuente": (d.get("sources") or [{}])[0].get("id"),
                "codigo_entidad": (d.get("buyer") or {}).get("id"), "entidad": arreglar((d.get("buyer") or {}).get("name")),
                "ruc_entidad": ruc_ent, "departamento": arreglar(depto), "proceso_id": t.get("id"),
                "nomenclatura": arreglar(t.get("title")), "descripcion": arreglar(t.get("description")),
                "metodo": t.get("procurementMethod"), "tipo_procedimiento": arreglar(t.get("procurementMethodDetails")),
                "categoria": t.get("mainProcurementCategory"), "estado": t.get("status"),
                "valor_referencial": val, "moneda": mon, "valor_pen": (t.get("value") or {}).get("amount_PEN"),
                "fecha_convocatoria": t.get("datePublished"), "inicio_ofertas": per.get("startDate"),
                "fin_ofertas": per.get("endDate"), "n_postores": t.get("numberOfTenderers"),
                "proyecto": arreglar(((d.get("planning") or {}).get("budget") or {}).get("project"))})
            for x in t.get("tenderers") or []:
                filas["postores"].append({"ocid": ocid, "archivo": archivo.name, "ruc": ruc_de(x.get("id")),
                                          "id_seace": x.get("id"), "nombre": arreglar(x.get("name"))})
            for a in d.get("awards") or []:
                av, am = monto(a.get("value"))
                for s in a.get("suppliers") or [{}]:
                    filas["adjudicaciones"].append({
                        "ocid": ocid, "archivo": archivo.name, "adjudicacion_id": a.get("id"), "fecha": a.get("date"),
                        "estado": a.get("status"), "monto": av, "moneda": am, "ruc": ruc_de(s.get("id")),
                        "id_seace": s.get("id"), "nombre": arreglar(s.get("name"))})
            for c in d.get("contracts") or []:
                cv, cm = monto(c.get("value"))
                cp = c.get("period") or {}
                filas["contratos"].append({
                    "ocid": ocid, "archivo": archivo.name, "contrato_id": c.get("id"), "adjudicacion_id": c.get("awardID"),
                    "fecha_firma": c.get("dateSigned"), "estado": c.get("status"), "monto": cv, "moneda": cm,
                    "inicio": cp.get("startDate"), "fin": cp.get("endDate")})
            for it in t.get("items") or []:
                cl = it.get("classification") or {}
                iv, _ = monto(it.get("totalValue"))
                filas["items"].append({
                    "ocid": ocid, "archivo": archivo.name, "item_id": it.get("id"), "cubso": cl.get("id"),
                    "cubso_desc": arreglar(cl.get("description")), "descripcion": arreglar((it.get("description") or "")[:300]),
                    "cantidad": it.get("quantity"), "valor": iv, "estado": it.get("statusDetails")})
            if muestra and n >= muestra:
                break
    for k, v in filas.items():
        pd.DataFrame(v).to_parquet(partes / f"{k}__{archivo.stem}.parquet", index=False)
    return archivo.name, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=3)
    ap.add_argument("--muestra", type=int, default=0)
    a = ap.parse_args()
    archivos = sorted(ORIGEN.glob("ocds_20*.jsonl.gz"))
    salida = DESTINO if not a.muestra else DESTINO / "muestra"
    partes_dir = salida / "_partes"
    partes_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Los archivos más grandes primero, para repartir mejor el trabajo entre los procesos
    archivos = sorted(archivos, key=lambda f: f.stat().st_size, reverse=True)
    with Pool(a.procesos) as pool:
        partes = pool.map(leer, [(f, a.muestra, partes_dir) for f in archivos], chunksize=1)
    print(f"Leídos {sum(n for _, n in partes):,} registros de {len(archivos)} archivos en {time.time() - t0:.0f} s",
          flush=True)
    tablas = {t: pd.concat([pd.read_parquet(f) for f in sorted(partes_dir.glob(f"{t}__*.parquet"))], ignore_index=True)
              for t in TABLAS}
    # Un ocid en varios archivos: queda su versión más reciente
    pr = tablas["procesos"].sort_values("fecha_version").drop_duplicates("ocid", keep="last")
    dup = len(tablas["procesos"]) - len(pr)
    archivo_de = dict(zip(pr.ocid, pr.archivo))
    for t in TABLAS[1:]:                  # de cada tabla, solo las filas de la versión que quedó
        df = tablas[t]
        tablas[t] = df[df.ocid.map(archivo_de).eq(df.archivo)].drop(columns="archivo")
    tablas["procesos"] = pr
    # Los consorcios no tienen RUC en el OCDS (traen un código interno del SEACE) ni la lista de sus integrantes
    for t in ("postores", "adjudicaciones"):
        tablas[t]["es_consorcio"] = tablas[t].ruc.isna() & tablas[t].nombre.fillna("").str.upper().str.startswith("CONSORCIO")
    for t, df in tablas.items():
        df.to_parquet(salida / f"{t}.parquet", index=False)
    # Controles
    p = tablas["procesos"]
    anio = pd.to_datetime(p.fecha_convocatoria.str[:10], errors="coerce").dt.year
    print(f"Procesos únicos: {len(p):,} (repetidos entre archivos: {dup:,})")
    print("Convocatoria por año:", anio.value_counts().sort_index().loc[2015:].to_dict())
    print(f"Con RUC de la entidad: {p.ruc_entidad.notna().mean():.1%} | con postores: {(p.n_postores.fillna(0) > 0).mean():.1%}")
    for t in ("postores", "adjudicaciones"):
        ok = tablas[t].ruc.fillna("").str.fullmatch(r"(10|15|17|20)\d{9}")
        cons = tablas[t].es_consorcio
        print(f"{t}: {len(tablas[t]):,} filas; RUC válido: {ok.mean():.1%}; consorcios (sin RUC): {cons.mean():.1%}; "
              f"otros sin RUC (extranjeros, etc.): {(~ok & ~cons).mean():.1%}")
    print("Tipo de procedimiento:", p.tipo_procedimiento.value_counts().head(8).to_dict())
    print("Quedan textos con tildes rotas:", int(p.entidad.fillna("").str.contains("Ã").sum()))
    print(f"Tablas en {salida}")


if __name__ == "__main__":
    main()
