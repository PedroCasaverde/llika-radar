"""Revisión de los pilotos de extracción con DeepSeek (extraer_deepseek.py): comprobaciones automáticas y un Excel para
que Pedro revise a mano lo que no se puede comprobar solo.

  python -X utf8 revisar_pilotos.py

Comprobaciones automáticas:
- Resoluciones: que cada RUC sancionado en el registro (ACC-7) aparezca entre los administrados extraídos, que los meses
  de inhabilitación coincidan con el registro y que el RUC tenga su dígito verificador bien.
- Informes: las órdenes contra las que copió Gian Carlo a mano (verdad de campo del piloto del 22/10), por número, monto,
  fecha y RUC.
- Ambos: que cada cita textual exista en el texto (exacta, aproximada o no encontrada).
Salida: datos/procesados/deepseek/piloto_revision.xlsx (fuera de git: trae nombres).
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from extraer_deepseek import PAPELERA, SAL, TCE, DATOS, cres, leer_texto, sin_trib, textos_disponibles
from tce_resoluciones import normalizar

PESOS = np.array([5, 4, 3, 2, 7, 6, 5, 4, 3, 2])


def ruc_ok(r):
    r = re.sub(r"\D", "", str(r or ""))
    if len(r) != 11 or r[:2] not in ("10", "15", "17", "20"):
        return False
    d = np.array(list(map(int, r)))
    x = 11 - int(d[:10] @ PESOS) % 11
    return (0 if x == 10 else 1 if x == 11 else x) == d[10]


def doc(x, n):
    """RUC (n=11) o DNI (n=8) como texto: el modelo a veces los devuelve como número, y el DNI puede empezar en 0."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        x = str(int(x))
    d = re.sub(r"\D", "", str(x))
    return d.zfill(n) if d else None


def plano(t):
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def cita(c, texto_plano):
    """exacta / aproximada (60 % o más de sus grupos de 4 palabras están en el texto) / no_encontrada / sin_cita."""
    q = plano(c)
    if not q:
        return "sin_cita"
    if q[:120] in texto_plano:
        return "exacta"
    w = q.split()
    grupos = [" ".join(w[i:i + 4]) for i in range(max(1, len(w) - 3))]
    return "aproximada" if np.mean([g in texto_plano for g in grupos]) >= 0.6 else "no_encontrada"


def leer_jsonl(archivo):
    return [json.loads(x) for x in archivo.open(encoding="utf-8")] if archivo.exists() else []


def num_orden(n):
    """Número de orden comparable: el primer grupo de dígitos, sin ceros a la izquierda."""
    m = re.search(r"\d+", str(n or ""))
    return m.group(0).lstrip("0") if m else None


# ------------------------------------------------------------------------------------------------------ resoluciones
def resoluciones():
    out = leer_jsonl(SAL / "piloto_resoluciones.jsonl")
    if not out:
        return pd.DataFrame(), {}
    mu = pd.read_csv(SAL / "piloto_resoluciones_muestra.csv", dtype=str)
    estrato = dict(zip(mu["numero"], mu["estrato"]))
    buscados = set(mu["numero"])
    textos = {}
    with (TCE / "textos.jsonl").open(encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            if d["numero"] in buscados:
                textos[d["numero"]] = plano(d["texto"])
    s = pd.read_parquet(DATOS / "procesados" / "radar" / "sanciones.parquet")
    s["n2"] = s["resolucion"].map(lambda r: sin_trib(normalizar(r)))
    s["meses_registro"] = ((s["fin"] - s["inicio"]).dt.days / 30.44).round()
    filas, metr = [], {"resoluciones": len(out), "con_error": 0, "rucs_registro": 0, "rucs_hallados": 0,
                       "meses_comparados": 0, "meses_iguales": 0, "usd": 0.0, "tokens_entrada": 0}
    for o in out:
        metr["usd"] += o.get("usd", 0.0)
        metr["tokens_entrada"] += o.get("tokens_entrada", 0)
        num = o["clave"]
        if "error" in o:
            metr["con_error"] += 1
            filas.append({"resolucion": num, "estrato": estrato.get(num), "error": o["error"]})
            continue
        t = textos.get(num, "")
        reg = s[s["n2"] == sin_trib(normalizar(num))]
        exps = o["salida"].get("expedientes") or []
        rucs_ext = set()
        for e in exps:
            rucs_ext.add(doc((e.get("administrado") or {}).get("ruc"), 11))
            rucs_ext |= {doc(i.get("ruc"), 11) for i in e.get("integrantes_consorcio") or []}
        metr["rucs_registro"] += reg["ruc"].nunique()
        metr["rucs_hallados"] += len(set(reg["ruc"]) & rucs_ext)
        for e in exps:
            adm, ent, con = e.get("administrado") or {}, e.get("entidad") or {}, e.get("contratacion") or {}
            san, imp = e.get("sancion") or {}, e.get("impedimento") or {}
            pv = imp.get("persona_vinculada") or {}
            dcu = e.get("documento_cuestionado") or {}
            ruc = doc(adm.get("ruc"), 11) or ""
            r_reg = reg[reg["ruc"] == ruc]
            meses_reg = r_reg["meses_registro"].iloc[0] if len(r_reg) and r_reg["fin"].notna().iloc[0] else None
            iguales = None
            if meses_reg is not None and san.get("meses") is not None:
                iguales = abs(float(san["meses"]) - meses_reg) <= 1
                metr["meses_comparados"] += 1
                metr["meses_iguales"] += int(iguales)
            filas.append({
                "resolucion": num, "estrato": estrato.get(num), "fecha_resolucion": o["salida"].get("fecha_resolucion"),
                "administrado": adm.get("nombre"), "ruc": doc(adm.get("ruc"), 11), "tipo_administrado": adm.get("tipo"),
                "ruc_valido": ruc_ok(ruc) if ruc else None, "en_registro": bool(len(r_reg)),
                "integrantes": "; ".join(f"{i.get('nombre')} ({i.get('ruc')})" for i in e.get("integrantes_consorcio") or []),
                "entidad": ent.get("nombre"), "contratacion": con.get("tipo"), "numero_contratacion": con.get("numero"),
                "procedimiento": con.get("procedimiento"), "fecha_contratacion": con.get("fecha"),
                "monto_soles": con.get("monto_soles"), "objeto": con.get("objeto"),
                "infracciones": "; ".join(f"{i.get('literal')}) {i.get('articulo')}" for i in e.get("infracciones") or []),
                "fecha_hechos": e.get("fecha_hechos"), "cita_fecha_hechos": e.get("cita_fecha_hechos"),
                "cita_fecha_hechos_en_texto": cita(e.get("cita_fecha_hechos"), t),
                "fecha_denuncia": e.get("fecha_denuncia"), "decision": e.get("decision"), "sancion": san.get("tipo"),
                "meses": san.get("meses"), "meses_registro": meses_reg, "meses_coinciden": iguales,
                "multa_soles": san.get("multa_soles"), "cita_decision_en_texto": cita(e.get("cita_decision"), t),
                "impedimento_literal": f"{imp.get('articulo')} {imp.get('literal')}" if imp else None,
                "causal": imp.get("causal"), "persona_vinculada": pv.get("nombre"), "cargo_vinculada": pv.get("cargo"),
                "entidad_vinculada": pv.get("entidad"), "relacion": pv.get("relacion"), "grado": pv.get("grado"),
                "cita_impedimento": imp.get("cita"), "cita_impedimento_en_texto": cita(imp.get("cita"), t) if imp else None,
                "documento": dcu.get("tipo"), "emisor": dcu.get("emisor"), "falso_o_inexacto": dcu.get("falso_o_inexacto"),
                "descargos": e.get("presento_descargos"), "revision_Pedro": ""})
    return pd.DataFrame(filas), metr


# ---------------------------------------------------------------------------------------------------------- informes
def informes():
    out = leer_jsonl(SAL / "piloto_informes.jsonl")
    if not out:
        return {}, {}
    mu = pd.read_csv(SAL / "piloto_informes_muestra.csv", dtype=str)
    estrato = dict(zip(mu["CresCodigoFormato"], mu["estrato"]))
    disp = textos_disponibles()
    gc = pd.read_csv(PAPELERA / "2026-10-22_ordenes_observadas_piloto.csv", dtype=str)
    gc["CresCodigoFormato"] = gc["enlace_pdf"].map(cres)
    gc = gc[gc["citada_en_informe"].str.lower().str.startswith("s", na=False)]
    hechos, prov, pers, ords, comp = [], [], [], [], []
    metr = {"informes": len(out), "con_error": 0, "usd": 0.0, "tokens_entrada": 0, "ordenes_gc": 0,
            "ordenes_gc_sin_numero": 0, "ordenes_gc_halladas": 0, "comparables_monto": 0, "monto_igual": 0,
            "comparables_fecha": 0, "fecha_igual": 0, "ruc_igual": 0, "comparables_ruc": 0,
            "ordenes_extraidas_en_piloto_gc": 0}
    for o in out:
        metr["usd"] += o.get("usd", 0.0)
        metr["tokens_entrada"] += o.get("tokens_entrada", 0)
        c = o["clave"]
        if "error" in o:
            metr["con_error"] += 1
            hechos.append({"informe": c, "estrato": estrato.get(c), "error": o["error"]})
            continue
        t = plano(leer_texto(disp[c])) if c in disp else ""
        sal = o["salida"]
        m = sal.get("muestra") or {}
        ext = []
        for h in sal.get("hechos") or []:
            v, d = h.get("vinculo") or {}, h.get("descargos") or {}
            hechos.append({
                "informe": c, "numero": sal.get("numero"), "estrato": estrato.get(c), "entidad": sal.get("entidad"),
                "muestra_ordenes": m.get("muestra_ordenes"), "universo_ordenes": m.get("universo_ordenes"),
                "muestra_soles": m.get("muestra_soles"), "cita_muestra_en_texto": cita(m.get("cita"), t),
                "hecho": h.get("n"), "sumilla": h.get("sumilla"), "tipos": ", ".join(h.get("tipos") or []),
                "desde": (h.get("fechas_hechos") or {}).get("desde"), "hasta": (h.get("fechas_hechos") or {}).get("hasta"),
                "perjuicio_soles": h.get("perjuicio_soles"), "normas": "; ".join(h.get("normas") or []),
                "ordenes": len(h.get("ordenes") or []), "proveedores": len(h.get("proveedores") or []),
                "personas": len(h.get("personas") or []), "vinculo": v.get("tipo"), "vinculo_descripcion":
                v.get("descripcion"), "cita_vinculo": v.get("cita"),
                "cita_vinculo_en_texto": cita(v.get("cita"), t) if v else None,
                "descargos_presentados": d.get("presentados"), "descargos_desvirtuan": d.get("desvirtuan"),
                "revision_Pedro": ""})
            for p in h.get("proveedores") or []:
                prov.append({"informe": c, "hecho": h.get("n"), "nombre": p.get("nombre"), "ruc": doc(p.get("ruc"), 11),
                             "ruc_valido": ruc_ok(p.get("ruc")) if p.get("ruc") else None,
                             "ruc_en_texto": (doc(p.get("ruc"), 11) in t) if doc(p.get("ruc"), 11) else None,
                             "rol": p.get("rol"), "cita": p.get("cita"), "cita_en_texto": cita(p.get("cita"), t),
                             "revision_Pedro": ""})
            for p in h.get("personas") or []:
                pers.append({"informe": c, "hecho": h.get("n"), "nombre": p.get("nombre"), "dni": doc(p.get("dni"), 8),
                             "dni_en_texto": (doc(p.get("dni"), 8) in t) if doc(p.get("dni"), 8) else None,
                             "cargo": p.get("cargo"), "rol": p.get("rol"), "responsabilidad": p.get("responsabilidad"),
                             "nombre_en_texto": plano(p.get("nombre")) in t, "revision_Pedro": ""})
            for x in h.get("ordenes") or []:
                ext.append({"informe": c, "hecho": h.get("n"), "tipo": x.get("tipo"), "numero": x.get("numero"),
                            "fecha": x.get("fecha"), "monto_soles": x.get("monto_soles"), "proveedor": x.get("proveedor"),
                            "ruc_proveedor": doc(x.get("ruc_proveedor"), 11), "clave": num_orden(x.get("numero"))})
        ords += ext
        g = gc[gc["CresCodigoFormato"] == c]
        if len(g):
            e = pd.DataFrame(ext) if ext else pd.DataFrame(columns=["clave", "monto_soles", "fecha", "ruc_proveedor"])
            metr["ordenes_extraidas_en_piloto_gc"] += len(e)
            for r in g.itertuples():
                k = num_orden(r.numero_orden)
                if k is None:       # Gian Carlo no anotó el número: no se puede buscar
                    metr["ordenes_gc_sin_numero"] += 1
                    continue
                cand = e[e["clave"] == k]
                metr["ordenes_gc"] += 1
                fila = {"informe": c, "numero_gc": r.numero_orden, "tipo_gc": r.tipo_orden, "fecha_gc": r.fecha_emision,
                        "monto_gc": r.monto, "ruc_gc": r.ruc_proveedor, "hallada": len(cand) > 0}
                if len(cand):
                    x = cand.iloc[0]
                    metr["ordenes_gc_halladas"] += 1
                    mi = fi = None      # solo se compara si Gian Carlo anotó el dato
                    if isinstance(r.monto, str) and r.monto.strip():
                        metr["comparables_monto"] += 1
                        mi = x["monto_soles"] is not None and abs(float(x["monto_soles"]) - float(r.monto)) <= 1
                        metr["monto_igual"] += int(mi)
                    if isinstance(r.fecha_emision, str) and r.fecha_emision.strip():
                        metr["comparables_fecha"] += 1
                        fi = str(x["fecha"])[:10] == r.fecha_emision.strip()[:10]
                        metr["fecha_igual"] += int(fi)
                    if isinstance(r.ruc_proveedor, str) and re.fullmatch(r"\d{11}", r.ruc_proveedor.strip()):  # sin los enmascarados
                        metr["comparables_ruc"] += 1
                        ri = doc(x["ruc_proveedor"], 11) == r.ruc_proveedor.strip()
                        metr["ruc_igual"] += int(ri)
                        fila["ruc_igual"] = ri
                    fila.update({"numero_ext": x["numero"], "fecha_ext": x["fecha"], "monto_ext": x["monto_soles"],
                                 "ruc_ext": x["ruc_proveedor"], "monto_igual": mi, "fecha_igual": fi})
                comp.append(fila)
    hojas = {"informes_hechos": pd.DataFrame(hechos), "informes_proveedores": pd.DataFrame(prov),
             "informes_personas": pd.DataFrame(pers), "informes_ordenes": pd.DataFrame(ords),
             "ordenes_vs_gian_carlo": pd.DataFrame(comp)}
    return hojas, metr


def main():
    r, mr = resoluciones()
    hojas, mi = informes()
    filas = []
    if mr:
        filas += [("Resoluciones leídas", mr["resoluciones"]), ("Resoluciones con error", mr["con_error"]),
                  ("RUC sancionados en el registro (ACC-7)", mr["rucs_registro"]),
                  ("de ellos, hallados entre los administrados", mr["rucs_hallados"]),
                  ("Meses de inhabilitación comparados con el registro", mr["meses_comparados"]),
                  ("de ellos, iguales (±1 mes)", mr["meses_iguales"]),
                  ("Costo de las resoluciones (dólares)", round(mr["usd"], 3)),
                  ("Tokens de entrada de las resoluciones", mr["tokens_entrada"])]
        for col in ("cita_fecha_hechos_en_texto", "cita_decision_en_texto", "cita_impedimento_en_texto"):
            if col in r:
                filas.append((f"Resoluciones, {col}", r[col].value_counts().to_dict().__repr__()))
    if mi:
        filas += [("Informes leídos", mi["informes"]), ("Informes con error", mi["con_error"]),
                  ("Órdenes de Gian Carlo citadas en el informe, con número", mi["ordenes_gc"]),
                  ("(sin número en su tabla, no comparables)", mi["ordenes_gc_sin_numero"]),
                  ("de ellas, halladas por el modelo", mi["ordenes_gc_halladas"]),
                  ("con monto para comparar", mi["comparables_monto"]), ("con el mismo monto (±1 sol)", mi["monto_igual"]),
                  ("con fecha para comparar", mi["comparables_fecha"]), ("con la misma fecha", mi["fecha_igual"]),
                  ("con RUC para comparar", mi["comparables_ruc"]), ("con el mismo RUC", mi["ruc_igual"]),
                  ("Órdenes extraídas en esos 20 informes", mi["ordenes_extraidas_en_piloto_gc"]),
                  ("Costo de los informes (dólares)", round(mi["usd"], 3)),
                  ("Tokens de entrada de los informes", mi["tokens_entrada"])]
        for hoja, col in (("informes_proveedores", "cita_en_texto"), ("informes_proveedores", "ruc_valido"),
                          ("informes_personas", "dni_en_texto"), ("informes_personas", "nombre_en_texto"),
                          ("informes_hechos", "cita_vinculo_en_texto")):
            d = hojas.get(hoja)
            if d is not None and col in d:
                filas.append((f"{hoja}, {col}", d[col].value_counts(dropna=False).to_dict().__repr__()))
    resumen = pd.DataFrame(filas, columns=["medida", "valor"])
    print(resumen.to_string(index=False))
    destino = SAL / "piloto_revision.xlsx"
    with pd.ExcelWriter(destino, engine="openpyxl") as w:
        resumen.to_excel(w, sheet_name="resumen", index=False)
        if len(r):
            r.to_excel(w, sheet_name="resoluciones", index=False)
        for nombre, d in hojas.items():
            if len(d):
                d.to_excel(w, sheet_name=nombre, index=False)
        from openpyxl.styles import Alignment, Font
        for ws in w.book.worksheets:
            ws.freeze_panes = "B2"
            for celda in ws[1]:
                celda.font = Font(name="Calibri", bold=True, size=10)
            for fila in ws.iter_rows(min_row=2):
                for celda in fila:
                    celda.font = Font(name="Calibri", size=10)
                    celda.alignment = Alignment(wrap_text=True, vertical="top")
            for col in ws.columns:
                largo = max(len(str(c.value or "")) for c in col[:50])
                ws.column_dimensions[col[0].column_letter].width = min(45, max(10, largo * 0.9))
    print("Excel:", destino)


if __name__ == "__main__":
    main()
