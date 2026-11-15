"""Paso 1 de HALLAZGOS 66: lee con reglas (sin OCR ni LLM) las fichas resumen bajadas por bajar_fichas.py.

Uso:  python -X utf8 leer_fichas.py
Salida en fichas/:
  fichas_informes.csv  una fila por informe: campos de la ficha, modalidad marcada, textos de hechos y recomendaciones
  fichas_hechos.csv    una fila por hecho (o situación adversa), con los montos que menciona
  fichas_personas.csv  una fila por persona comprendida: DNI, nombre y tipo de responsabilidad
  fichas_texto.jsonl   el texto completo de cada ficha (para revisar o para el LLM del paso 2)
"""
import json
import re
from pathlib import Path

import fitz
import pandas as pd

AQUI = Path(__file__).parent
FICHAS = AQUI / "fichas"

ETIQUETAS = {
    # Las fichas antiguas de control simultáneo (2018-2019) son de un oficio, no de un informe: «N° de oficio», etc.
    "numero_informe": r"N° de informe:|N° de oficio:|N° de Reporte de Avance:",
    "titulo": r"Título del informe:|Título del Reporte:|Asunto:",
    "objetivo": r"Objetivo general:|Objetivo de la auditoría:|Detalle:",
    "entidad": r"Entidad sujeta a control:|Entidad auditada:|Entidad:",
    "monto_objeto": r"Monto objeto del servicio \(de corresponder\):|Monto de la materia de control:",
    "monto_auditado": r"Monto auditado:",          # auditorías de cumplimiento
    "monto_examinado": r"Monto examinado:",
    "region": r"Región:",
    "provincia": r"Provincia:",
    "distrito": r"Distrito:",
    "fecha_emision": r"Fecha de emisión del? (?:informe|oficio|reporte):",
    "unidad_emisora": r"Unidad orgánica (?:u OCI )?que emite el (?:informe|oficio|reporte):",
    "reconstruccion": r"Reconstrucción con cambios:",
}
CORTES = r"Ubigeo:|II\. Tipo de servicio|II\. Modalidad|\(marcar con"
PIE = re.compile(r"(?m)^\s*\d{4}-C[A-Z]{2}-[\w]+-\d+\s*$")
# Los miles pueden ir con coma, punto, uno o varios espacios, o hasta un salto de línea del PDF («S/ 1  319 500,00»).
MONTO = re.compile(r"S/\.?\s*(\d{1,3}(?:(?:\s?[,.´'’]\s{0,2}|\s{1,3})\d{3})*(?:[.,] ?\d{2})?(?!\d)|\d+(?:[.,]\d{2})?(?!\d))")
SIN_ADVERSAS = re.compile(r"no se (?:han )?identificad[oa]s? situaciones adversas|no se identificaron situaciones adversas"
                          r"|^\s*(?:sin|no (?:presenta|posee|existen?|hay)) (?:observaciones|situaciones? adversas)", re.I | re.M)
COLUMNAS = ("civil", "penal", "adm_entidad", "adm_pas")


def a_numero(s):
    """'835,751.10' -> 835751.1 ; '100 098,00' -> 100098.0 ; '74,347,16' -> 74347.16 ; '553.203.00' -> 553203.0.
    Si el último separador va seguido de exactamente dos cifras, es el decimal; todos los demás separan miles."""
    s = re.sub(r"\s+", "", s)
    m = re.search(r"[.,](\d{2})$", s)
    entero, dec = (s[:m.start()], m.group(1)) if m else (s, "")
    s = re.sub(r"[.,´'’]", "", entero) + (f".{dec}" if dec else "")
    try:
        return float(s)
    except ValueError:
        return None


def campos(texto):
    plano = re.sub(r"\s+", " ", texto)
    todas = "|".join(f"(?:{p})" for p in list(ETIQUETAS.values()) + [CORTES])
    out = {}
    for clave, patron in ETIQUETAS.items():
        m = re.search(f"(?:{patron})\\s*(.*?)\\s*(?=(?:{todas})|$)", plano)
        out[clave] = m.group(1).strip() if m else None
    for clave in ("monto_objeto", "monto_auditado", "monto_examinado"):
        m = MONTO.search(out[clave] or "")
        out[f"{clave}_num"] = a_numero(m.group(1)) if m else None
    out["es_oficio"] = bool(re.search(r"N° de oficio:", plano))
    # Algunas fichas sacan las etiquetas y los valores en otro orden («Región: Provincia: Distrito: PIURA PIURA
    # CASTILLA», o «Fecha de emisión de 22/09/2023 informe:»).
    if not out["region"]:
        m = re.search(r"Región:\s*\nProvincia:\s*\nDistrito:\s*\n+(.+)\n(.+)\n(.+)\n", texto)
        if m:
            out["region"], out["provincia"], out["distrito"] = (v.strip() or None for v in m.groups())
    if not out["fecha_emision"] or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", out["fecha_emision"] or ""):
        m = re.search(r"Fecha de emisión.{0,40}?(\d{2}/\d{2}/\d{4})", plano)
        out["fecha_emision"] = m.group(1) if m else out["fecha_emision"]
    for clave in ("region", "provincia", "distrito"):
        out[clave] = out[clave] or None
    out["numero_informe"] = re.sub(r"^N° ", "", out["numero_informe"] or "") or None
    if out["reconstruccion"]:
        out["reconstruccion"] = "Si" if re.match(r"Si\s+X", out["reconstruccion"]) else "No"
    return out


def modalidad_marcada(texto):
    """La opción marcada en «II. Tipo de servicio»: en el texto, la X va en la línea que sigue a la opción. Se busca solo
    dentro de esa sección, porque una «X» puede aparecer en el título («TALLA M X 100») o en la entidad («ZONA
    REGISTRAL N° X»)."""
    m = re.search(r"II\. (?:Tipo de servicio|Modalidad)(.*?)III\. Resultados", texto, re.S)
    if not m:
        return None
    lineas = [l.strip() for l in m.group(1).split("\n")]
    for i, l in enumerate(lineas):
        if l == "X":
            opcion = []
            for previa in reversed(lineas[:i]):
                if re.fullmatch(r"\d+", previa) or previa.startswith("(marcar") or not previa:
                    break
                opcion.insert(0, previa)
            if opcion:
                return " ".join(opcion)
    return None


def secciones(texto):
    """Hechos (con su rótulo) y recomendaciones, de la sección III."""
    t = PIE.sub("", texto)
    m = re.search(r"III\. Resultados[^\n]*\n\s*1\s*\n(.*?):\s*\n(.*)", t, re.S)
    if not m:
        return None, None, None
    rotulo, resto = re.sub(r"\s+", " ", m.group(1)).strip(), m.group(2)
    partes = re.split(r"\n\s*2\s*\n\s*Recomendaciones:\s*\n", resto, maxsplit=1)
    hechos = partes[0]
    recom = None
    if len(partes) == 2:
        recom = re.split(r"\n\s*3\s*\n\s*Personas comprendidas", partes[1], maxsplit=1)[0]
    hechos = re.split(r"\n\s*3\s*\n\s*Personas comprendidas", hechos, maxsplit=1)[0]
    return rotulo, hechos.strip(), (recom.strip() if recom else None)


def separar_hechos(texto):
    if not texto:
        return []
    trozos = re.split(r"(?m)^\s*(\d{1,2})\s*[.\-–)]\s+", "\n" + texto)
    if len(trozos) < 3:
        return [texto.strip()]
    return [trozos[i + 1].strip() for i in range(1, len(trozos) - 1, 2)]


def personas(doc):
    """Tabla «Personas comprendidas»: DNI, nombre y la X de cada tipo de responsabilidad, por la posición de cada palabra."""
    filas, centros = [], None
    for pag in doc:
        palabras = pag.get_text("words")
        enc = {w[4]: w for w in palabras if w[4] in ("DNI", "Civil", "Penal")}
        adm = sorted((w for w in palabras if w[4] == "Adm."), key=lambda w: w[0])
        if "DNI" in enc and "Civil" in enc and "Penal" in enc and adm:
            cx = lambda w1, w2=None: ((w1[0] + (w2 or w1)[2]) / 2)
            # Las fichas antiguas (hasta ~2021) no traen la columna «Adm. PAS»: solo Civil, Penal y Adm. ENT.
            centros = {"civil": cx(enc["Civil"]), "penal": cx(enc["Penal"])}
            for a in adm:
                # «ENT»/«PAS» va a la derecha de «Adm.» en la misma línea, o debajo en algunas fichas (encabezado en dos renglones)
                siguiente = next((w for w in palabras if w[4] in ("ENT", "PAS") and (
                    (abs(w[1] - a[1]) < 3 and 0 < w[0] - a[2] < 10) or (0 < w[1] - a[1] < 20 and abs(w[0] - a[0]) < 10))),
                    None)
                clave = "adm_pas" if siguiente is not None and siguiente[4] == "PAS" else "adm_entidad"
                centros[clave] = cx(a, siguiente)
            limite_nombre = enc["Civil"][0] - 3
            y_desde = enc["DNI"][3]
        elif centros is None:
            continue
        else:
            y_desde = 0
        renglones = {}
        for w in palabras:
            if w[1] < y_desde:
                continue
            renglones.setdefault(round((w[1] + w[3]) / 2 / 3), []).append(w)
        for clave in sorted(renglones):
            ws = sorted(renglones[clave], key=lambda w: w[0])
            doc_id = next((w for w in ws if re.fullmatch(r"\d{8}|\d{9,12}", w[4]) and w[0] < 120), None)
            nombre = " ".join(w[4] for w in ws if w is not doc_id and w[2] <= limite_nombre and w[4] != "X")
            marcas = [w for w in ws if w[4] == "X" and w[0] > limite_nombre]
            if doc_id is not None:
                fila = {"documento": doc_id[4], "nombre": nombre, **{c: 0 for c in COLUMNAS}}
                for x in marcas:
                    xc = (x[0] + x[2]) / 2
                    fila[min(centros, key=lambda c: abs(centros[c] - xc))] = 1
                filas.append(fila)
            elif filas and nombre and not marcas and not re.search(r"\d{4}-C[A-Z]{2}", nombre):
                filas[-1]["nombre"] += " " + nombre          # nombre largo partido en dos renglones
    return filas


def main():
    inv = pd.read_csv(FICHAS / "inventario_fichas.csv", dtype=str)
    inv = inv[inv.estado == "ok"]
    informes, hechos, gente = [], [], []
    with open(FICHAS / "fichas_texto.jsonl", "w", encoding="utf-8") as fj:
        for fila in inv.itertuples():
            try:
                doc = fitz.open(FICHAS / "pdf" / fila.Periodo / fila.archivo)
                texto = "\n".join(p.get_text() for p in doc)
            except Exception as e:      # noqa: BLE001
                informes.append({"CresCodigoFormato": fila.CresCodigoFormato, "error": type(e).__name__})
                continue
            fj.write(json.dumps({"CresCodigoFormato": fila.CresCodigoFormato, "texto": texto}, ensure_ascii=False) + "\n")
            c = campos(texto)
            rotulo, txt_hechos, txt_recom = secciones(texto)
            lista_h = separar_hechos(txt_hechos)
            ps = personas(doc)
            informes.append({
                "CresCodigoFormato": fila.CresCodigoFormato, "Periodo": fila.Periodo,
                "tipo_ficha": "reporte_avance" if "reporte de avance" in texto[:200].lower() else
                              "posterior" if "control posterior" in texto[:400].lower() else
                              "simultaneo" if "control simult" in texto[:400].lower() else "otro",
                **c, "modalidad_marcada": modalidad_marcada(texto), "rotulo_hechos": rotulo,
                "sin_situaciones_adversas": bool(SIN_ADVERSAS.search(texto)), "n_hechos": len(lista_h),
                "n_personas": len(ps), "texto_hechos": txt_hechos, "texto_recomendaciones": txt_recom,
                "paginas": doc.page_count, "chars": len(texto)})
            for i, h in enumerate(lista_h, 1):
                montos = [a_numero(m) for m in MONTO.findall(h)]
                montos = [m for m in montos if m]
                hechos.append({"CresCodigoFormato": fila.CresCodigoFormato, "n": i, "texto": h,
                               "montos": ";".join(f"{m:.2f}" for m in montos), "monto_max": max(montos) if montos else None})
            for p in ps:
                gente.append({"CresCodigoFormato": fila.CresCodigoFormato, **p, "fuente": "ficha"})
    # Si la ficha no trae la tabla de personas (o no hay ficha) y el buscador sí las lista, se toman del buscador.
    con_personas = {g["CresCodigoFormato"] for g in gente}
    bus = pd.read_csv(AQUI / "listas" / "lista_unica_informes_ampliada.csv", dtype=str)
    for c, t in zip(bus.CresCodigoFormato, bus.Text):
        if c in con_personas or not isinstance(t, str):
            continue
        for m in re.finditer(r"DNI : (\d+) (.*?) RESPONSABILIDAD :(.*?)(?=DNI :|$)", t, re.S):
            tipos = m.group(3)
            gente.append({"CresCodigoFormato": c, "documento": m.group(1), "nombre": m.group(2).strip(),
                          "civil": int("CIVIL" in tipos), "penal": int("PENAL" in tipos),
                          "adm_entidad": int("ADMINISTRATIVO ENTIDAD" in tipos),
                          "adm_pas": int("ADMINISTRATIVO PAS" in tipos), "fuente": "buscador"})
    pd.DataFrame(informes).to_csv(FICHAS / "fichas_informes.csv", index=False)
    pd.DataFrame(hechos).to_csv(FICHAS / "fichas_hechos.csv", index=False)
    pd.DataFrame(gente).to_csv(FICHAS / "fichas_personas.csv", index=False)
    inf = pd.DataFrame(informes)
    print(f"Fichas leídas: {len(inf)}; hechos: {len(hechos)}; personas: {len(gente)}")
    for col in ("numero_informe", "titulo", "entidad", "fecha_emision", "modalidad_marcada", "rotulo_hechos"):
        if col in inf:
            print(f"  {col}: {inf[col].notna().mean():.1%} con dato")


if __name__ == "__main__":
    main()
