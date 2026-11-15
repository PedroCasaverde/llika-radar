"""Control de calidad de lo que leer_fichas.py sacó de las fichas resumen (HALLAZGOS 66), antes de clasificar con LLM.

Compara cada campo con una fuente independiente: los metadatos del buscador de la Contraloría (lista ampliada), que
traen número, entidad, fecha, ubigeo, modalidad, título y, para los informes con responsables, el DNI, el nombre y el
tipo de responsabilidad de cada persona. Además revisa los hechos y los montos.

Uso:  python -X utf8 validar_fichas.py
Salida: fichas/validacion/*.csv con cada discrepancia, y el resumen en pantalla.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).parent
F = AQUI / "fichas"
OUT = F / "validacion"


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()


def personas_buscador(texto):
    """'DNI : 00000000 APELLIDO APELLIDO NOMBRES RESPONSABILIDAD :  ADMINISTRATIVO ENTIDAD   PENAL  DNI : …'"""
    out = []
    if not isinstance(texto, str):
        return out
    for m in re.finditer(r"DNI : (\d+) (.*?) RESPONSABILIDAD :(.*?)(?=DNI :|$)", texto, re.S):
        tipos = m.group(3)
        out.append({"documento": m.group(1), "nombre_buscador": m.group(2).strip(),
                    "b_civil": int("CIVIL" in tipos), "b_penal": int("PENAL" in tipos),
                    "b_adm_entidad": int("ADMINISTRATIVO ENTIDAD" in tipos),
                    "b_adm_pas": int("ADMINISTRATIVO PAS" in tipos)})
    return out


def main():
    OUT.mkdir(exist_ok=True)
    inf = pd.read_csv(F / "fichas_informes.csv", dtype=str)
    hec = pd.read_csv(F / "fichas_hechos.csv", dtype=str)
    per = pd.read_csv(F / "fichas_personas.csv", dtype=str)
    bus = pd.read_csv(AQUI / "listas" / "lista_unica_informes_ampliada.csv", dtype=str)
    m = inf.merge(bus, on="CresCodigoFormato", how="left", suffixes=("", "_b"))
    res = []

    def chequeo(nombre, ok, detalle_cols):
        ok = ok.fillna(False)
        res.append((nombre, int(ok.sum()), len(ok), ok.mean()))
        m.loc[~ok, ["CresCodigoFormato"] + detalle_cols].to_csv(OUT / f"{nombre}.csv", index=False)

    # A. Campos frente al buscador
    chequeo("numero_informe", m.numero_informe.map(norm) == m.NumeroInforme.map(norm), ["numero_informe", "NumeroInforme"])
    chequeo("entidad", m.entidad.map(norm) == m.Entidad.map(norm), ["entidad", "Entidad"])
    fecha_b = pd.to_datetime(m.FechaEmision, format="%Y/%m/%d", errors="coerce")
    fecha_f = pd.to_datetime(m.fecha_emision, format="%d/%m/%Y", errors="coerce")
    chequeo("fecha_emision", fecha_b == fecha_f, ["fecha_emision", "FechaEmision"])
    chequeo("region", m.region.map(norm) == m.Departamento.map(norm), ["region", "Departamento"])
    chequeo("provincia", m.provincia.map(norm) == m.Provincia.map(norm), ["provincia", "Provincia"])
    chequeo("distrito", m.distrito.map(norm) == m.Distrito.map(norm), ["distrito", "Distrito"])
    mod_f = m.modalidad_marcada.map(norm).str.replace(r"\bDE\b", "", regex=True).str.split().map(lambda x: set(x or []))
    mod_b = m.ModalidadServicio.map(norm).str.replace(r"\bDE\b", "", regex=True).str.split().map(lambda x: set(x or []))
    chequeo("modalidad", pd.Series([bool(a == b or (a and b and (a <= b or b <= a))) for a, b in zip(mod_f, mod_b)],
                                   index=m.index), ["modalidad_marcada", "ModalidadServicio"])
    sim = pd.Series([SequenceMatcher(None, norm(a), norm(b)).ratio() for a, b in zip(m.titulo, m.Descripcion)],
                    index=m.index)
    m["similitud_titulo"] = sim.round(2)
    chequeo("titulo", sim >= 0.9, ["titulo", "Descripcion", "similitud_titulo"])

    # B. Personas frente al buscador: mismo DNI, mismo tipo de responsabilidad, mismo nombre (sin importar el orden)
    pb = pd.DataFrame([{"CresCodigoFormato": c, **p} for c, t in zip(bus.CresCodigoFormato, bus.Text)
                       for p in personas_buscador(t)])
    j = pb.merge(per, on=["CresCodigoFormato", "documento"], how="outer", indicator=True)
    falta_en_ficha = j[j._merge == "left_only"]
    sobra_en_ficha = j[j._merge == "right_only"]
    ambos = j[j._merge == "both"].copy()
    for c in ("civil", "penal", "adm_entidad", "adm_pas"):
        ambos[c] = ambos[c].astype(int)
    ambos["resp_ok"] = ((ambos.civil == ambos.b_civil) & (ambos.penal == ambos.b_penal)
                        & (ambos.adm_entidad == ambos.b_adm_entidad) & (ambos.adm_pas == ambos.b_adm_pas))
    ambos["nombre_ok"] = [set(norm(a).split()) == set(norm(b).split()) for a, b in zip(ambos.nombre, ambos.nombre_buscador)]
    res.append(("personas: DNI de la ficha que están en el buscador", len(ambos), len(ambos) + len(sobra_en_ficha),
                len(ambos) / max(1, len(ambos) + len(sobra_en_ficha))))
    res.append(("personas: DNI del buscador que están en la ficha", len(ambos), len(ambos) + len(falta_en_ficha),
                len(ambos) / max(1, len(ambos) + len(falta_en_ficha))))
    res.append(("personas: tipo de responsabilidad igual", int(ambos.resp_ok.sum()), len(ambos), ambos.resp_ok.mean()))
    res.append(("personas: nombre igual (mismas palabras)", int(ambos.nombre_ok.sum()), len(ambos), ambos.nombre_ok.mean()))
    falta_en_ficha.to_csv(OUT / "personas_faltan_en_ficha.csv", index=False)
    sobra_en_ficha.to_csv(OUT / "personas_sobran_en_ficha.csv", index=False)
    ambos[~ambos.resp_ok].to_csv(OUT / "personas_responsabilidad_distinta.csv", index=False)
    ambos[~ambos.nombre_ok][["CresCodigoFormato", "documento", "nombre", "nombre_buscador"]].to_csv(
        OUT / "personas_nombre_distinto.csv", index=False)

    # C. Hechos: vacíos, muy cortos, con texto de otra sección, y fichas sin hechos
    hec["largo"] = hec.texto.fillna("").str.len()
    sospechosos = hec[(hec.largo < 40) | hec.texto.fillna("").str.contains(
        r"Recomendaciones:|Personas comprendidas|Apéndice N", regex=True)]
    sospechosos.to_csv(OUT / "hechos_sospechosos.csv", index=False)
    res.append(("hechos sin sospecha (largo >= 40 y sin texto de otra sección)", len(hec) - len(sospechosos), len(hec),
                1 - len(sospechosos) / len(hec)))
    sin_h = inf[pd.to_numeric(inf.n_hechos, errors="coerce").fillna(0) == 0]
    sin_h.to_csv(OUT / "fichas_sin_hechos.csv", index=False)
    res.append(("fichas con al menos un hecho", len(inf) - len(sin_h), len(inf), 1 - len(sin_h) / len(inf)))

    # D. Montos: todo «S/» del texto de los hechos, crudo y leído, para revisar los raros
    filas = []
    for c, t in zip(hec.CresCodigoFormato, hec.texto.fillna("")):
        for mm in re.finditer(r"S/\.?\s?[\d][\d ,.]*", t):
            filas.append({"CresCodigoFormato": c, "crudo": mm.group(0).strip()})
    mon = pd.DataFrame(filas)
    if len(mon):
        mon.to_csv(OUT / "montos_crudos.csv", index=False)
        res.append(("montos «S/» encontrados en los hechos", len(mon), len(mon), 1.0))

    print(f"{'Chequeo':<62} {'bien':>7} {'de':>7} {'%':>7}")
    for nombre, ok, total, pct in res:
        print(f"{nombre:<62} {ok:>7} {total:>7} {pct:>7.1%}")
    print("Detalle de cada discrepancia en", OUT)


if __name__ == "__main__":
    main()
