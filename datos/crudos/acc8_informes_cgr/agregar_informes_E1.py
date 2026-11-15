"""Suma a la lista los informes de la respuesta de la Contraloría al pedido E1 (Información.XLSX) que nuestra búsqueda
por palabras no había encontrado (HALLAZGOS 66). Solo se usan para ubicar informes públicos: la respuesta no entra al
modelo.

Uso:  python -X utf8 agregar_informes_E1.py
- Escribe listas/lista_complementaria_E1.csv (los 75: con enlace al PDF o sin él).
- Baja la ficha resumen de los que tienen enlace y los agrega a fichas/inventario_fichas.csv.
Después: leer_fichas.py, clasificar_hechos.py (plantillas, correr, informes) y el lote 16 para leer el informe completo.
"""
import csv
import re
import time
from pathlib import Path

import pandas as pd
import requests

AQUI = Path(__file__).parent
EXCEL = AQUI.parents[2] / "Información.XLSX"


def norm(s):
    return re.sub(r"[^0-9A-Z]", "", str(s).upper())


def main():
    x = pd.read_excel(EXCEL, header=1)
    x["cres"] = x["URL SPIC"].astype(str).str.extract(r"CRES_CODIGO=([0-9A-Z]+)")[0]
    x["CresCodigoFormato"] = x.cres.map(lambda c: f"{c[:4]}-{c[4:7]}-{c[7:11]}-{c[11:]}"
                                        if isinstance(c, str) and len(c) >= 16 else None)
    lista = pd.read_csv(AQUI / "listas" / "lista_unica_informes_ampliada.csv", dtype=str,
                        usecols=["CresCodigoFormato", "NumeroInforme"])
    ya = x.CresCodigoFormato.isin(lista.CresCodigoFormato) | x["NUMERO INFORME"].map(norm).isin(
        lista.NumeroInforme.map(norm))
    n = x[~ya].copy()
    n["Periodo"] = pd.to_datetime(n.FECHA_INFORME, errors="coerce").dt.year.astype("Int64").astype(str)
    n["RutaCloudInforme"] = n.cres.map(lambda c: f"http://apps8.contraloria.gob.pe/SPIC/srvDownload/ViewPDF?"
                                                 f"CRES_CODIGO={c}&TIPOARCHIVO=ADJUNTO" if isinstance(c, str) else None)
    n["ResumenEjecutivo"] = n.RutaCloudInforme.str.replace("TIPOARCHIVO=ADJUNTO", "TIPOARCHIVO=RE", regex=False)
    # Sin código CRES (no tienen PDF publicado): se identifican con un código propio para no perderlos
    n["CresCodigoFormato"] = n.CresCodigoFormato.fillna("E1-" + n["NUMERO INFORME"].map(norm))
    comp = n.rename(columns={"NUMERO INFORME": "NumeroInforme", "ENTIDAD AUDITADA": "Entidad",
                             "TIPO SERVICIO": "ModalidadServicio", "TITULO_INFORME": "Descripcion",
                             "FECHA_INFORME": "FechaEmision"})[
        ["CresCodigoFormato", "NumeroInforme", "Periodo", "Entidad", "ModalidadServicio", "Descripcion", "FechaEmision",
         "RutaCloudInforme", "ResumenEjecutivo"]]
    comp["fuente"] = "respuesta_E1"
    comp.to_csv(AQUI / "listas" / "lista_complementaria_E1.csv", index=False)
    con = comp[comp.RutaCloudInforme.notna()]
    print(f"Informes nuevos: {len(comp)}; con PDF publicado: {len(con)}; sin enlace: {len(comp) - len(con)}")

    inv = AQUI / "fichas" / "inventario_fichas.csv"
    hechos = set(pd.read_csv(inv, dtype=str).CresCodigoFormato)
    s = requests.Session()
    with open(inv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for fila in con[~con.CresCodigoFormato.isin(hechos)].itertuples():
            ruta = AQUI / "fichas" / "pdf" / str(fila.Periodo) / f"{fila.CresCodigoFormato}.pdf"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            try:
                r = s.get(fila.ResumenEjecutivo.replace("http://", "https://"), timeout=60)
                ok = r.ok and r.content.startswith(b"%PDF")
                if ok:
                    ruta.write_bytes(r.content)
                w.writerow([fila.CresCodigoFormato, fila.Periodo, ruta.name if ok else "", len(r.content) if ok else "",
                            "ok" if ok else "no_es_pdf"])
            except Exception as e:      # noqa: BLE001
                w.writerow([fila.CresCodigoFormato, fila.Periodo, "", "", f"error: {type(e).__name__}"])
            time.sleep(0.3)
    print("Fichas agregadas al inventario")


if __name__ == "__main__":
    main()
