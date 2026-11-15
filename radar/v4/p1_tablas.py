"""Fase 1 (HALLAZGOS 160, sección 39): tablas por unidad, sin 2022 ni lo conocido desde 2026. No rehace la base.

Eventos (un hecho por fila, con actor y fechas), proveedor-año y entidad-año. Prueba de entrada: las etiquetas armadas desde Eventos reproducen puntajes_d18 con 0 diferencias.

    python -X utf8 p1_tablas.py
Salida: datos/procesados/v4/casos/ y modelos_corrida/p1_tablas.txt
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(RAIZ / "datos" / "_scripts"))
sys.path.insert(0, str(RAIZ / "datos" / "crudos" / "acc8_informes_cgr"))
_a, sys.argv = sys.argv, ["x"]
from casos import DIR, OUT, PRUEBA, TIPOS, etiquetas_caso  # noqa: E402
from comun import entidades, ruc_por_nombre  # noqa: E402
from etiquetas import BASE, HOY_EXPLORACION, K, Hechos  # noqa: E402
from evaluacion import cargar_x, universo  # noqa: E402
from extraer_deepseek import sin_trib  # noqa: E402
from m6_proveedor import Fuentes  # noqa: E402
from tce_resoluciones import normalizar  # noqa: E402
sys.argv = _a

ANIOS = (2018, 2019, 2020, 2021, 2023)
T0 = time.time()
FRASE = re.compile(r"(?:mediante|a trav[eé]s d[e]l?|con)\b[^.]{0,320}?(?:mesa de partes|puso en conocimiento|pone en conocimiento|comunic[oó]|denunci[oó]|inform[oó]|remiti[oó]|solicit[oó] (?:se )?(?:aplic|inici))[^.]{0,220}", re.I)
OECE = re.compile(r"OSCE-D|OECE-D|gesti[oó]n de ri[e]?sgos|identificaci[oó]n de ri[e]?sgos|\bDGR\b|\bSIRE\b|registro nacional de proveedores|\bDRNP\b|supervisi[oó]n de oficio|per[uú] compras", re.I)
CTRL = re.compile(r"contralor[ií]a|[oó]rgano de control institucional|\bOCI\b", re.I)
TRIB = re.compile(r"secretar[ií]a del tribunal|abri[oó] el presente expediente|sala del tribunal", re.I)
ENT = re.compile(r"la entidad|sanci[oó]n\s*[-–—]\s*entidad", re.I)
TER = re.compile(r"el señor|la señora|la empresa|el consorcio|denunciante|el ciudadano", re.I)


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def actor_de(t):
    t = re.sub(r"\s+", " ", t or "")
    h = re.search(r"ANTECEDENTES", t)
    ini = t[h.end():h.end() + 8000] if h else t[:8000]
    for s in FRASE.finditer(ini):
        g = s.group(0)
        if re.search(r"tribunal|sanci[oó]n|infracci[oó]n|mesa de partes|procedimiento", g, re.I):
            for nombre, c in (("OECE", OECE), ("Contraloría u OCI", CTRL), ("Tribunal", TRIB), ("entidad", ENT), ("tercero", TER)):
                if c.search(g):
                    return nombre
            return "no se sabe"
    return "no se sabe"


def clave(z):
    return sin_trib(normalizar(z))


def tabla_eventos(H, F):
    lit = pd.read_parquet(BASE / "resoluciones.parquet", columns=["resolucion", "impedimento_literal"]).drop_duplicates("resolucion").set_index("resolucion")["impedimento_literal"]
    ex = pd.read_parquet(RAIZ / "datos/procesados/validado/resoluciones_expedientes.parquet", columns=["resolucion", "literales", "fecha_denuncia"])
    ex["k"] = ex["resolucion"].map(clave)
    denuncia = pd.to_datetime(ex.dropna(subset=["fecha_denuncia"]).drop_duplicates("k").set_index("k")["fecha_denuncia"], errors="coerce")
    literales = ex.groupby("k")["literales"].agg(lambda s: ",".join(map(str, s)))
    est = H.ev[["fuente", "documento", "hecho", "ruc_entidad", "ruc", "anio_hechos", "fecha_conocido", "es_impedimento", "peso"]].copy()
    s = H.antecedentes.dropna(subset=["ruc", "anio_hechos", "fecha_conocido"]).assign(fuente="S", peso=1.0)[["fuente", "documento", "hecho", "ruc_entidad", "ruc", "anio_hechos", "fecha_conocido", "es_impedimento", "peso"]]
    p = F.p.copy()
    p["ruc_entidad"] = ruc_por_nombre(p["entidad"], entidades())
    pen = pd.DataFrame({"fuente": "P", "documento": p["contrato"].astype(str), "hecho": None, "ruc_entidad": p["ruc_entidad"], "ruc": p["ruc"], "anio_hechos": p["fecha"].dt.year, "fecha_conocido": p["fecha_conocido"],
                        "es_impedimento": False, "peso": 1.0, "monto": p["monto"], "tipo_penalidad": np.where(p["mora"], "mora", "otra"), "entidad_texto": p["entidad"]})
    mu = H.registros[H.registros["tipo"] == "multa"].dropna(subset=["anio_hechos", "fecha_conocido"])
    mul = pd.DataFrame({"fuente": "M", "ruc": mu["ruc"], "anio_hechos": mu["anio_hechos"], "fecha_conocido": mu["fecha_conocido"], "peso": 1.0, "es_impedimento": False})
    E = pd.concat([est, s, pen, mul], ignore_index=True)
    E = E.astype({"ruc": str})
    E["anio_hechos"] = E["anio_hechos"].astype(int)
    E["fecha_conocido"] = pd.to_datetime(E["fecha_conocido"])
    E["es_impedimento"] = E["es_impedimento"].fillna(False).astype(bool)
    paso(f"filas por fuente: {E['fuente'].value_counts().to_dict()}; penalidades con entidad {pen['ruc_entidad'].notna().mean():.1%}")
    tribunal = E["fuente"].isin(["C", "S"]).to_numpy()
    E["k"] = None
    E.loc[tribunal, "k"] = E.loc[tribunal, "documento"].map(clave)
    quiero = set(E["k"].dropna())
    actores = {}
    with (RAIZ / "datos/crudos/tce_resoluciones/textos.jsonl").open(encoding="utf-8") as f:
        for linea in f:
            x = json.loads(linea)
            k = clave(x["numero"])
            if k in quiero:
                actores[k] = actor_de(x.get("texto"))
    E["actor"] = E["k"].map(actores)
    E.loc[E["fuente"].isin(["A", "B"]), "actor"] = "Contraloría u OCI"
    E.loc[E["fuente"].eq("P"), "actor"] = "entidad"
    E.loc[E["fuente"].eq("M"), "actor"] = "Tribunal"
    E["actor"] = E["actor"].fillna("sin texto")
    E["fecha_denuncia"] = E["k"].map(denuncia)
    E["literales"] = E["k"].map(literales)
    E["impedimento_real"] = (E["fuente"] == "C").to_numpy() & E["documento"].map(lit).notna().to_numpy()
    paso(f"actor en sanciones: {E.loc[tribunal, 'actor'].value_counts().to_dict()}; resoluciones con texto {len(actores):,} de {len(quiero):,}")
    return E.drop(columns="k")


def prueba_entrada(E, H):
    filas = []
    for a in PRUEBA:
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet", columns=K + ["y", "y_estricta", "y_rev", "y_imp_real"]).astype(TIPOS)
        v = etiquetas_caso(E, b, HOY_EXPLORACION)
        fu = H.etiqueta_amplia(b, str(HOY_EXPLORACION.date()))["fuentes"].fillna("").astype(str)
        dif = {c: int((b[c].astype(bool).to_numpy() != v[c].to_numpy()).sum()) for c in ("y", "y_estricta", "y_rev", "y_imp_real")}
        dif["y_S"] = int((fu.str.contains("S").to_numpy() != v["y_S"].to_numpy()).sum())
        dif["y_P"] = int((fu.str.contains("P").to_numpy() != v["y_P"].to_numpy()).sum())
        filas.append({"anio": a, "casos": len(b), **dif})
        paso(f"prueba de entrada {a}: {filas[-1]}")
    return filas


def tabla_proveedor(H, F, anios=ANIOS):
    vistas = []
    for a in anios:
        x = cargar_x([a])
        hist = H.historia(x)
        x = pd.concat([x, hist], axis=1)
        x = x[universo(x, hist)]
        x = x.assign(ruc=x["ruc"].astype(str))
        cols = ["p_monto", "p_ordenes", "p_entidades", "p_exp_ocds_monto", "p_exp_ocds_ganados", "p_cat_monto", "p_anios_previos_activo", "p_nuevo",
                "h_prov_hechos", "h_prov_sanciones_otras", "h_prov_penalidades", "h_prov_multas", "h_prov_inhabilitaciones", "h_prov_inhabilitado_al_cierre"]
        g = x.groupby("ruc", sort=False).agg({**{c: "first" for c in cols}, "c_monto": "sum", "c_ordenes": "sum", "ruc_entidad": "nunique"}).rename(columns={"ruc_entidad": "casos"})
        g = g.join(F.al_cierre(a, g.index)).reset_index().assign(anio=a)
        vistas.append(g)
        paso(f"proveedor-año {a}: {len(g):,} proveedores")
    return pd.concat(vistas, ignore_index=True)


def tabla_entidad(E):
    i = pd.read_parquet(BASE / "informes.parquet", columns=["informe", "ruc_entidad", "servicio", "alcance_desde", "alcance_hasta", "fecha_emision", "fecha_publicacion"])
    f = i["fecha_publicacion"].fillna(i["fecha_emision"])
    j = i[f < pd.Timestamp("2026-01-01")].dropna(subset=["ruc_entidad", "alcance_desde", "alcance_hasta"]).copy()
    j["d"], j["h"] = pd.to_datetime(j["alcance_desde"], errors="coerce").dt.year, pd.to_datetime(j["alcance_hasta"], errors="coerce").dt.year
    j = j.dropna(subset=["d", "h"])
    filas = [(str(e), int(a)) for e, d, h in zip(j["ruc_entidad"], j["d"], j["h"]) for a in ANIOS if d <= a <= h]
    t = pd.DataFrame(filas, columns=["ruc_entidad", "anio"]).groupby(["ruc_entidad", "anio"]).size().rename("informes_que_cubren").reset_index()
    hal = E[E["fuente"].isin(["A", "B"])].dropna(subset=["ruc_entidad"]).astype({"ruc_entidad": str}).groupby(["ruc_entidad", "anio_hechos"]).size().rename("hallazgos").reset_index().rename(columns={"anio_hechos": "anio"})
    t = t.merge(hal, on=["ruc_entidad", "anio"], how="outer")
    t = t[t["anio"].isin(ANIOS)].fillna(0)
    t["control_sin_hallazgo"] = (t["informes_que_cubren"] > 0) & (t["hallazgos"] == 0)
    paso(f"entidades-año con control {int((t['informes_que_cubren'] > 0).sum()):,}; con control sin hallazgo {int(t['control_sin_hallazgo'].sum()):,}; límite: solo 2 419 informes traen alcance")
    return t


def main():
    DIR.mkdir(exist_ok=True)
    (OUT / "por_caso").mkdir(exist_ok=True)
    H = Hechos(conocimiento="publicacion")
    F = Fuentes()
    E = tabla_eventos(H, F)
    filas = prueba_entrada(E, H)
    malas = sum(v for f in filas for k, v in f.items() if k not in ("anio", "casos"))
    if malas:
        paso(f"PRUEBA DE ENTRADA FALLÓ: {malas} diferencias. Se para.")
        sys.exit(1)
    paso("PRUEBA DE ENTRADA OK: 0 diferencias")
    E.to_parquet(DIR / "eventos.parquet", index=False)
    P = tabla_proveedor(H, F)
    P.to_parquet(DIR / "proveedor_anio.parquet", index=False)
    T = tabla_entidad(E)
    T.to_parquet(DIR / "entidad_anio.parquet", index=False)
    (DIR / "manifiesto_p1.json").write_text(json.dumps({"script": "p1_tablas.py", "eventos": len(E), "proveedor_anio": len(P), "entidad_anio": len(T), "prueba_entrada": filas, "anios": ANIOS, "sellado": "sin 2022 ni lo conocido desde 2026"}, ensure_ascii=False, indent=1), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
