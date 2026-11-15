"""Expediente de cada fila de la lista: señales que no cambian el orden, para que el auditor o el ciudadano tengan más
fuentes y puedan cruzarlas (HALLAZGOS 160, sección 84; Pedro, 9/11: «que entre, pero para el expediente»).

Señales de esta versión (probadas en 2023 con sonda_senales_84.py: separan del promedio, pero no mejoran el orden):
- mismo dueño con varios RUC: la definición del v0 (radar/03_vinculos.py), U1 (la persona y su empresa) y U2 (dos o más
  empresas de la misma persona), cada RUC bajo el tope en la entidad y el año y juntos sobre él, con el corte de socios
  vigente el día de cada orden;
- proveedor que sigue a un funcionario: un funcionario de decisión llegó a la entidad en los dos años previos o en el año
  de la lista, desde otra entidad pública a la que el proveedor le vendía mientras él estaba ahí; el proveedor no le
  vendía a esta entidad en los dos años anteriores a esa llegada y le vende después.

No toca la lista congelada ni su huella: lee la lista y escribe un archivo aparte. Sin etiquetas. El texto que se muestra
no lleva nombres ni DNI de personas.

    python -I -X utf8 expediente.py 2025
Salida: resultados/lista_unica/expediente_<año>.parquet (con RUC: no sale de datos/) y expediente_<año>_resumen.json
"""
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etiquetas import BASE, Hechos  # noqa: E402
from evaluacion import X_PATH  # noqa: E402
import comun as CM  # noqa: E402  (radar/comun.py: alertas del v0 por orden y entidades)

PROC = BASE.parents[1]
SALIDA_L = BASE.parent / "resultados" / "lista_unica"
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def soles(x):
    return f"S/ {x:,.0f}".replace(",", " ")


def cuantos(n, uno, varios):
    """«1 orden», «3 órdenes»."""
    return n.astype(int).map(lambda k: f"{k} {uno if k == 1 else varios}")


def mismo_dueno(a):
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc_entidad", "ruc", "fecha", "monto", "tope"], filters=[("anio", "==", a)])
    o = o.astype({"ruc_entidad": str, "ruc": str})
    o["fecha"] = o["fecha"].astype("datetime64[ns]")
    partes = []
    for f in sorted(glob.glob(str(PROC / "acc3_conformacion" / "conformacion_corte*.parquet"))):
        corte = pd.Timestamp(Path(f).stem.replace("conformacion_corte", ""))
        if corte > pd.Timestamp(f"{a}-12-31") or corte < pd.Timestamp(f"{a - 1}-12-01"):
            continue
        v = pd.read_parquet(f, columns=["ruc_proveedor", "dni_integrante", "usable_para_cruce"])
        v = v[v["usable_para_cruce"] & v["dni_integrante"].notna()]
        partes.append(pd.DataFrame({"corte": corte, "emp": v["ruc_proveedor"].astype(str), "dni": v["dni_integrante"].astype(str)}).drop_duplicates())
    L = pd.concat(partes, ignore_index=True)
    L["corte"] = L["corte"].astype("datetime64[ns]")
    cortes = pd.DataFrame({"corte": np.sort(L["corte"].unique())}).astype({"corte": "datetime64[ns]"})
    o = pd.merge_asof(o.sort_values("fecha"), cortes, left_on="fecha", right_on="corte", direction="backward")
    assert o["corte"].notna().all(), f"hay órdenes de {a} sin un corte anterior"
    ventas = o.groupby(["ruc_entidad", "ruc"]).agg(monto=("monto", "sum"), tope=("tope", "max")).reset_index()
    ec = o[o["ruc"].str.startswith("20")][["ruc_entidad", "ruc", "corte"]].drop_duplicates()
    cp = ec.merge(L, left_on=["ruc", "corte"], right_on=["emp", "corte"])[["ruc_entidad", "ruc", "dni"]].drop_duplicates().merge(ventas, on=["ruc_entidad", "ruc"])
    r10 = ventas[ventas["ruc"].str.startswith("10")].assign(dni=lambda d: d["ruc"].str[2:10])
    u1 = r10.merge(cp, on=["ruc_entidad", "dni"], suffixes=("_p", "_e"))
    u1 = u1[(u1["monto_p"] <= u1["tope_p"]) & (u1["monto_e"] <= u1["tope_p"]) & (u1["monto_p"] + u1["monto_e"] > u1["tope_p"])]
    u1 = u1.assign(n=2, suma=u1["monto_p"] + u1["monto_e"], tope=u1["tope_p"])
    g = cp.groupby(["ruc_entidad", "dni"]).agg(n=("ruc", "nunique"), suma=("monto", "sum"), maximo=("monto", "max"), tope=("tope", "max")).reset_index()
    u2 = g[(g["n"] >= 2) & (g["maximo"] <= g["tope"]) & (g["suma"] > g["tope"])]
    casos = pd.concat([
        u1[["ruc_entidad", "ruc_p", "n", "suma", "tope"]].rename(columns={"ruc_p": "ruc"}).assign(tipo="la persona y su empresa"),
        u1[["ruc_entidad", "ruc_e", "n", "suma", "tope"]].rename(columns={"ruc_e": "ruc"}).assign(tipo="la persona y su empresa"),
        cp[["ruc_entidad", "ruc", "dni"]].merge(u2[["ruc_entidad", "dni", "n", "suma", "tope"]], on=["ruc_entidad", "dni"]).drop(columns="dni").assign(tipo="varias empresas del mismo dueño")])
    casos = casos.sort_values("suma", ascending=False).drop_duplicates(["ruc_entidad", "ruc"])
    casos["md_texto"] = ("Compra partida entre RUC del mismo dueño (" + casos["tipo"] + "): " + casos["n"].astype(int).astype(str)
                         + f" RUC le vendieron a esta entidad en {a}, cada uno bajo el tope; juntos suman "
                         + casos["suma"].map(soles) + ", sobre el tope de " + casos["tope"].map(soles) + ".")
    return casos[["ruc_entidad", "ruc", "md_texto"]].assign(anio=a, md_mismo_dueno=True)


def sigue_funcionario(a):
    e = pd.read_parquet(PROC / "radar" / "empleos_previos.parquet", columns=["persona", "ruc_entidad", "cargo", "decide", "inicio", "ruc_empleador", "nombre_empleador",
                                                                            "empleador_publico", "ruc_valido", "inicio_previo", "cese_previo"])
    e = e[e["decide"] & e["empleador_publico"] & e["ruc_valido"] & e["ruc_entidad"].notna() & e["inicio_previo"].notna()].astype({"ruc_entidad": str, "ruc_empleador": str})
    for c in ("inicio", "inicio_previo", "cese_previo"):
        e[c] = e[c].astype("datetime64[ns]")
    e = e[(e["ruc_entidad"] != e["ruc_empleador"]) & e["inicio"].between(pd.Timestamp(f"{a - 2}-01-01"), pd.Timestamp(f"{a}-12-31"))]
    fin = e["cese_previo"].fillna(e["inicio"])
    e["fin_a"] = fin.where(fin <= e["inicio"], e["inicio"])
    e = e[e["fin_a"] > e["inicio_previo"]].drop_duplicates(["persona", "ruc_entidad", "ruc_empleador", "inicio"]).reset_index(drop=True)
    e["par"] = np.arange(len(e))
    ents = sorted(set(e["ruc_entidad"]) | set(e["ruc_empleador"]))
    desde = max(pd.Timestamp("2018-01-01"), e["inicio_previo"].min())
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc_entidad", "ruc", "fecha"], filters=[("ruc_entidad", "in", ents), ("fecha", ">=", desde), ("fecha", "<=", pd.Timestamp(f"{a}-12-31"))])
    o = o.astype({"ruc_entidad": str, "ruc": str}).drop_duplicates()
    o["fecha"] = o["fecha"].astype("datetime64[ns]")
    pa = o.merge(e[["par", "ruc_empleador", "inicio_previo", "fin_a"]], left_on="ruc_entidad", right_on="ruc_empleador")
    pa = pa[(pa["fecha"] >= pa["inicio_previo"]) & (pa["fecha"] <= pa["fin_a"])][["par", "ruc"]].drop_duplicates()
    ob = o.merge(e[["par", "ruc_entidad", "inicio"]], on="ruc_entidad")
    antes = ob[(ob["fecha"] < ob["inicio"]) & (ob["fecha"] >= ob["inicio"] - pd.Timedelta(days=730))][["par", "ruc"]].drop_duplicates()
    despues = ob[(ob["fecha"] >= ob["inicio"]) & (ob["fecha"].dt.year == a)][["par", "ruc", "ruc_entidad"]].drop_duplicates()
    s = despues.merge(pa, on=["par", "ruc"]).merge(antes, on=["par", "ruc"], how="left", indicator=True)
    s = s[s["_merge"] == "left_only"].merge(e[["par", "cargo", "inicio", "nombre_empleador"]], on="par").sort_values("inicio")
    s = s.drop_duplicates(["ruc_entidad", "ruc"])
    s["sf_texto"] = ("Le vende a esta entidad desde que llegó, en " + s["inicio"].dt.strftime("%m/%Y") + ", una persona con cargo de decisión ("
                     + s["cargo"].fillna("cargo no indicado").str.capitalize() + ") que antes trabajaba en " + s["nombre_empleador"].fillna("otra entidad pública").str.title()
                     + "; en ese tiempo, este proveedor le vendía a esa entidad.")
    return s[["ruc_entidad", "ruc", "sf_texto"]].assign(anio=a, sf_sigue_funcionario=True), {"traslados": int(len(e)), "funcionarios": int(e["persona"].nunique())}


def por_orden(nombre, a, cols):
    """Una alerta del v0 (una fila por orden) llevada a los casos del año."""
    al = pd.read_parquet(CM.SALIDA / f"alerta_{nombre}.parquet", columns=["id"] + cols)
    o = pd.read_parquet(CM.SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "monto"], filters=[("anio", "==", a)])
    return o.merge(al, on="id").astype({"ruc_entidad": str, "ruc": str})


def vende_inhabilitado(a):
    x = por_orden("sancionado", a, ["definitiva", "resolucion"])
    g = x.groupby(["ruc_entidad", "ruc"]).agg(n=("id", "size"), monto=("monto", "sum"), resol=("resolucion", "first"), defin=("definitiva", "max")).reset_index()
    g["vi_texto"] = ("Recibió " + cuantos(g["n"], "orden", "órdenes") + " por " + g["monto"].map(soles) + f" en {a} mientras tenía una inhabilitación vigente ("
                     + g["resol"].astype(str) + ")." + np.where(g["defin"].astype(bool), " Es una inhabilitación definitiva.", ""))
    return g[["ruc_entidad", "ruc", "vi_texto"]].assign(anio=a, vi_vende_inhabilitado=True)


def empresa_espejo(a):
    x = por_orden("empresa_espejo", a, ["camino", "gremio", "ruc_sanc", "resolucion"]).sort_values("gremio")
    g = x.drop_duplicates(["ruc_entidad", "ruc"]).copy()
    empresa = g["camino"].eq("integra una empresa sancionada") & g["ruc_sanc"].astype(str).str.startswith("20")
    g["ee_texto"] = np.where(empresa, "Un socio o representante integra también una empresa sancionada (RUC " + g["ruc_sanc"].astype(str) + ", " + g["resolucion"].astype(str) + ") mientras dura su sanción.",
                             "Un socio o representante está sancionado como persona (" + g["resolucion"].astype(str) + ") mientras esta empresa vende.")
    g["ee_texto"] = g["ee_texto"] + np.where(g["gremio"].astype(bool), " El vínculo pasa por un gremio, así que pesa menos.", "")
    return g[["ruc_entidad", "ruc", "ee_texto"]].assign(anio=a, ee_empresa_espejo=True)


def doble_percepcion(a):
    x = por_orden("doble_percepcion", a, ["cargo", "ent_cargo", "docencia"]).sort_values("docencia")
    g = x.drop_duplicates(["ruc_entidad", "ruc"]).copy()
    ent = CM.entidades().drop_duplicates("RUC").set_index("RUC")["NOMBRE_DE_ENTIDAD"].str.title()
    g["dp_texto"] = ("El titular de este RUC declaró un cargo (" + g["cargo"].fillna("").str.capitalize() + ") en " + g["ent_cargo"].map(ent).fillna("otra entidad pública")
                     + " en las mismas fechas en que cobró órdenes de esta entidad." + np.where(g["docencia"].astype(bool), " Esta entidad es educativa: puede tratarse de docencia, que la ley permite.", ""))
    return g[["ruc_entidad", "ruc", "dp_texto"]].assign(anio=a, dp_doble_percepcion=True)


def puerta_giratoria(a):
    x = pd.read_parquet(CM.SALIDA / "alerta_puerta_giratoria.parquet", columns=["ruc_entidad", "ruc", "anio", "cargo", "locador", "vigente_en_empleador", "salio_antes_de_entrar", "educativo", "meses_desde_salida"])
    # sin locadores ni colaboradores por servicios: como en las alertas de funcionarios del v0, son personal contratado
    x = x[(x["anio"] == a) & ~x["locador"].astype(bool)].astype({"ruc_entidad": str, "ruc": str}).sort_values("vigente_en_empleador", ascending=False)
    g = x.drop_duplicates(["ruc_entidad", "ruc"]).copy()
    cargo = g["cargo"].fillna("cargo no indicado").str.capitalize()
    meses = g["meses_desde_salida"].round().astype("Int64").astype(str)
    g["pg_texto"] = np.where(g["vigente_en_empleador"].astype(bool), "Una persona con cargo en esta entidad (" + cargo + ") declaró trabajar también para este proveedor.",
                             "Una persona con cargo en esta entidad (" + cargo + ") trabajó antes en este proveedor; salió " + meses + " meses antes de entrar.")
    g["pg_texto"] = g["pg_texto"] + np.where(g["educativo"].astype(bool), " El empleo es en educación.", "")
    return g[["ruc_entidad", "ruc", "pg_texto"]].assign(anio=a, pg_puerta_giratoria=True)


def de_la_x(a):
    """Misma dirección, proveedor nuevo y señales de los procesos (OCDS), de la tabla de casos del año."""
    x = pd.read_parquet(X_PATH, columns=["ruc_entidad", "ruc", "anio", "c_misma_direccion_en_entidad", "p_nuevo", "p_ocds_ganados_postor_unico",
                                         "p_ocds_postula_inhabilitado", "p_ocds_gana_inhabilitado"], filters=[("anio", "==", a)]).astype({"ruc_entidad": str, "ruc": str, "anio": int})
    md = x["c_misma_direccion_en_entidad"].fillna(0)
    x["dir_misma_direccion"] = md > 0
    x["dir_texto"] = np.where(x["dir_misma_direccion"], "Comparte dirección fiscal con " + cuantos(md, "proveedor", "proveedores") + f" más de esta entidad en {a} (padrón de SUNAT, 2026).", None)
    x["nu_proveedor_nuevo"] = x["p_nuevo"].fillna(0) > 0
    x["nu_texto"] = np.where(x["nu_proveedor_nuevo"], f"{a} es su primer año vendiéndole al Estado (desde 2018).", None)
    partes = []
    for col, uno, varios in (("p_ocds_ganados_postor_unico", "ganó {} proceso sin competencia", "ganó {} procesos sin competencia"),
                             ("p_ocds_postula_inhabilitado", "postuló {} vez estando inhabilitado", "postuló {} veces estando inhabilitado"),
                             ("p_ocds_gana_inhabilitado", "ganó {} proceso estando inhabilitado", "ganó {} procesos estando inhabilitado")):
        v = x[col].fillna(0).astype(int)
        partes.append(np.where(v > 0, v.map(lambda n, u=uno, w=varios: (u if n == 1 else w).format(n)), ""))
    txt = pd.Series(["; ".join(p for p in fila if p) for fila in zip(*partes)], index=x.index)
    x["oc_procesos"] = txt.ne("")
    x["oc_texto"] = np.where(x["oc_procesos"], f"En {a}, " + txt + " (datos abiertos de contratación).", None)
    return x[["ruc_entidad", "ruc", "anio", "dir_misma_direccion", "dir_texto", "nu_proveedor_nuevo", "nu_texto", "oc_procesos", "oc_texto"]]


def informes_previos(a):
    H = Hechos(conocimiento="publicacion")
    e = H.ev[H.ev["fuente"].isin(["A", "B"]) & (H.ev["fecha_conocido"] <= pd.Timestamp(f"{a}-12-31"))].astype({"ruc": str}).sort_values("fecha_conocido")
    g = e.groupby("ruc").agg(n=("documento", "nunique"), ultimo=("documento", "last"), fecha=("fecha_conocido", "last")).reset_index()
    g["ic_texto"] = ("Aparece en " + cuantos(g["n"], "informe de control conocido", "informes de control conocidos") + f" hasta el cierre de {a}; el más reciente, "
                     + g["ultimo"].astype(str) + ", del " + pd.to_datetime(g["fecha"]).dt.strftime("%d/%m/%Y") + ".")
    return g[["ruc", "ic_texto"]].assign(ic_informes_previos=True)


def main():
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    ruta = SALIDA_L / f"lista_{a}_unica.parquet"
    ruta = ruta if ruta.exists() else SALIDA_L / f"lista_{a}.parquet"
    huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
    L = pd.read_parquet(ruta, columns=["ruc_entidad", "ruc", "anio", "motivo"]).astype({"ruc_entidad": str, "ruc": str, "anio": int})
    paso(f"ETAPA lista de {a}: {len(L):,} filas")
    md = mismo_dueno(a)
    paso(f"ETAPA mismo dueño: {len(md):,} casos en {a}")
    sf, info = sigue_funcionario(a)
    paso(f"ETAPA sigue al funcionario: {len(sf):,} casos en {a}; {info}")
    K3 = ["ruc_entidad", "ruc", "anio"]
    x = L.merge(md, on=K3, how="left").merge(sf, on=K3, how="left")
    for nombre, f in (("vende inhabilitado", vende_inhabilitado), ("empresa espejo", empresa_espejo), ("doble percepción", doble_percepcion), ("puerta giratoria", puerta_giratoria)):
        s = f(a)
        x = x.merge(s, on=K3, how="left")
        paso(f"ETAPA {nombre}: {len(s):,} casos en {a}")
    x = x.merge(de_la_x(a), on=K3, how="left")
    x = x.merge(informes_previos(a), on="ruc", how="left")
    paso("ETAPA misma dirección, proveedor nuevo, procesos e informes previos")
    BANDERAS = {"md_mismo_dueno": "mismo dueño con varios RUC", "sf_sigue_funcionario": "sigue a un funcionario", "vi_vende_inhabilitado": "vende estando inhabilitado",
                "ee_empresa_espejo": "empresa espejo", "dp_doble_percepcion": "doble percepción", "pg_puerta_giratoria": "puerta giratoria",
                "dir_misma_direccion": "misma dirección", "nu_proveedor_nuevo": "proveedor nuevo", "ic_informes_previos": "informes de control previos",
                "oc_procesos": "señales de los procesos"}
    x[list(BANDERAS)] = x[list(BANDERAS)].fillna(False).astype(bool)
    assert len(x) == len(L), "el expediente no tiene una fila por fila de la lista"
    assert hashlib.sha256(ruta.read_bytes()).hexdigest() == huella, "la lista cambió"
    textos = [c for c in x.columns if c.endswith("_texto")]
    assert not x[textos].astype(str).apply(lambda s: s.str.contains(r"\b10\d{9}\b", regex=True)).any().any(), "un texto muestra un RUC de persona natural"
    x["n_senales_expediente"] = x[list(BANDERAS)].sum(axis=1)
    x.drop(columns="motivo").to_parquet(SALIDA_L / f"expediente_{a}.parquet", index=False)
    res = {"anio": a, "lista": ruta.name, "huella_lista_sin_cambios": huella[:16], "filas": len(x),
           "filas_por_senal": {v: int(x[k].sum()) for k, v in BANDERAS.items()},
           "sigue_funcionario_construccion": info,
           "filas_con_alguna": int((x["n_senales_expediente"] > 0).sum()),
           "filas_por_numero_de_senales": x["n_senales_expediente"].value_counts().sort_index().to_dict(),
           "por_motivo": {m: {**{v: int(g[k].sum()) for k, v in BANDERAS.items()}, "filas": int(len(g))} for m, g in x.groupby("motivo")}}
    (SALIDA_L / f"expediente_{a}_resumen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    paso("FIN")


if __name__ == "__main__":
    main()
