"""Paso 2 de HALLAZGOS 66: clasificar los 16 417 hechos de las fichas resumen.

  python -X utf8 clasificar_hechos.py plantillas   # gratis: aplica las etiquetas puestas a mano a las familias de plantilla
  python -X utf8 clasificar_hechos.py prueba       # 100 hechos con deepseek-flash y con deepseek-v4-pro, y Excel para revisar
  python -X utf8 clasificar_hechos.py correr --modelo deepseek-flash   # el resto, con tope de gasto
  python -X utf8 clasificar_hechos.py informes     # tipos y prioridad por informe -> prioridad_informes.csv

La clave se lee del .env de la raíz del repositorio (DEEPSEEK_API_KEY=...), o del archivo que diga la variable LLIKA_ENV;
no se imprime ni se manda a otro lado. Al LLM solo va el
texto del hecho con el título, la modalidad y el año del informe (no la tabla de personas: no hace falta para clasificar).
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

AQUI = Path(__file__).parent
F = AQUI / "fichas"
C = F / "clasificacion"
ENV = Path(os.environ.get("LLIKA_ENV") or Path(__file__).resolve().parents[3] / ".env")   # git no sube el .env
URL = "https://api.deepseek.com/chat/completions"
# Precios por millón de tokens (api-docs.deepseek.com/quick_start/pricing, 31/10/2026): (acierto de caché, entrada, salida)
PRECIOS = {"deepseek-flash": {"barato": (0.003, 0.15, 0.6), "caro": (0.006, 0.3, 1.2)},
           "deepseek-v4-pro": {"barato": (0.022, 0.66, 1.98), "caro": (0.044, 1.32, 3.96)}}
TAM_TANDA = 20

TIPOS = {
    "fraccionamiento": "dividir una misma necesidad en varias compras u órdenes para quedar bajo el tope y evitar el "
                       "procedimiento de selección",
    "elusion_procedimiento": "contratar por montos de hasta 8 UIT o de forma directa cuando correspondía un procedimiento "
                             "de selección, o contratación directa sin causal válida",
    "indagacion_cotizaciones": "requerimiento, términos de referencia o indagación de mercado deficientes; cotizaciones "
                               "falsas, inconsistentes o de proveedores sin relación con el objeto",
    "conformidad_pago": "conformidad o pago sin que se entregue el bien o se preste el servicio, sin entregables, "
                        "pagos indebidos o duplicados",
    "incumplimiento_penalidad": "incumplimiento contractual del proveedor, penalidades no aplicadas, garantías",
    "proveedor_impedido": "contratar a un proveedor impedido, sancionado, inhabilitado o sin inscripción vigente en el RNP",
    "vinculo_conflicto": "parentesco, nepotismo, funcionario o servidor contratado como proveedor de su entidad, ex "
                         "trabajador, socio o conflicto de intereses",
    "sobreprecio": "precios mayores a los de mercado o perjuicio por diferencia de precios",
    "personal_por_ordenes": "contratar personal o funciones permanentes mediante órdenes de servicio o locación; doble "
                            "percepción",
    "presupuesto_certificacion": "contratar o comprometer sin certificación presupuestal o fuera del PAC o del PDP",
    "control_interno": "falta de directivas o normativa interna, expedientes incompletos, archivo, fiscalización "
                       "posterior no realizada",
    "publicacion_registro": "no registrar o no publicar (o registrar fuera de plazo) órdenes o contratos en el SEACE, la "
                            "PLADICOP, el SIRICC u otra plataforma",
    "acuerdo_marco_gestion": "gestión en los catálogos electrónicos de acuerdos marco de Perú Compras: proformas sin "
                             "formalizar, estados de órdenes sin actualizar",
    "obras": "ejecución, supervisión o expedientes de obras",
    "otro_compras": "otra irregularidad en contrataciones que no entra en los tipos anteriores",
    "ajeno": "no trata de contrataciones públicas (ordenanzas, tributos, personal bajo concurso, gestión ambiental, etc.)",
    "sin_observaciones": "el texto dice que no se identificaron observaciones o situaciones adversas",
}
VINCULOS = ["ninguno", "familiar", "funcionario_proveedor", "socio_o_representante", "ex_trabajador", "otro"]

INSTRUCCIONES = f"""Eres analista de control gubernamental en el Perú. Clasificas hechos (o situaciones adversas) tomados de
las fichas resumen de informes de control de la Contraloría, para un proyecto académico que prioriza compras públicas
menores (hasta 8 UIT) con riesgo. Cada hecho trae el título del informe, la modalidad del servicio de control y el año.

Tipos posibles (código: significado):
{chr(10).join(f"- {k}: {v}" for k, v in TIPOS.items())}

Para cada hecho devuelve un objeto con:
- "id": el mismo id que recibiste.
- "tipos": lista de 1 a 3 códigos de la lista, el más importante primero.
- "nivel": "indicio_irregularidad" (el informe afirma una irregularidad ocurrida), "riesgo" (situación adversa que
  podría afectar, típica del control simultáneo) o "ninguno" (sin observaciones, o el texto no describe un hecho).
- "nombra_ordenes": true si el texto cita números de órdenes, contratos o procesos concretos (no basta decir
  «19 órdenes»).
- "nombra_proveedor": true si el texto da el nombre o el RUC de un proveedor.
- "vinculo": uno de {VINCULOS}.
- "perjuicio": true si menciona perjuicio económico; "monto_perjuicio": el monto en soles como número, o null.
- "monto_observado": el monto total observado en soles como número, o null.
- "anios_hechos": lista de años en que ocurrieron los hechos (no el año del informe), o [].
- "objeto": qué se compró o contrató, en 1 a 5 palabras (por ejemplo «combustible», «servicio de impresión»), o "".
- "relevancia_compras_menores": "alta" (compras u órdenes de hasta 8 UIT o contrataciones menores), "media" (otras
  contrataciones de bienes y servicios), "baja" (contrataciones grandes, obras) o "nula" (ajeno).
- "confianza": "alta", "media" o "baja", según lo claro que sea el texto.
- "por_que": una frase de 20 palabras o menos que justifique los tipos.

Si el texto no describe el hecho (por ejemplo, solo trae conclusiones o recomendaciones), usa el título para el tipo,
pon "nivel": "ninguno" y "confianza": "baja".
Responde solo con JSON válido de la forma {{"hechos": [ ... ]}}, con un objeto por cada hecho recibido, en el mismo orden."""

# --------------------------------------------------------------------------------------------- etiquetas de plantilla
# Revisadas a mano leyendo los ejemplos de cada familia (familias.csv). Las familias que no están aquí (con hechos de
# fondo, o textos que no describen el hecho) van al LLM.
PUB = [1, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 26, 27, 28, 29, 32, 33, 37, 38, 40, 41, 42, 43,
       45, 46, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 60, 62, 64, 65, 67, 68, 70, 71, 74, 75, 76, 82, 83, 84, 85, 86, 88,
       90]
AM = [2, 31, 36, 39, 44, 63, 66, 69, 80, 87]
AJENO = [3, 4, 8, 11, 23, 34, 59, 89]
SIN_OBS = [35, 47, 77, 79, 92]
ETIQUETA_FAMILIA = {**{f: ("publicacion_registro", "media") for f in PUB},
                    **{f: ("acuerdo_marco_gestion", "baja") for f in AM},
                    **{f: ("ajeno", "nula") for f in AJENO},
                    **{f: ("sin_observaciones", "nula") for f in SIN_OBS}}
NUM_ORDEN = re.compile(r"(?i)(?:orden(?:es)? de (?:compra|servicio)s?|o/[cs])\s*(?:n[°º.]*|nro\.?|n\. ?°)\s*[\d-]{2,}")


def clave_api():
    if not ENV.exists():
        sys.exit(f"Falta {ENV}: créalo con la línea DEEPSEEK_API_KEY=... o indica otro archivo con la variable LLIKA_ENV")
    for linea in ENV.read_text(encoding="utf-8").splitlines():
        if linea.startswith("DEEPSEEK_API_KEY"):
            return linea.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No encontré DEEPSEEK_API_KEY en el .env")


def horario():
    ahora = datetime.now(timezone.utc)
    caro = ahora.weekday() < 5 and (1 <= ahora.hour < 4 or 6 <= ahora.hour < 10)
    return "caro" if caro else "barato"


def hechos_base():
    h = pd.read_csv(F / "fichas_hechos.csv", dtype=str)
    h["id_hecho"] = h.CresCodigoFormato + "#" + h.n
    i = pd.read_csv(F / "fichas_informes.csv", dtype=str, low_memory=False)
    h = h.merge(i[["CresCodigoFormato", "tipo_ficha", "modalidad_marcada", "titulo", "Periodo"]], on="CresCodigoFormato",
                how="left")
    fam = pd.read_csv(C / "hechos_familia.csv", dtype={"id_hecho": str})
    return h.merge(fam[["id_hecho", "fam_id"]], on="id_hecho", how="left")


def plantillas():
    h = hechos_base()
    h["fam_id"] = pd.to_numeric(h.fam_id, errors="coerce")
    con = h[h.fam_id.isin(ETIQUETA_FAMILIA)].copy()
    con["tipo_principal"] = con.fam_id.map(lambda f: ETIQUETA_FAMILIA[int(f)][0])
    con["tipos"] = con.tipo_principal
    con["relevancia_compras_menores"] = con.fam_id.map(lambda f: ETIQUETA_FAMILIA[int(f)][1])
    con["nivel"] = con.apply(lambda r: "ninguno" if r.tipo_principal == "sin_observaciones" else
                             ("indicio_irregularidad" if r.tipo_ficha == "posterior" else "riesgo"), axis=1)
    con["nombra_ordenes"] = con.texto.fillna("").str.contains(NUM_ORDEN)
    con["nombra_proveedor"] = False
    con["vinculo"] = "ninguno"
    con["perjuicio"] = False
    con["monto_observado"] = pd.to_numeric(con.monto_max, errors="coerce")
    con["confianza"] = "alta"
    con["por_que"] = "plantilla (familia " + con.fam_id.astype(int).astype(str) + ")"
    con["metodo"] = "plantilla"
    cols = ["id_hecho", "CresCodigoFormato", "tipos", "tipo_principal", "nivel", "nombra_ordenes", "nombra_proveedor",
            "vinculo", "perjuicio", "monto_observado", "relevancia_compras_menores", "confianza", "por_que", "metodo"]
    con[cols].to_csv(C / "clasificados_plantilla.csv", index=False)
    resto = h[~h.id_hecho.isin(con.id_hecho)]
    resto[["id_hecho", "CresCodigoFormato", "tipo_ficha", "modalidad_marcada", "titulo", "Periodo", "texto"]].to_csv(
        C / "pendientes_llm.csv", index=False)
    print(f"Plantilla: {len(con)} hechos clasificados gratis; quedan para el LLM: {len(resto)}")
    print(con.tipo_principal.value_counts().to_string())


# ------------------------------------------------------------------------------------------------------------- LLM
gasto = {"usd": 0.0, "llamadas": 0}
cerrojo = threading.Lock()


def llamar(modelo, tanda, clave, tope):
    lineas = []
    for r in tanda.itertuples():
        texto = re.sub(r"\s+", " ", str(r.texto))[:4000]
        lineas.append(json.dumps({"id": r.id_hecho, "titulo_informe": str(r.titulo)[:400],
                                  "modalidad": r.modalidad_marcada, "anio_informe": r.Periodo, "texto": texto},
                                 ensure_ascii=False))
    cuerpo = {"model": modelo, "thinking": {"type": "disabled"}, "temperature": 0,
              "response_format": {"type": "json_object"}, "max_tokens": 8000,
              "messages": [{"role": "system", "content": INSTRUCCIONES},
                           {"role": "user", "content": "Hechos:\n" + "\n".join(lineas)}]}
    for intento in range(4):
        with cerrojo:
            if gasto["usd"] >= tope:
                raise RuntimeError(f"Tope de gasto alcanzado ({gasto['usd']:.2f} dólares)")
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {clave}"}, json=cuerpo, timeout=300)
            if r.status_code in (429, 500, 502, 503):
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            d = r.json()
            u = d["usage"]
            p = PRECIOS[modelo][horario()]
            usd = (u.get("prompt_cache_hit_tokens", 0) * p[0] + u.get("prompt_cache_miss_tokens", u["prompt_tokens"]) * p[1]
                   + u["completion_tokens"] * p[2]) / 1e6
            with cerrojo:
                gasto["usd"] += usd
                gasto["llamadas"] += 1
            if d["choices"][0]["finish_reason"] != "stop":
                raise ValueError(f"finish_reason={d['choices'][0]['finish_reason']}")
            salida = json.loads(d["choices"][0]["message"]["content"])["hechos"]
            ids = {o.get("id") for o in salida}
            if not set(tanda.id_hecho) <= ids:
                raise ValueError(f"faltan ids: {set(tanda.id_hecho) - ids}")
            return [{**o, "modelo": modelo} for o in salida if o.get("id") in set(tanda.id_hecho)]
        except RuntimeError:
            raise
        except Exception as e:      # noqa: BLE001 - se reintenta
            ultimo = e
            time.sleep(5 * (intento + 1))
    print(f"  tanda fallida tras 4 intentos: {ultimo}", flush=True)
    return [{"id": i, "modelo": modelo, "error": str(ultimo)[:200]} for i in tanda.id_hecho]


def correr_llm(df, modelo, tope, hilos=6, archivo=None, pausar_en_caro=True):
    clave = clave_api()
    tandas = [df.iloc[i:i + TAM_TANDA] for i in range(0, len(df), TAM_TANDA)]
    out, t0 = [], time.time()
    with ThreadPoolExecutor(hilos) as ex:
        futuros = []
        for t in tandas:
            while pausar_en_caro and horario() == "caro":
                print("Horario caro de DeepSeek: espero 10 minutos", flush=True)
                time.sleep(600)
            futuros.append(ex.submit(llamar, modelo, t, clave, tope))
        for k, fu in enumerate(as_completed(futuros), 1):
            res = fu.result()
            out.extend(res)
            if archivo is not None:
                with open(archivo, "a", encoding="utf-8") as fa:
                    for o in res:
                        fa.write(json.dumps(o, ensure_ascii=False) + "\n")
            if k % 20 == 0 or k == len(futuros):
                print(f"  {k}/{len(futuros)} tandas, {gasto['usd']:.3f} dólares, {(time.time() - t0) / 60:.1f} min",
                      flush=True)
    return pd.DataFrame(out)


def prueba():
    p = pd.read_csv(C / "pendientes_llm.csv", dtype=str)
    t = p.texto.fillna("").str.lower()
    estratos = {"fraccion": r"fraccion", "vinculo": r"parentesco|familiar|conflicto de inter|nepotismo|impedid",
                "pago": r"conformidad|pago|no ejecut", "penalidad": r"penalidad", "cotizacion": r"cotizaci|indagaci",
                "precio": r"sobreprecio|precio"}
    elegidos = []
    for nombre, patron in estratos.items():
        cand = p[t.str.contains(patron, regex=True) & ~p.id_hecho.isin(elegidos)]
        elegidos += cand.sample(min(10, len(cand)), random_state=7).id_hecho.tolist()
    for tipo in ("posterior", "simultaneo"):
        cand = p[(p.tipo_ficha == tipo) & ~p.id_hecho.isin(elegidos)]
        elegidos += cand.sample(20, random_state=7).id_hecho.tolist()
    m = p[p.id_hecho.isin(elegidos)].reset_index(drop=True)
    print(f"Prueba con {len(m)} hechos")
    res = {}
    for modelo in ("deepseek-flash", "deepseek-v4-pro"):
        antes = gasto["usd"]
        res[modelo] = correr_llm(m, modelo, tope=1.0, hilos=5, pausar_en_caro=False)
        print(f"{modelo}: {gasto['usd'] - antes:.4f} dólares")
    a, b = res["deepseek-flash"].set_index("id"), res["deepseek-v4-pro"].set_index("id")
    filas = []
    for r in m.itertuples():
        fila = {"id_hecho": r.id_hecho, "modalidad": r.modalidad_marcada, "año": r.Periodo, "título": r.titulo,
                "hecho": r.texto}
        for etiqueta, d in (("flash", a), ("pro", b)):
            o = d.loc[r.id_hecho] if r.id_hecho in d.index else {}
            for campo in ("tipos", "nivel", "nombra_ordenes", "nombra_proveedor", "vinculo", "perjuicio",
                          "monto_perjuicio", "relevancia_compras_menores", "confianza", "por_que"):
                v = o.get(campo) if hasattr(o, "get") else None
                fila[f"{etiqueta}: {campo}"] = ", ".join(v) if isinstance(v, list) else v
        fila["¿coinciden en el tipo principal?"] = (str(fila["flash: tipos"]).split(",")[0].strip()
                                                    == str(fila["pro: tipos"]).split(",")[0].strip())
        filas.append(fila)
    out = pd.DataFrame(filas)
    out.to_csv(C / "prueba_100.csv", index=False)
    print("Coinciden en el tipo principal:", f"{out['¿coinciden en el tipo principal?'].mean():.0%}")
    print(f"Gasto total de la prueba: {gasto['usd']:.4f} dólares")


def correr(modelo, tope):
    p = pd.read_csv(C / "pendientes_llm.csv", dtype=str)
    archivo = C / f"llm_{modelo}.jsonl"
    hechos = set()
    if archivo.exists():
        for l in open(archivo, encoding="utf-8"):
            o = json.loads(l)
            if "error" not in o:
                hechos.add(o["id"])
    pend = p[~p.id_hecho.isin(hechos)]
    print(f"Por clasificar con {modelo}: {len(pend)} (ya hechos: {len(hechos)}); tope {tope} dólares; horario {horario()}")
    correr_llm(pend, modelo, tope, archivo=archivo)
    print(f"Gasto: {gasto['usd']:.3f} dólares en {gasto['llamadas']} llamadas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["plantillas", "prueba", "correr", "informes"])
    ap.add_argument("--modelo", default="deepseek-flash")
    ap.add_argument("--tope", type=float, default=5.0)
    a = ap.parse_args()
    C.mkdir(exist_ok=True)
    if a.paso != "informes":
        {"plantillas": plantillas, "prueba": prueba}.get(a.paso, lambda: correr(a.modelo, a.tope))()


# ------------------------------------------------------------------------------------- consolidación y prioridad
FONDO_FUERTE = {"fraccionamiento", "elusion_procedimiento", "indagacion_cotizaciones", "conformidad_pago",
                "incumplimiento_penalidad", "proveedor_impedido", "vinculo_conflicto", "sobreprecio",
                "personal_por_ordenes", "otro_compras", "presupuesto_certificacion"}
FONDO_DEBIL = {"control_interno", "obras"}
RUTINA = {"publicacion_registro", "acuerdo_marco_gestion"}
# Número concreto de orden, contrato o proceso en el texto («N° 452», «n.° 003-2020», «2018-00000850»). El LLM a veces
# marca nombra_ordenes con solo «cinco órdenes»; en la revisión de 50 hechos acertó 44. Se exige además esta regla.
CITA_NUMERO = re.compile(r"(?i)(?:\bn\s?[.°º]+\s?°?|\bnro\.?|\bnúmero)\s*[\d]|\b\d{1,5}-20\d{2}\b|\b20\d{2}-\d{4,}\b")


def informes():
    pl = pd.read_csv(C / "clasificados_plantilla.csv", dtype=str)
    pl["tipos"] = pl.tipos.map(lambda t: [t])
    llm = pd.DataFrame([json.loads(l) for l in open(C / "llm_deepseek-flash.jsonl", encoding="utf-8")])
    esc = C / "llm_pro_escalados.jsonl"
    if esc.exists():                               # los de confianza baja, repasados con v4-pro, reemplazan a flash
        pro = pd.read_json(esc, lines=True)
        pro = pro[pro.get("error").isna()] if "error" in pro else pro
        llm = pd.concat([llm[~llm.id.isin(pro.id)], pro])
    llm = llm.rename(columns={"id": "id_hecho"})
    llm["metodo"] = llm.modelo
    llm["nivel"] = llm.nivel.replace({"ningo": "ninguno"})
    llm["CresCodigoFormato"] = llm.id_hecho.str.rsplit("#", n=1).str[0]
    todos = pd.concat([pl, llm], ignore_index=True)
    todos["tipo_principal"] = todos.tipos.map(lambda t: t[0] if isinstance(t, list) and t else None)
    h = pd.read_csv(F / "fichas_hechos.csv", dtype=str)
    h["id_hecho"] = h.CresCodigoFormato + "#" + h.n
    todos = todos.merge(h[["id_hecho", "texto"]], on="id_hecho", how="left")
    cita = todos.texto.fillna("").str.contains(CITA_NUMERO)
    todos["cita_ordenes"] = todos.nombra_ordenes.astype(str).str.lower().eq("true") & cita
    for c in ("nombra_proveedor", "perjuicio"):
        todos[c] = todos[c].astype(str).str.lower().eq("true")
    cols_h = ["id_hecho", "CresCodigoFormato", "tipo_principal", "tipos", "nivel", "cita_ordenes", "nombra_proveedor",
              "vinculo", "perjuicio", "monto_perjuicio", "monto_observado", "anios_hechos", "objeto",
              "relevancia_compras_menores", "confianza", "por_que", "metodo"]
    out_h = todos.reindex(columns=cols_h).copy()
    out_h["tipos"] = out_h.tipos.map(lambda t: ";".join(t) if isinstance(t, list) else t)
    out_h["anios_hechos"] = out_h.anios_hechos.map(lambda t: ";".join(map(str, t)) if isinstance(t, list) else t)
    out_h.to_csv(C / "hechos_clasificados.csv", index=False)

    inf = pd.read_csv(F / "fichas_informes.csv", dtype=str, low_memory=False)
    filas = []
    for cod, g in todos.groupby("CresCodigoFormato"):
        tipos = {t for ts in g.tipos for t in (ts if isinstance(ts, list) else [])}
        filas.append({"CresCodigoFormato": cod, "tipos": ";".join(sorted(tipos)),
                      "fondo_fuerte": bool(tipos & FONDO_FUERTE), "fondo_debil": bool(tipos & FONDO_DEBIL),
                      "rutina": bool(tipos & RUTINA), "ajeno_o_sin_obs": tipos <= {"ajeno", "sin_observaciones"},
                      "cita_ordenes": bool(g.cita_ordenes.any()), "nombra_proveedor": bool(g.nombra_proveedor.any()),
                      "vinculo": ";".join(sorted(set(g.vinculo.dropna()) - {"ninguno"})),
                      "perjuicio": bool(g.perjuicio.any()),
                      "relevancia_max": min(g.relevancia_compras_menores.dropna(),
                                            key=["alta", "media", "baja", "nula"].index, default=None)})
    r = pd.DataFrame(filas).merge(inf[["CresCodigoFormato", "Periodo", "tipo_ficha", "modalidad_marcada", "titulo",
                                       "entidad", "n_personas"]], on="CresCodigoFormato", how="right")
    r["n_personas"] = pd.to_numeric(r.n_personas, errors="coerce").fillna(0)
    posterior = r.tipo_ficha.eq("posterior")
    senal = (r.n_personas > 0) | r.perjuicio.eq(True) | r.cita_ordenes.eq(True) | r.nombra_proveedor.eq(True) \
        | r.vinculo.fillna("").ne("")
    r["prioridad"] = 5
    r.loc[r.rutina.eq(True), "prioridad"] = 4
    r.loc[r.fondo_debil.eq(True) | (r.fondo_fuerte.eq(True) & ~posterior), "prioridad"] = 3
    r.loc[r.fondo_fuerte.eq(True) & posterior, "prioridad"] = 2
    r.loc[r.fondo_fuerte.eq(True) & posterior & senal, "prioridad"] = 1
    r.to_csv(C / "informes_clasificados.csv", index=False)
    destino = AQUI.parents[2] / "ocr_informes" / "prioridad_informes.csv"
    rel = r.relevancia_max.map({"alta": 0, "media": 1, "baja": 2, "nula": 3}).fillna(4)
    r.assign(rel=rel).sort_values(["prioridad", "rel", "Periodo"], ascending=[True, True, False])[
        ["CresCodigoFormato", "prioridad"]].to_csv(destino, index=False)
    print(f"Hechos: {len(out_h)} | informes: {len(r)}")
    print(r.prioridad.value_counts().sort_index().to_string())
    print("Prioridad escrita en", destino)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "informes":
    informes()
