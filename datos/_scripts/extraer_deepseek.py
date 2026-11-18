"""Extracción con DeepSeek de las resoluciones del Tribunal (HALLAZGOS 67, paso 5) y de los informes de control
(HALLAZGOS 66, paso 4). Por ahora, solo los pilotos que Pedro revisa a mano antes de correr todo (PENDIENTES, plan del
1/11).

  python -X utf8 extraer_deepseek.py piloto-resoluciones --categorias categorias_impedimento.csv
                                                           # 25 resoluciones de tipos y años distintos
  python -X utf8 extraer_deepseek.py piloto-informes       # los 20 del piloto de Gian Carlo y 5 con vínculo o impedido
  python -X utf8 extraer_deepseek.py informes-fondo --tope 25   # todos los informes de fondo (prioridad 1 a 3) que ya
                                                           # tienen texto; se puede volver a correr: salta los hechos
  --seco: solo arma la muestra y estima el costo, sin llamar a DeepSeek

- La clave se lee del .env de la raíz del repositorio (DEEPSEEK_API_KEY=...), o del archivo que diga la variable
  LLIKA_ENV; no se imprime ni se manda a otro lado.
- deepseek-flash sin razonamiento (thinking: disabled), temperatura 0 y respuesta en JSON. Cada dato importante trae una
  cita textual corta: sirve para revisarlo sin abrir el PDF y para comprobar que el texto existe.
- Los 20 informes del piloto de Gian Carlo (22/10) traen sus órdenes copiadas a mano: son la verdad de campo para medir
  cuántas órdenes encuentra el modelo. Su archivo va en datos/crudos/acc8_informes_cgr/piloto/ (fuera de git) o en la
  ruta que diga la variable LLIKA_PILOTO_GC.
- Salidas en datos/procesados/deepseek/ (fuera de git: traen nombres y DNI).
"""
import argparse
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "datos"
CRUDOS = DATOS / "crudos"
TCE = CRUDOS / "tce_resoluciones"
CGR = CRUDOS / "acc8_informes_cgr"
SAL = DATOS / "procesados" / "deepseek"
PILOTO_GC = Path(os.environ.get("LLIKA_PILOTO_GC") or CGR / "piloto" / "2026-10-22_ordenes_observadas_piloto.csv")
sys.path.insert(0, str(CGR))
sys.path.insert(0, str(Path(__file__).parent))
from clasificar_hechos import PRECIOS, URL, clave_api, horario  # noqa: E402
from tce_resoluciones import normalizar  # noqa: E402

MODELO = "deepseek-flash"
MAX_CARACTERES = 400_000        # unos 110 mil tokens: casi todo entra entero; si no, se corta y se anota
TOPE_USD = 2.0                  # los dos pilotos juntos deberían costar menos de 0,5 dólares

RES_SISTEMA = """Eres abogado especialista en contrataciones del Estado peruano. Lees una resolución del Tribunal de
Contrataciones (TCE o TCP) y extraes sus datos para un proyecto académico de la UTEC que estudia el riesgo en las compras
públicas. Extrae solo lo que dice el texto; si un dato no está, usa null. No inventes RUC, números ni fechas.

Una resolución puede resolver un expediente o varios (resoluciones en serie, con un cuadro de entidades, administrados y
órdenes). Devuelve un elemento en "expedientes" por cada administrado sancionado o investigado. Si el administrado es un
consorcio, va un solo elemento con sus integrantes.

Devuelve solo JSON válido con esta forma:
{"numero": "número de la resolución",
 "fecha_resolucion": "AAAA-MM-DD",
 "sala": "primera, segunda, ... o null",
 "tipo": "sancionador" | "reconsideracion" | "otro",
 "expedientes": [{
   "administrado": {"nombre": "...", "ruc": "11 dígitos o null", "tipo": "persona_natural" | "empresa" | "consorcio"},
   "integrantes_consorcio": [{"nombre": "...", "ruc": "11 dígitos o null", "participacion_pct": número o null}],
   "entidad": {"nombre": "...", "ruc": "11 dígitos o null"},
   "contratacion": {"tipo": "orden_de_compra" | "orden_de_servicio" | "contrato" | "procedimiento_de_seleccion" |
                    "catalogo_electronico" | "rnp" | "otro",
                    "numero": "tal como aparece, o null", "procedimiento": "nomenclatura del procedimiento, o null",
                    "fecha": "AAAA-MM-DD o null", "monto_soles": número o null, "objeto": "en pocas palabras"},
   "infracciones": [{"literal": "letra", "articulo": "por ejemplo 50.1", "ley": "por ejemplo Ley 30225, modificada por
                     el DL 1444", "resumen": "en 15 palabras o menos"}],
   "fecha_hechos": "AAAA-MM-DD: cuándo se cometió la infracción (fecha de la orden o del contrato, de la presentación
                    del documento, o de la resolución del contrato), o null",
   "cita_fecha_hechos": "cita textual de 30 palabras o menos que muestre esa fecha",
   "fecha_denuncia": "AAAA-MM-DD: cuándo la entidad u otro comunicó la infracción al Tribunal, o null",
   "decision": "sanciona" | "no_ha_lugar" | "archiva" | "otro",
   "sancion": {"tipo": "inhabilitacion_temporal" | "inhabilitacion_definitiva" | "multa" | "ninguna",
               "meses": número o null, "multa_soles": número o null},
   "cita_decision": "cita textual de 30 palabras o menos de la parte resolutiva",
   "impedimento": null, o si la infracción es contratar estando impedido:
     {"articulo": "por ejemplo 11.1", "literal": "letra", "causal": "en 15 palabras o menos",
      "persona_vinculada": {"nombre": "...", "cargo": "...", "entidad": "...",
                            "relacion": "es_el_mismo" | "conyuge" | "conviviente" | "padre_o_madre" | "hijo_o_hija" |
                                        "hermano_o_hermana" | "abuelo_o_abuela" | "nieto_o_nieta" | "suegro_o_suegra" |
                                        "yerno_o_nuera" | "cunado_o_cunada" | "socio_o_representante" | "otro",
                            "grado": "por ejemplo segundo de consanguinidad, o null"},
      "cita": "cita textual de 40 palabras o menos"},
   "documento_cuestionado": null, o si es por documento falso o información inexacta:
     {"tipo": "...", "emisor": "...", "falso_o_inexacto": "falso" | "inexacto" | "ambos"},
   "presento_descargos": true | false | null}]}"""

INF_SISTEMA = """Eres auditor gubernamental en el Perú. Lees un informe de control de la Contraloría General de la
República y extraes sus datos para un proyecto académico de la UTEC que prioriza las compras públicas menores (hasta 8
UIT) con riesgo. Extrae solo lo que dice el texto; si un dato no está, usa null. No inventes RUC, DNI, números de orden
ni montos. El texto puede venir de un OCR y traer errores de lectura.

Devuelve solo JSON válido con esta forma:
{"numero": "número del informe",
 "entidad": "...",
 "periodo_auditado": {"desde": "AAAA-MM-DD o null", "hasta": "AAAA-MM-DD o null"},
 "muestra": {"universo_ordenes": número o null, "universo_soles": número o null, "muestra_ordenes": número o null,
             "muestra_soles": número o null, "cita": "cita textual de 30 palabras o menos, o null"},
 "hechos": [{
   "n": 1,
   "sumilla": "en 25 palabras o menos",
   "tipos": ["fraccionamiento" | "elusion_procedimiento" | "indagacion_cotizaciones" | "conformidad_pago" |
             "incumplimiento_penalidad" | "proveedor_impedido" | "vinculo_conflicto" | "sobreprecio" |
             "personal_por_ordenes" | "otro"],
   "fechas_hechos": {"desde": "AAAA-MM-DD o null", "hasta": "AAAA-MM-DD o null"},
   "perjuicio_soles": número o null,
   "normas": ["norma y artículo vulnerados, en pocas palabras"],
   "ordenes": [{"tipo": "O/C" | "O/S" | "contrato" | "otro", "numero": "tal como aparece", "fecha": "AAAA-MM-DD o null",
                "monto_soles": número o null, "proveedor": "nombre o null", "ruc_proveedor": "11 dígitos o null"}],
   "proveedores": [{"nombre": "...", "ruc": "11 dígitos o null", "rol": "beneficiado" | "cotizante" | "postor" | "otro",
                    "cita": "cita textual de 25 palabras o menos"}],
   "personas": [{"nombre": "...", "dni": "8 dígitos o null", "cargo": "...",
                 "rol": "funcionario_o_servidor" | "proveedor" | "familiar" | "otro",
                 "responsabilidad": "administrativa" | "civil" | "penal" | "ninguna" | null}],
   "vinculo": null, o si el hecho describe un vínculo entre un proveedor y la entidad o sus funcionarios:
     {"tipo": "familiar" | "funcionario_proveedor" | "socio_o_representante" | "ex_trabajador" | "otro",
      "descripcion": "en 25 palabras o menos", "cita": "cita textual de 40 palabras o menos"},
   "descargos": {"presentados": true | false | null, "desvirtuan": "si" | "no" | "en_parte" | null}}]}
En "ordenes" van todas las órdenes o contratos que el hecho cita con número, también las de cuadros y apéndices."""


# ------------------------------------------------------------------------------------------------------------ llamada
def llamar(sistema, usuario, clave):
    cuerpo = {"model": MODELO, "thinking": {"type": "disabled"}, "temperature": 0,
              "response_format": {"type": "json_object"}, "max_tokens": 64000,
              "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]}
    ultimo = None
    for intento in range(3):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {clave}"}, json=cuerpo, timeout=900)
            if r.status_code in (429, 500, 502, 503):
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            d = r.json()
            u = d["usage"]
            p = PRECIOS[MODELO][horario()]
            usd = (u.get("prompt_cache_hit_tokens", 0) * p[0] + u.get("prompt_cache_miss_tokens", u["prompt_tokens"]) * p[1]
                   + u["completion_tokens"] * p[2]) / 1e6
            fin = d["choices"][0]["finish_reason"]
            if fin != "stop":
                return {"error": f"finish_reason={fin}", "usd": usd, "tokens_entrada": u["prompt_tokens"],
                        "tokens_salida": u["completion_tokens"]}
            return {"salida": json.loads(d["choices"][0]["message"]["content"]), "usd": usd,
                    "tokens_entrada": u["prompt_tokens"], "tokens_salida": u["completion_tokens"]}
        except Exception as e:      # noqa: BLE001 - se reintenta y, si no, se anota
            ultimo = e
    return {"error": f"{type(ultimo).__name__}: {ultimo}"[:300], "usd": 0.0}


def correr(casos, sistema, archivo, seco=False, hilos=5, tope=None, esperar_barato=True):
    """casos: lista de (clave, texto_usuario). Guarda cada respuesta en cuanto llega; salta las ya hechas."""
    caracteres = sum(len(c[1]) for c in casos) + len(sistema) * len(casos)
    p = PRECIOS[MODELO][horario()]
    est = caracteres / 3.6 / 1e6 * p[1] + len(casos) * 4000 / 1e6 * p[2]   # ~3,6 caracteres por token en español
    print(f"{len(casos)} casos, {caracteres:,} caracteres: unos {caracteres / 3.6 / 1e6:.2f} millones de tokens de "
          f"entrada; costo estimado {est:.2f} dólares (horario {horario()})")
    if seco:
        return
    SAL.mkdir(parents=True, exist_ok=True)
    hechas = set()
    if archivo.exists():
        # solo cuenta como hecho lo que salió bien: lo que dio error se vuelve a intentar
        hechas = {d["clave"] for d in map(json.loads, archivo.open(encoding="utf-8")) if "salida" in d}
    pendientes = [c for c in casos if c[0] not in hechas]
    print(f"Ya hechos: {len(hechas)}; por hacer: {len(pendientes)}", flush=True)
    tope = tope or TOPE_USD
    clave = clave_api()
    gasto = 0.0

    def uno(caso):
        while esperar_barato and horario() == "caro":      # en el horario caro de DeepSeek se espera (salvo pilotos chicos)
            time.sleep(600)
        return caso[0], llamar(sistema, caso[1], clave)

    with ThreadPoolExecutor(hilos) as ex, archivo.open("a", encoding="utf-8") as f:
        for k, (c, res) in enumerate(ex.map(uno, pendientes), 1):
            gasto += res.get("usd", 0.0)
            f.write(json.dumps({"clave": c, **res}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {k}/{len(pendientes)} {c}: {'error ' + res['error'] if 'error' in res else 'ok'}; "
                  f"{gasto:.3f} dólares", flush=True)
            if gasto > tope:
                sys.exit(f"Tope de {tope} dólares alcanzado")
    print(f"Listo: {len(pendientes)} llamadas, {gasto:.3f} dólares (horario {horario()})")


def recortar(texto):
    t = re.sub(r"[ \t]+", " ", texto)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return (t[:MAX_CARACTERES] + "\n[TEXTO CORTADO]") if len(t) > MAX_CARACTERES else t


# -------------------------------------------------------------------------------------------- piloto de resoluciones
def sin_trib(c):
    return re.sub(r"-T[A-Z]+", "", c) if isinstance(c, str) else None


def muestra_resoluciones(categorias):
    """25 resoluciones de tipos y años distintos. Semilla fija: la muestra se puede rehacer igual."""
    s = pd.read_parquet(DATOS / "procesados" / "radar" / "sanciones.parquet")
    s = s[s["inicio"] >= "2018-01-01"].copy()
    s["n2"] = s["resolucion"].map(lambda r: sin_trib(normalizar(r)))
    ind = {}
    with (TCE / "textos.jsonl").open(encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            ind[sin_trib(normalizar(d["numero"]))] = (d["numero"], d["anio"], d["resumen"] or "", d["paginas"])
    s = s[s["n2"].isin(ind)]
    s["resumen"] = s["n2"].map(lambda k: ind[k][2])
    m = s["motivo"].str.upper()
    rucs = s.groupby("n2")["ruc"].nunique()
    consorcio = s["resumen"].str.contains("CONSORCIO", case=False)
    elegidos = []

    def tomar(cand, n, etiqueta, por_anio=False):
        cand = cand[~cand["n2"].isin([e[0] for e in elegidos])].drop_duplicates("n2")
        if por_anio:   # uno por año distinto, del más viejo al más nuevo
            cand = cand.assign(a=cand["inicio"].dt.year).sample(frac=1, random_state=7).drop_duplicates("a")
        for k in cand.sample(min(n, len(cand)), random_state=7)["n2"]:
            elegidos.append((k, etiqueta))

    cat = pd.read_csv(categorias)      # su n2 ya viene normalizado y sin el tribunal (tipos_impedimento.py)
    for c, n in (("directo d", 2), ("directo c", 1), ("pariente de d", 3), ("pariente de c", 1),
                 ("empresa o representante de d", 1), ("pariente de e", 1), ("pariente de a", 1)):
        tomar(s[s["n2"].isin(cat.loc[cat["categoria"] == c, "n2"])], n, f"impedimento: {c}")
    tomar(s[m.str.contains("FALS|INEXACT") & ~consorcio], 5, "documento falso o información inexacta", por_anio=True)
    tomar(s[m.str.contains("RESUELVA EL CONTRATO|RESOLUCION DEL CONTRATO|RESCISION") & ~consorcio], 4,
          "resolución del contrato", por_anio=True)
    tomar(s[s["n2"].isin(rucs[rucs >= 3].index) & ~consorcio], 2, "en serie (3 o más administrados)")
    tomar(s[consorcio], 2, "consorcio")
    tomar(s[s["definitiva"]], 1, "inhabilitación definitiva")
    tomar(s, 25 - len(elegidos), "al azar")
    return pd.DataFrame(elegidos, columns=["n2", "estrato"]).assign(
        numero=lambda x: x["n2"].map(lambda k: ind[k][0]), anio=lambda x: x["n2"].map(lambda k: ind[k][1]),
        paginas=lambda x: x["n2"].map(lambda k: ind[k][3]))


def piloto_resoluciones(categorias, seco):
    mu = muestra_resoluciones(categorias)
    SAL.mkdir(parents=True, exist_ok=True)
    mu.to_csv(SAL / "piloto_resoluciones_muestra.csv", index=False)
    print(mu["estrato"].value_counts().to_string())
    buscados = set(mu["numero"])
    casos = []
    with (TCE / "textos.jsonl").open(encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            if d["numero"] in buscados:
                casos.append((d["numero"], f"Resolución {d['numero']} ({d['anio']}). Texto:\n{recortar(d['texto'])}"))
    correr(casos, RES_SISTEMA, SAL / "piloto_resoluciones.jsonl", seco)


# ------------------------------------------------------------------------------------------------ piloto de informes
def textos_disponibles():
    """CresCodigoFormato -> (origen, ruta o (zip, nombre)) de todos los informes ya leídos."""
    out = {}
    # el OCR local puede ir en dos partes, cada una con su resumen (resumen.csv y resumen_p1.csv)
    loc = pd.concat([pd.read_csv(r, dtype=str) for r in sorted((CGR / "textos_local").glob("resumen*.csv"))])
    for r in loc[loc["estado"] == "ok"].itertuples():
        out[r.CresCodigoFormato] = ("local", CGR / "textos_local" / r.Periodo / r.archivo)
    for z in (RAIZ / "ocr_informes" / "resultados").glob("lote_*/textos_lote_*.zip"):
        with zipfile.ZipFile(z) as zz:
            nombres = set(zz.namelist())
            for res in [n for n in nombres if re.search(r"resumen_\d+\.csv$", n)]:
                with zz.open(res) as fr:
                    t = pd.read_csv(fr, dtype=str)
                for r in t[t["estado"] == "ok"].itertuples():
                    cand = [n for n in nombres if n.endswith("/" + r.archivo) or n == r.archivo]
                    if cand and r.CresCodigoFormato not in out:
                        out[r.CresCodigoFormato] = ("kaggle", (z, cand[0]))
    return out


def leer_texto(ref):
    origen, x = ref
    if origen == "local":
        return Path(x).read_text(encoding="utf-8", errors="replace")
    with zipfile.ZipFile(x[0]) as zz:
        return zz.read(x[1]).decode("utf-8", "replace")


def cres(enlace):
    """CRES_CODIGO=2026CPO082800030 -> 2026-CPO-0828-00030 (el formato de las listas)."""
    c = re.search(r"CRES_CODIGO=(\w+)", enlace).group(1)
    return f"{c[:4]}-{c[4:7]}-{c[7:11]}-{c[11:]}"


def muestra_informes():
    if not PILOTO_GC.exists():
        sys.exit(f"Falta {PILOTO_GC}, el piloto de órdenes copiadas a mano; indica otra ruta con la variable LLIKA_PILOTO_GC")
    gc = pd.read_csv(PILOTO_GC, dtype=str)
    gc["CresCodigoFormato"] = gc["enlace_pdf"].map(cres)
    disp = textos_disponibles()
    filas = [(c, "piloto de Gian Carlo (órdenes copiadas a mano)") for c in gc["CresCodigoFormato"].unique()]
    h = pd.read_csv(CGR / "fichas" / "clasificacion" / "hechos_clasificados.csv", dtype=str)
    h = h[h["CresCodigoFormato"].isin(disp) & ~h["CresCodigoFormato"].isin([f[0] for f in filas])]
    for v, n in (("familiar", 2), ("funcionario_proveedor", 1), ("socio_o_representante", 1)):
        for c in h[h["vinculo"] == v].drop_duplicates("CresCodigoFormato").sample(n, random_state=7)["CresCodigoFormato"]:
            filas.append((c, f"vínculo: {v}"))
    imp = h[h["tipos"].str.contains("proveedor_impedido", na=False) & ~h["CresCodigoFormato"].isin([f[0] for f in filas])]
    for c in imp.drop_duplicates("CresCodigoFormato").sample(1, random_state=7)["CresCodigoFormato"]:
        filas.append((c, "proveedor impedido"))
    mu = pd.DataFrame(filas, columns=["CresCodigoFormato", "estrato"])
    mu["texto_disponible"] = mu["CresCodigoFormato"].isin(disp)
    return mu, disp


def piloto_informes(seco):
    SAL.mkdir(parents=True, exist_ok=True)
    mu, disp = muestra_informes()
    mu.to_csv(SAL / "piloto_informes_muestra.csv", index=False)
    print(mu.groupby("estrato")["texto_disponible"].agg(["size", "sum"]).to_string())
    casos = [(c, f"Informe {c}. Texto:\n{recortar(leer_texto(disp[c]))}")
             for c in mu.loc[mu["texto_disponible"], "CresCodigoFormato"]]
    correr(casos, INF_SISTEMA, SAL / "piloto_informes.jsonl", seco)



def informes_fondo(seco, tope):
    """Todos los informes de fondo con texto. Los 25 del piloto (mismas instrucciones) se copian para no pagarlos dos veces."""
    destino = SAL / "informes_fondo.jsonl"
    SAL.mkdir(parents=True, exist_ok=True)
    # candado: si otra corrida está en marcha, esta no arranca (dos a la vez pagarían dos veces lo mismo)
    candado = SAL / "informes_fondo.lock"
    try:
        candado.touch(exist_ok=False)
    except FileExistsError:
        sys.exit(f"Hay otra corrida en marcha ({candado}); si no la hay, borrar el candado.")
    try:
        _informes_fondo(destino, seco, tope)
    finally:
        candado.unlink(missing_ok=True)


def _informes_fondo(destino, seco, tope):
    if not destino.exists() and (SAL / "piloto_informes.jsonl").exists():
        destino.write_text((SAL / "piloto_informes.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    disp = textos_disponibles()
    prio = pd.read_csv(RAIZ / "ocr_informes" / "prioridad_informes.csv", dtype=str)
    fondo = [c for c in prio.loc[prio["prioridad"].astype(int) <= 3, "CresCodigoFormato"] if c in disp]
    print(f"Informes de fondo: {(prio['prioridad'].astype(int) <= 3).sum()}; con texto: {len(fondo)}")
    hechas = ({d["clave"] for d in map(json.loads, destino.open(encoding="utf-8")) if "salida" in d}
              if destino.exists() else set())
    casos = [(c, f"Informe {c}. Texto:\n{recortar(leer_texto(disp[c]))}") for c in fondo if c not in hechas]
    correr(casos, INF_SISTEMA, destino, seco, tope=tope)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["piloto-resoluciones", "piloto-informes", "informes-fondo"])
    ap.add_argument("--categorias", default=None, help="categorias_impedimento.csv (tipo de impedimento por resolución)")
    ap.add_argument("--seco", action="store_true", help="solo arma la muestra y estima el costo")
    ap.add_argument("--tope", type=float, default=TOPE_USD, help="dólares; se detiene al pasarlo")
    a = ap.parse_args()
    if a.paso == "piloto-resoluciones":
        piloto_resoluciones(a.categorias, a.seco)
    elif a.paso == "piloto-informes":
        piloto_informes(a.seco)
    else:
        informes_fondo(a.seco, a.tope)
