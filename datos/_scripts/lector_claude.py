"""Lector de resoluciones del Tribunal e informes de control con Claude Haiku 5.5 por la API de Mensajes en lote (Batches,
a mitad de precio). Reemplaza a DeepSeek para la corrida completa si el piloto sale igual o mejor (HALLAZGOS 71).

  python -X utf8 lector_claude.py estimar  --fuente resoluciones --conjunto piloto
  python -X utf8 lector_claude.py prueba   --fuente resoluciones          # 1 documento, sin lote, para probar el esquema
  python -X utf8 lector_claude.py enviar   --fuente informes --conjunto piloto --tope 1 --enviar
  python -X utf8 lector_claude.py estado                                  # lotes enviados y su avance
  python -X utf8 lector_claude.py bajar                                   # trae los resultados de los lotes terminados

Cuidados para no gastar de más ni correr dos veces:
- «enviar» sin --enviar solo estima. Con --enviar, se niega si la estimación pasa el --tope (en dólares).
- Nunca manda un documento que ya tiene resultado o que está en un lote abierto (registro en registro_lotes.jsonl).
- El texto se corta para que la entrada quede bajo 100 mil tokens: arriba de eso Haiku 5.5 cobra cinco veces más.
- Sin razonamiento (thinking disabled) y con salida en JSON según el esquema: no se paga por pensar ni por reintentar
  JSON roto.
La clave se lee de D:/UTEC/Maestria/.env (ANTHROPIC_API_KEY_LLIKA); no se imprime ni se manda a otro lado. Salidas en
datos/procesados/claude/ (fuera de git: traen nombres y DNI).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd

from extraer_deepseek import SAL as SAL_DS, TCE, leer_texto, muestra_informes, textos_disponibles

DATOS = Path(__file__).resolve().parents[1]
SAL = DATOS / "procesados" / "claude"
REGISTRO = SAL / "registro_lotes.jsonl"
ENV = Path(r"D:/UTEC/Maestria/.env")
MODELO = "claude-haiku-5-5"
# dólares por millón de tokens, con el lote a mitad de precio; prompt de hasta 100 mil tokens (documentación de Anthropic)
PRECIO_ENTRADA, PRECIO_SALIDA = 0.10 * 0.5, 0.50 * 0.5
CARACTERES_POR_TOKEN = 2.8     # español con el tokenizador nuevo (cuenta ~30 % más tokens que el anterior): conservador
MAX_CARACTERES = 200_000       # el texto de OCR rinde ~2,1 caracteres por token: queda bajo los 100 mil tokens
SALIDA_ESTIMADA = {"resoluciones": 2_500, "informes": 5_000}   # tokens de salida por documento, medidos en el piloto


def S(*tipos):
    return {"type": tipos[0]} if len(tipos) == 1 else {"type": list(tipos)}


def obj(props, nulo=False):
    o = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return {"anyOf": [o, {"type": "null"}]} if nulo else o


def lista(items):
    return {"type": "array", "items": items}


# La API acepta a lo más 16 campos con tipos unidos (nulos): los datos sueltos van como texto, con "" si faltan, y los
# números como texto («1520.50»); solo el impedimento y el vínculo pueden ser null.
TXT = TXTN = NUMN = BOOLN = S("string")

ESQUEMA_RES = obj({
    "numero": TXT, "fecha_resolucion": TXTN, "tipo": TXT, "resultado_recurso": TXTN,
    "expedientes": lista(obj({
        "administrado": obj({"nombre": TXT, "ruc": TXTN, "tipo": TXT}),
        "integrantes_consorcio": lista(obj({"nombre": TXT, "ruc": TXTN, "meses_sancion": NUMN})),
        "entidad": obj({"nombre": TXTN, "ruc": TXTN}),
        "contratacion": obj({"tipo": TXTN, "numero": TXTN, "procedimiento": TXTN, "fecha": TXTN, "fecha_es": TXTN,
                             "monto_soles": NUMN, "objeto": TXTN}),
        "infracciones": lista(obj({"literal": TXT, "articulo": TXT, "norma_citada": TXTN, "resumen": TXT,
                                   "fecha_hechos": TXTN, "cita_fecha_hechos": TXTN, "decision": TXT})),
        "fecha_denuncia": TXTN,
        "sancion_final": obj({"tipo": TXT, "meses": NUMN, "multa_soles": NUMN}),
        "cita_decision": TXTN,
        "impedimento": obj({"articulo": TXTN, "literal": TXTN, "causal": TXTN,
                            "autoridad": obj({"nombre": TXTN, "cargo": TXTN, "entidad": TXTN}),
                            "relacion_con_administrado": TXTN, "grado": TXTN, "cita": TXTN}, nulo=True),
        "documentos_cuestionados": lista(obj({"tipo": TXT, "emisor": TXTN, "falso_o_inexacto": TXTN})),
        "presento_descargos": BOOLN}))})

ESQUEMA_INF = obj({
    "numero": TXTN, "entidad": TXTN,
    "periodo_auditado": obj({"desde": TXTN, "hasta": TXTN}),
    "muestra": obj({"universo_ordenes": NUMN, "universo_soles": NUMN, "muestra_ordenes": NUMN, "muestra_soles": NUMN,
                    "cita": TXTN}),
    "hechos": lista(obj({
        "n": S("integer"), "sumilla": TXT, "tipos": lista(TXT),
        "fechas_hechos": obj({"desde": TXTN, "hasta": TXTN}),
        "perjuicio_soles": NUMN, "normas": lista(TXT),
        "ordenes": lista(obj({"tipo": TXTN, "numero": TXTN, "fecha": TXTN, "monto_soles": NUMN, "proveedor": TXTN,
                              "ruc_proveedor": TXTN})),
        "proveedores": lista(obj({"nombre": TXT, "ruc": TXTN, "rol": TXT, "cita": TXTN})),
        "personas": lista(obj({"nombre": TXT, "dni": TXTN, "cargo": TXTN, "rol": TXT, "responsabilidad": TXTN})),
        "vinculo": obj({"tipo": TXT, "descripcion": TXT, "cita": TXTN}, nulo=True),
        "descargos": obj({"presentados": BOOLN, "desvirtuan": TXTN})}))})

COMUN = """Extrae solo lo que dice el texto; si un dato no está, usa una cadena vacía "". Los montos y los meses van como texto con
solo dígitos y punto decimal (por ejemplo "1520.50"); sí/no como "si" o "no". No inventes RUC, DNI, números, montos ni fechas,
y no agregues datos que el texto no trae (por ejemplo, no completes la norma con modificatorias que no cita). Fechas en
AAAA-MM-DD. RUC de 11 dígitos y DNI de 8, como texto. Las citas son textuales y cortas (30 palabras o menos). El texto
viene de un PDF o de un OCR y puede traer errores de lectura."""

SIS_RES = f"""Eres abogado especialista en contrataciones del Estado peruano. Lees una resolución del Tribunal de
Contrataciones (TCE o TCP) para un proyecto académico de la UTEC sobre el riesgo en las compras públicas.
{COMUN}

- tipo: "sancionador" si resuelve un procedimiento sancionador; "reconsideracion" si resuelve un recurso de
  reconsideración; "otro" si no. En una reconsideración, resultado_recurso es "fundado", "fundado_en_parte",
  "infundado" o "improcedente".
- expedientes: uno por cada administrado sancionado o investigado (las resoluciones en serie traen varios). Un consorcio
  va como un solo expediente, con sus integrantes y los meses de sanción de cada uno.
- administrado.tipo: "persona_natural", "empresa" o "consorcio".
- contratacion: la orden, el contrato, el procedimiento o el trámite del RNP donde se cometió la infracción.
  contratacion.tipo: "orden_de_compra", "orden_de_servicio", "contrato", "procedimiento_de_seleccion",
  "catalogo_electronico", "rnp" u "otro". Si no hay fecha de la orden o del contrato, usa la de la buena pro o la de
  presentación de ofertas, y dilo en fecha_es ("orden", "contrato", "buena_pro", "oferta", "tramite_rnp" u "otro").
- infracciones: una por cada infracción imputada, con su propia fecha_hechos (cuándo se cometió: fecha de la orden o
  del contrato, de la presentación del documento, del vencimiento del plazo incumplido o de la resolución del contrato;
  nunca la fecha de la resolución del Tribunal ni la de la denuncia) y su propia decision: "sanciona", "no_ha_lugar",
  "archiva" u "otro". literal: solo la letra (por ejemplo "c"); articulo: solo el numeral (por ejemplo "50.1");
  norma_citada: la ley tal como la cita la resolución; resumen: 15 palabras o menos.
- sancion_final: la que queda vigente al terminar esta resolución (en una reconsideración infundada, la que se confirma).
  tipo: "inhabilitacion_temporal", "inhabilitacion_definitiva", "multa" o "ninguna".
- impedimento: solo si una infracción es contratar estando impedido; si no, null (el objeto entero). autoridad es quien ejerce el cargo que
  genera el impedimento (alcalde, regidor, consejero, funcionario), que puede ser el propio administrado.
  relacion_con_administrado: "es_la_misma", "conyuge", "conviviente", "padre_o_madre", "hijo_o_hija",
  "hermano_o_hermana", "abuelo_o_abuela", "nieto_o_nieta", "suegro_o_suegra", "yerno_o_nuera", "cunado_o_cunada",
  "socio_o_representante" u "otro".
- documentos_cuestionados: todos los documentos falsos o inexactos, cada uno con quien lo emitió y si es "falso",
  "inexacto" o "ambos"; lista vacía si no hay."""

SIS_INF = f"""Eres auditor gubernamental en el Perú. Lees un informe de control de la Contraloría General de la República
para un proyecto académico de la UTEC que prioriza las compras públicas menores (hasta 8 UIT) con riesgo.
{COMUN}

- hechos: uno por cada observación, hecho o situación adversa. tipos, de esta lista: "fraccionamiento",
  "elusion_procedimiento", "indagacion_cotizaciones", "conformidad_pago", "incumplimiento_penalidad",
  "proveedor_impedido", "vinculo_conflicto", "sobreprecio", "personal_por_ordenes", "otro".
- ordenes: todas las órdenes o contratos que el hecho cita con número, también las de cuadros y apéndices; tipo "O/C",
  "O/S", "contrato" u "otro". Si una orden se paga en varios comprobantes, una sola fila con el monto de la orden.
  Recorre los cuadros, anexos y apéndices y copia cada orden en su propia fila, aunque sean decenas: no resumas, no
  agrupes y no escribas «entre otras».
- proveedores: rol "beneficiado", "cotizante", "postor" u "otro". personas: rol "funcionario_o_servidor", "proveedor",
  "familiar" u "otro"; responsabilidad "administrativa", "civil", "penal" o "ninguna".
- vinculo: solo si el hecho describe un vínculo entre un proveedor y la entidad o sus funcionarios; tipo "familiar",
  "funcionario_proveedor", "socio_o_representante", "ex_trabajador" u "otro". Si no, null (el objeto entero).
- descargos: si las personas comprendidas presentaron comentarios y si estos desvirtúan el hecho ("si", "no",
  "en_parte" o "")."""


def clave():
    for linea in ENV.read_text(encoding="utf-8").splitlines():
        if linea.startswith("ANTHROPIC_API_KEY_LLIKA"):
            return linea.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No encontré ANTHROPIC_API_KEY_LLIKA en el .env")


def recortar(t):
    t = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", t))
    if len(t) <= MAX_CARACTERES:
        return t, False
    cabeza = int(MAX_CARACTERES * 0.8)       # el inicio trae los hechos; el final, la decisión
    return t[:cabeza] + "\n[...]\n" + t[-(MAX_CARACTERES - cabeza):], True


def documentos(fuente, conjunto):
    """[(custom_id, texto)] del conjunto: «piloto» son los mismos 25 del piloto de DeepSeek."""
    if fuente == "resoluciones":
        if conjunto == "piloto":
            buscados = set(pd.read_csv(SAL_DS / "piloto_resoluciones_muestra.csv", dtype=str)["numero"])
        else:
            buscados = None
        out = []
        with (TCE / "textos.jsonl").open(encoding="utf-8") as f:
            for linea in f:
                d = json.loads(linea)
                if buscados is None or d["numero"] in buscados:
                    out.append((d["numero"], f"Resolución {d['numero']} ({d['anio']}). Texto:\n{d['texto']}"))
        return list(dict(out).items())   # sin repetidos
    disp = textos_disponibles()
    if conjunto == "piloto":
        cres = list(muestra_informes()[0].query("texto_disponible")["CresCodigoFormato"])
    else:
        prio = pd.read_csv(DATOS.parent / "ocr_informes" / "prioridad_informes.csv", dtype=str)
        cres = [c for c in prio.loc[prio["prioridad"].astype(int) <= 3, "CresCodigoFormato"] if c in disp]
    return [(c, f"Informe {c}. Texto:\n{leer_texto(disp[c])}") for c in cres]


def ya_tomados():
    """custom_id con resultado bueno o en un lote todavía abierto: no se vuelven a mandar."""
    hechos = set()
    for f in SAL.glob("resultados_*.jsonl"):
        for linea in f.open(encoding="utf-8"):
            r = json.loads(linea)
            if "salida" in r:
                hechos.add(r["custom_id"])
    abiertos = set()
    if REGISTRO.exists():
        for linea in REGISTRO.open(encoding="utf-8"):
            r = json.loads(linea)
            if not r.get("bajado"):
                abiertos |= set(r["ids"])
    return hechos, abiertos


def peticion(fuente, cid, texto):
    t, cortado = recortar(texto)
    sistema = SIS_RES if fuente == "resoluciones" else SIS_INF
    esquema = ESQUEMA_RES if fuente == "resoluciones" else ESQUEMA_INF
    params = {"model": MODELO, "max_tokens": 16_000 if fuente == "resoluciones" else 32_000,
              "thinking": {"type": "disabled"},
              "system": [{"type": "text", "text": sistema, "cache_control": {"type": "ephemeral"}}],
              "output_config": {"format": {"type": "json_schema", "schema": esquema}},
              "messages": [{"role": "user", "content": t}]}
    return {"custom_id": re.sub(r"[^A-Za-z0-9_-]", "_", cid)[:64], "params": params}, cortado, len(t) + len(sistema)


def estimar(fuente, docs):
    caracteres = sum(len(recortar(t)[0]) for _, t in docs)
    entrada = caracteres / CARACTERES_POR_TOKEN
    usd = entrada / 1e6 * PRECIO_ENTRADA + len(docs) * SALIDA_ESTIMADA[fuente] / 1e6 * PRECIO_SALIDA
    cortados = sum(recortar(t)[1] for _, t in docs)
    return usd, entrada, cortados


def cmd_estimar(a):
    docs = documentos(a.fuente, a.conjunto)
    hechos, abiertos = ya_tomados()
    docs = [d for d in docs if re.sub(r"[^A-Za-z0-9_-]", "_", d[0])[:64] not in hechos | abiertos]
    usd, entrada, cortados = estimar(a.fuente, docs)
    print(f"{a.fuente}, {a.conjunto}: {len(docs)} documentos por mandar; ~{entrada / 1e6:.1f} millones de tokens de "
          f"entrada; {cortados} cortados a {MAX_CARACTERES:,} caracteres; costo estimado {usd:.2f} dólares (lote)")
    return docs, usd


def cmd_prueba(a):
    """Un documento por la vía normal (sin lote, precio completo): sirve para ver que el esquema y la clave andan."""
    cid, texto = documentos(a.fuente, "piloto")[0]
    p, _, _ = peticion(a.fuente, cid, texto)
    cliente = anthropic.Anthropic(api_key=clave())
    r = cliente.messages.create(**p["params"])
    txt = next((b.text for b in r.content if b.type == "text"), "")
    usd = (r.usage.input_tokens * 0.10 + r.usage.output_tokens * 0.50) / 1e6
    print(f"{cid}: stop_reason={r.stop_reason}; entrada {r.usage.input_tokens:,} tokens, salida "
          f"{r.usage.output_tokens:,}; {usd:.4f} dólares")
    print(json.dumps(json.loads(txt), ensure_ascii=False, indent=1)[:3000])


def cmd_enviar(a):
    docs, usd = cmd_estimar(a)
    if not docs:
        print("Nada que mandar: todo tiene resultado o está en un lote abierto.")
        return
    if usd > a.tope:
        sys.exit(f"La estimación ({usd:.2f}) pasa el tope ({a.tope}). No se manda nada.")
    if not a.enviar:
        print("Solo estimación. Para mandar, agregar --enviar.")
        return
    cliente = anthropic.Anthropic(api_key=clave())
    pets = [peticion(a.fuente, cid, t)[0] for cid, t in docs]
    # Un lote admite hasta 256 MB: se parte en tandas de ~150 MB. Cada tanda se anota apenas se crea, para que un corte a
    # medio camino no lleve a mandar dos veces lo mismo.
    tandas, actual, tam = [], [], 0
    for p in pets:
        b = len(json.dumps(p, ensure_ascii=False).encode("utf-8"))
        if actual and tam + b > 150_000_000:
            tandas.append(actual)
            actual, tam = [], 0
        actual.append(p)
        tam += b
    tandas.append(actual)
    SAL.mkdir(parents=True, exist_ok=True)
    for t in tandas:
        lote = cliente.messages.batches.create(requests=t)
        with REGISTRO.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"lote": lote.id, "fuente": a.fuente, "conjunto": a.conjunto, "ids": [p["custom_id"]
                                for p in t], "usd_estimado": round(usd * len(t) / len(pets), 3),
                                "enviado": time.strftime("%Y-%m-%d %H:%M"), "bajado": False}, ensure_ascii=False) + "\n")
        print(f"Lote {lote.id}: {len(t)} documentos, estado {lote.processing_status}", flush=True)


def cmd_estado(a):
    if not REGISTRO.exists():
        print("No hay lotes.")
        return
    cliente = anthropic.Anthropic(api_key=clave())
    for linea in REGISTRO.open(encoding="utf-8"):
        r = json.loads(linea)
        b = cliente.messages.batches.retrieve(r["lote"])
        c = b.request_counts
        print(f"{r['lote']} {r['fuente']} {r['conjunto']} ({r['enviado']}): {b.processing_status}; ok {c.succeeded}, "
              f"error {c.errored}, en proceso {c.processing}; bajado {r['bajado']}")


def cmd_bajar(a):
    if not REGISTRO.exists():
        return
    cliente = anthropic.Anthropic(api_key=clave())
    filas = [json.loads(x) for x in REGISTRO.open(encoding="utf-8")]
    for r in filas:
        if r["bajado"] or cliente.messages.batches.retrieve(r["lote"]).processing_status != "ended":
            continue
        usd, n = 0.0, 0
        with (SAL / f"resultados_{r['fuente']}.jsonl").open("a", encoding="utf-8") as f:
            for res in cliente.messages.batches.results(r["lote"]):
                fila = {"custom_id": res.custom_id, "lote": r["lote"]}
                if res.result.type == "succeeded":
                    m = res.result.message
                    u = m.usage
                    usd += ((u.input_tokens + (u.cache_creation_input_tokens or 0) * 1.25 +
                             (u.cache_read_input_tokens or 0) * 0.1) * PRECIO_ENTRADA +
                            u.output_tokens * PRECIO_SALIDA) / 1e6
                    txt = next((b.text for b in m.content if b.type == "text"), "")
                    fila.update({"stop_reason": m.stop_reason, "tokens_entrada": u.input_tokens,
                                 "tokens_salida": u.output_tokens})
                    try:
                        if m.stop_reason != "end_turn":
                            raise ValueError(m.stop_reason)
                        fila["salida"] = json.loads(txt)
                    except Exception as e:      # noqa: BLE001 - queda anotado y se puede volver a mandar
                        fila["error"] = f"{type(e).__name__}: {e}"[:200]
                else:
                    fila["error"] = res.result.type
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                n += 1
        r["bajado"], r["usd_real"] = True, round(usd, 4)
        print(f"{r['lote']}: {n} resultados, {usd:.3f} dólares")
    REGISTRO.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["estimar", "prueba", "enviar", "estado", "bajar"])
    ap.add_argument("--fuente", choices=["resoluciones", "informes"], default="resoluciones")
    ap.add_argument("--conjunto", choices=["piloto", "todo"], default="piloto")
    ap.add_argument("--tope", type=float, default=1.0, help="dólares; si la estimación lo pasa, no se manda")
    ap.add_argument("--enviar", action="store_true", help="sin esto, «enviar» solo estima")
    a = ap.parse_args()
    SAL.mkdir(parents=True, exist_ok=True)
    {"estimar": cmd_estimar, "prueba": cmd_prueba, "enviar": cmd_enviar, "estado": cmd_estado,
     "bajar": cmd_bajar}[a.paso](a)
