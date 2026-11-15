"""La vista 2 (hallazgos del control) medida por hora, en 2020, 2021 y 2023 (HALLAZGOS 157 y 160, C13 y C14).

Una hora por orden revisada (HALLAZGOS 59). Detección justa (155): un caso A cuenta si se revisa una de sus órdenes
citadas; un caso solo B, si se revisa su orden más cara. Casos: los de prueba de puntajes_d5, en las entidades-año con
algún hallazgo A o B. Políticas, con un presupuesto de órdenes por entidad-año:
- casos enteros por monto: los casos de mayor a menor monto, abriendo todas sus órdenes de la más cara a la más barata;
- vista 2: los mismos casos por monto, abriendo solo la orden más cara (la más cercana al tope) de cada uno;
- regla del tope: las órdenes de la entidad-año de la más cara a la más barata, una por una (referencia de la 155);
- azar: valor esperado exacto (hipergeométrica).
Comparador principal (160, M4): la regla del tope, las órdenes por monto. Abrir casos enteros queda como referencia de lo
que da la unidad de revisión: frente a eso, cualquier lista que abra una orden por caso gana. IC al 90 % remuestreando
entidades, tres cortes juntos. Presupuesto principal: 20 órdenes; sensibilidades: 10, 50 y las órdenes de los 5 casos más
caros; solo los hallazgos A; hechos distintos en vez de casos; solo control posterior; y sin cruces dudosos ni acreedores.

    python -I -X utf8 d16_hallazgos_por_hora.py
Salida: resultados/d16_hallazgos_por_hora.json y .md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etiquetas import BASE, HOY_EXPLORACION, K, SELLADO_ANIO, SELLADO_DESDE, Hechos  # noqa: E402

OUT = BASE.parent / "resultados"
PRUEBA = (2020, 2021, 2023)
RNG = np.random.default_rng(0)
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def citadas_de(H):
    """Órdenes citadas, respetando las sensibilidades de H."""
    return citadas(H) - (set(pd.read_parquet(BASE / "hechos_orden.parquet", columns=["id", "dudosa"]).query("dudosa == True")["id"]) if H.sin_dudosas else set())


def citadas(H):
    inf = pd.read_parquet(BASE / "informes.parquet", columns=["informe", "fecha_emision", "fecha_publicacion"]).set_index("informe")
    f_inf = inf["fecha_publicacion"].fillna(inf["fecha_emision"])
    ho = pd.read_parquet(BASE / "hechos_orden.parquet")
    A = ho[(ho["fuente"] == "A") & ho["entidad_coincide"].astype(bool)].copy()
    A["fecha_conocido"] = A["documento"].map(f_inf)
    A = A[(A["anio"] != SELLADO_ANIO) & (A["fecha_conocido"] < SELLADO_DESDE) & (A["fecha_conocido"] <= HOY_EXPLORACION)]
    return set(A["id"])


def hechos_del_caso(H):
    """Hechos (informe y número) de cada caso; un hecho grande cuenta una sola vez en la sensibilidad de hechos distintos."""
    e = H.ev[H.ev["fuente"].isin(["A", "B"])].astype({"ruc_entidad": str, "ruc": str})
    e = e.assign(h=e["documento"].astype(str) + "#" + e["hecho"].astype(str))
    return e.groupby(["ruc_entidad", "ruc", "anio_hechos"])["h"].agg(set)


def posteriores():
    i = pd.read_parquet(BASE / "informes.parquet", columns=["informe", "servicio"])
    return set(i.loc[~i["servicio"].fillna("").str.upper().str.contains("SIMULTANEO"), "informe"])


def armar(H, a, ids_citadas):
    te = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet", columns=K + ["grupo", "c_monto"])
    te = te.astype({"ruc_entidad": str, "ruc": str, "anio": int}).reset_index(drop=True)
    est = Hechos.etiqueta(H, te, fecha_corte=HOY_EXPLORACION)
    fu = est["fuentes"].fillna("")
    te["tiene_A"] = fu.str.contains("A").to_numpy()
    te["y_rev"] = fu.str.contains("A|B").to_numpy()
    te = te[te.groupby("grupo")["y_rev"].transform("any")].reset_index(drop=True)
    o = pd.read_parquet(BASE / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "anio", "monto"], filters=[("anio", "==", a)])
    o = o.astype({"ruc_entidad": str, "ruc": str}).assign(anio=a)
    o = o.merge(te[K + ["grupo", "y_rev", "tiene_A", "c_monto"]], on=K, how="inner")
    o = o.assign(j=RNG.random(len(o))).sort_values(["grupo", "monto", "j"], ascending=[True, False, True]).reset_index(drop=True)
    o["r_caso"] = o.groupby(K).cumcount() + 1
    o["citada"] = o["id"].isin(ids_citadas)
    o["delata"] = (o["y_rev"] & o["tiene_A"] & o["citada"]) | (o["y_rev"] & ~o["tiene_A"] & (o["r_caso"] == 1))
    o["delata_A"] = o["y_rev"] & o["tiene_A"] & o["citada"]
    o["caso"] = o.groupby(K, sort=False).ngroup()
    return te, o


def simular(o, B_por_grupo, col, hechos=None):
    """Hallazgos por entidad-año con cada política, revisando B órdenes. Con `hechos` ({caso: hechos}), cuenta hechos
    distintos hallados en vez de casos."""
    filas = []
    cuenta = (lambda cs: len(set().union(*[hechos.get(c, set()) for c in cs])) if len(cs) else 0) if hechos is not None else len
    for g, d in o.groupby("grupo", sort=False):
        B = int(B_por_grupo.get(g, 0)) if isinstance(B_por_grupo, dict) else B_por_grupo
        caso, dl = d["caso"].to_numpy(), d[col].to_numpy()
        # regla del tope: las órdenes por monto, una por una
        regla = cuenta(set(caso[:B][dl[:B]]))
        # casos por monto: el orden de los casos (por su monto total) y sus órdenes de la más cara a la más barata
        cm = d.groupby("caso", sort=False)["c_monto"].first().sort_values(ascending=False, kind="stable")
        pos_por_caso = d.groupby("caso", sort=False).indices
        enteros, gastado = set(), 0
        for c in cm.index:
            idx = pos_por_caso[c]                         # ya vienen de la más cara a la más barata
            toma = idx[: max(0, B - gastado)]
            gastado += len(toma)
            if dl[toma].any():
                enteros.add(c)
            if gastado >= B:
                break
        top = [pos_por_caso[c][0] for c in cm.index[:B]]   # la vista 2: solo la orden más cara de cada caso
        vista2 = cuenta(set(caso[top][dl[top]])) if top else 0
        filas.append({"grupo": g, "casos enteros por monto": cuenta(enteros), "vista 2": vista2, "regla del tope": regla, "B": B})
    return pd.DataFrame(filas).set_index("grupo")


def azar(o, B_por_grupo, col):
    N = o.groupby("grupo").size()
    k = o[o[col]].groupby(["grupo", "caso"]).size().rename("k").reset_index()
    k["N"] = k["grupo"].map(N)
    b = np.minimum(k["grupo"].map(B_por_grupo).fillna(0).to_numpy() if isinstance(B_por_grupo, dict) else B_por_grupo, k["N"])
    lp0 = gammaln(k["N"] - k["k"] + 1) - gammaln(k["N"] - k["k"] - b + 1) - gammaln(k["N"] + 1) + gammaln(k["N"] - b + 1)
    p0 = np.where(k["N"] - k["k"] >= b, np.exp(lp0), 0.0)
    return pd.Series(1 - p0, index=k.index).groupby(k["grupo"]).sum()


def razon(r, c1, c0, n=2000):
    pe = r.assign(e=pd.Series(r.index, index=r.index).str.split("|").str[0]).groupby("e")[[c1, c0]].sum()
    boot = [pe.iloc[RNG.integers(0, len(pe), len(pe))].pipe(lambda m: m[c1].sum() / max(m[c0].sum(), 1e-9)) for _ in range(n)]
    return {"razon": float(pe[c1].sum() / max(pe[c0].sum(), 1e-9)), "ic90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))]}


def main():
    H = Hechos(conocimiento="publicacion")
    ids = citadas(H)
    por_caso, post = hechos_del_caso(H), posteriores()
    con_post = H.ev[H.ev["fuente"].isin(["A", "B"]) & H.ev["documento"].isin(post)].astype({"ruc_entidad": str, "ruc": str})
    llaves_post = set(zip(con_post["ruc_entidad"], con_post["ruc"], con_post["anio_hechos"]))
    datos = {}
    for a in PRUEBA:
        te, o = armar(H, a, ids)
        o["delata_post"] = o["delata"].to_numpy() & np.array([k in llaves_post for k in zip(o["ruc_entidad"], o["ruc"], o["anio"])])
        datos[a] = (te, o)
        paso(f"{a}: {te['grupo'].nunique()} entidades-año con hallazgos, {len(te):,} casos, {int(te['y_rev'].sum())} hallazgos "
             f"({int((te['y_rev'] & te['tiene_A']).sum())} con A); {len(o):,} órdenes")

    def correr(datos_, col, etiqueta, por_hechos=False):
        partes = []
        for a, (te, o) in datos_.items():
            if etiqueta == "los 5 casos más caros":
                n_caso = o.groupby("caso").size()
                cm = o.groupby(["grupo", "caso"])["c_monto"].first().reset_index()
                cm["r"] = cm.groupby("grupo")["c_monto"].rank(ascending=False, method="first")
                B = cm[cm["r"] <= 5].assign(n=lambda z: z["caso"].map(n_caso)).groupby("grupo")["n"].sum().to_dict()
            else:
                B = int(etiqueta)
            hechos = None
            if por_hechos:
                c = o.drop_duplicates("caso")
                hechos = {cc: por_caso.get((e, r, an), set()) for cc, e, r, an in zip(c["caso"], c["ruc_entidad"], c["ruc"], c["anio"])}
            r = simular(o, B, col, hechos)
            if not por_hechos:
                r["azar"] = azar(o, B, col).reindex(r.index).fillna(0)
            partes.append(r)
        r = pd.concat(partes)
        horas = float(r["B"].sum())
        cols = ["casos enteros por monto", "vista 2", "regla del tope"] + ([] if por_hechos else ["azar"])
        fila = {"entidades_anio": len(r), "horas": horas, "hallazgos": {c: float(r[c].sum()) for c in cols},
                "vista2_frente_a_la_regla": razon(r, "vista 2", "regla del tope"),
                "vista2_frente_a_casos_enteros": razon(r, "vista 2", "casos enteros por monto"),
                "por_anio": {str(a): razon(r[r.index.str.endswith(str(a))], "vista 2", "regla del tope")["razon"] for a in PRUEBA}}
        if not por_hechos:
            fila["vista2_frente_al_azar"] = razon(r, "vista 2", "azar")
        fila["horas_por_hallazgo"] = {c: horas / max(v, 1e-9) for c, v in fila["hallazgos"].items()}
        v = fila["vista2_frente_a_la_regla"]
        paso(f"[{col}; {etiqueta}{'; hechos' if por_hechos else ''}] {fila['hallazgos']}; vista 2 / regla {v['razon']:.2f} [{v['ic90'][0]:.2f} a {v['ic90'][1]:.2f}]; por año {fila['por_anio']}")
        return fila

    res = {"presupuestos": {}}
    for nombre, col in (("A o B", "delata"), ("solo A", "delata_A")):
        for etiqueta in ("20", "10", "50", "los 5 casos más caros"):
            res["presupuestos"][f"{nombre} | {etiqueta}"] = correr(datos, col, etiqueta)
    res["presupuestos"]["hechos distintos | 20"] = correr(datos, "delata", "20", por_hechos=True)
    res["presupuestos"]["solo control posterior | 20"] = correr(datos, "delata_post", "20")
    H2 = Hechos(conocimiento="publicacion", sin_dudosas=True, sin_acreedores=True)
    ids2 = citadas_de(H2)
    res["presupuestos"]["sin dudosas ni acreedores | 20"] = correr({a: armar(H2, a, ids2) for a in PRUEBA}, "delata", "20")
    (OUT / "d16_hallazgos_por_hora.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Vista 2: hallazgos del control por hora (HALLAZGOS 157 y 160)", "",
         "| Detección y presupuesto | Horas | Casos enteros | Vista 2 | Regla del tope | Azar | Vista 2 / regla del tope [IC 90 %] | Vista 2 / casos enteros |",
         "|---|---|---|---|---|---|---|---|"]
    for k_, f in res["presupuestos"].items():
        h, v, e = f["hallazgos"], f["vista2_frente_a_la_regla"], f["vista2_frente_a_casos_enteros"]
        L.append(f"| {k_} | {f['horas']:,.0f} | {h['casos enteros por monto']:.0f} | {h['vista 2']:.0f} | {h['regla del tope']:.0f} | "
                 f"{h.get('azar', float('nan')):.1f} | {v['razon']:.2f} [{v['ic90'][0]:.2f} a {v['ic90'][1]:.2f}] | {e['razon']:.2f} |")
    (OUT / "d16_hallazgos_por_hora.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
