"""El auditor sin inflar (HALLAZGOS 160, sección 42; Pedro, 4/11: «corrige bien al auditor y no lo infles»). Solo lectura, sin API y sin 2022. Fijado antes de correr.

Tres cosas inflaban al auditor en la sección 41, y se corrigen:
1. Entraba solo con lo que observó. Ahora paga lo que revisó: 1 / 0,322 revisiones por caso observado (611 de 1 896, un techo: favorece al auditor), y 5 revisiones (la mediana) donde hubo control sin hallazgo.
2. Se comparaba solo donde ya había hallado algo. Ahora, en todas las entidades-año con control conocidas.
3. Se le contaba lo que supo después. Ahora un acierto cuenta si la fecha de emisión del informe es anterior a la denuncia del hecho (o a su publicación si no hay fecha de denuncia). A LLIKA se le aplica lo mismo, con el cierre del año.

    python -X utf8 auditor_corregido.py
Salida: resultados/por_caso/auditor_corregido.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from casos import DIR, OUT, PRUEBA, SALIDA, TIPOS, etiquetas_caso, eventos  # noqa: E402
from etiquetas import BASE, HOY_EXPLORACION, K  # noqa: E402

TASA = 611 / 1896
REV_SIN_HALLAZGO = 5
LISTAS = {"alerta y monto": "s_alerta_monto", "modelo por proveedor": "s_S_prov", "tamaño": "s_tamano_proveedor", "regla por monto": "s_monto"}
VERDADES = {"impedimento": "y_imp_real", "otra sanción por la orden": "y_C_otra", "proveedor sancionado": "y_S", "proveedor con penalidad": "y_P"}
RNG = np.random.default_rng(0)


def fechas_evento(E, d):
    """Primera fecha en que el hecho entró en el circuito: denuncia, o publicación si no hay. En C, por caso; en S y P, por proveedor y año."""
    e = E.assign(f=E["fecha_denuncia"].fillna(E["fecha_conocido"]))
    e = e[e["f"] <= HOY_EXPLORACION]
    out = {}
    c = e[e["fuente"] == "C"].astype({"ruc_entidad": str}).rename(columns={"anio_hechos": "anio"})
    r = c[c["impedimento_real"]].groupby(K)["f"].min()
    o = c.groupby(K)["f"].min()
    idx = pd.MultiIndex.from_frame(d[K])
    out["y_imp_real"] = r.reindex(idx).to_numpy()
    out["y_C_otra"] = o.reindex(idx).to_numpy()
    i2 = pd.MultiIndex.from_frame(d[["ruc", "anio"]])
    for y, f in (("y_S", "S"), ("y_P", "P")):
        out[y] = e[e["fuente"] == f].rename(columns={"anio_hechos": "anio"}).groupby(["ruc", "anio"])["f"].min().reindex(i2).to_numpy()
    return out


def fecha_auditor(E, d):
    inf = pd.read_parquet(BASE / "informes.parquet", columns=["informe", "fecha_emision", "fecha_publicacion"]).set_index("informe")
    em = inf["fecha_emision"].fillna(inf["fecha_publicacion"])
    a = E[E["fuente"].isin(["A", "B"])].astype({"ruc_entidad": str}).rename(columns={"anio_hechos": "anio"})
    a = a.assign(f=pd.to_datetime(a["documento"].map(em)).fillna(a["fecha_conocido"]))
    return a.groupby(K)["f"].min().reindex(pd.MultiIndex.from_frame(d[K])).to_numpy()


def cuenta(t, y, f_lista):
    hit = t[y].to_numpy(dtype=bool)
    desc = hit & (pd.to_datetime(t[f_lista]).to_numpy() < pd.to_datetime(t[f"f_{y}"]).to_numpy())
    return {"casos": len(t), "aciertos": int(hit.sum()), "descubre": int(desc.sum()), "prov_anio": t.groupby(["ruc", "anio"]).ngroups, "prov_anio_descubre": t[desc].groupby(["ruc", "anio"]).ngroups}


def elegir(d, col, n_por):
    s = d[col].astype(float).to_numpy()
    s = s + 1e-9 * RNG.random(len(s)) * (np.abs(s) + 1)
    ll = n_por.index.name
    q = d.assign(_s=s).sort_values([ll, "_s"], ascending=[True, False])
    return q[q.groupby(ll).cumcount() < q[ll].map(n_por)]


def ic(a, b, n=2000):
    pe = pd.DataFrame({"a": a, "b": b}).fillna(0)
    if pe["b"].sum() == 0:
        return None
    r = np.random.default_rng(0)
    bs = [(lambda w: (w * pe["a"]).sum() / max((w * pe["b"]).sum(), 1e-9))(r.poisson(1.0, len(pe))) for _ in range(n)]
    return [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]


def main():
    E = eventos()
    partes = []
    for a in PRUEBA:
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet", columns=K + ["grupo", "s_monto", "s_tamano_proveedor", "s_alerta_monto"]).astype(TIPOS)
        b = b.merge(pd.read_parquet(SALIDA / f"puntajes_p2_{a}.parquet", columns=K + ["s_S_prov"]).astype(TIPOS), on=K)
        partes.append(b.join(etiquetas_caso(E, b, HOY_EXPLORACION)))
    d = pd.concat(partes, ignore_index=True)
    d["f_lista"] = pd.to_datetime(d["anio"].astype(str) + "-12-31")
    d["f_auditor"] = fecha_auditor(E, d)
    for y, f in fechas_evento(E, d).items():
        d[f"f_{y}"] = f
    aud = d[d["y_rev"]]
    res = {"auditor": {v: cuenta(aud, y, "f_auditor") for v, y in VERDADES.items()}, "tasa_observada_por_revisada": TASA}
    print("auditor, 1 315 casos:", res["auditor"], flush=True)
    # entidades-año con control, con o sin hallazgo
    T = pd.read_parquet(DIR / "entidad_anio.parquet")
    T["grupo"] = T["ruc_entidad"].astype(str) + "|" + T["anio"].astype(int).astype(str)
    grupos = set(d["grupo"])
    sin = set(T.loc[T["control_sin_hallazgo"] & T["anio"].isin(PRUEBA), "grupo"]) & grupos
    obs = aud.groupby("grupo").size()
    n_g = pd.concat([np.ceil(obs / TASA).astype(int), pd.Series(REV_SIN_HALLAZGO, index=sorted(sin - set(obs.index)))]).rename_axis("grupo")
    print(f"entidades-año: {len(n_g):,} ({len(obs)} con hallazgo y {len(n_g) - len(obs)} con control sin hallazgo); revisiones que paga {int(n_g.sum()):,}", flush=True)
    t = d[d["grupo"].isin(n_g.index)].copy()
    n_a = t.assign(n=t["grupo"].map(n_g)).drop_duplicates("grupo").groupby("anio")["n"].sum().rename_axis("anio")
    print("revisiones por año:", n_a.to_dict(), flush=True)
    for ambito, base, n_por in (("donde_hubo_control", t, n_g), ("todo_el_pais", d, n_a)):
        res[ambito] = {}
        for lista, col in LISTAS.items():
            top = elegir(base, col, n_por)
            fila = {}
            for v, y in VERDADES.items():
                c = cuenta(top, y, "f_lista")
                A = res["auditor"][v]
                ent_l = top[top[y].to_numpy() & (pd.to_datetime(top["f_lista"]).to_numpy() < pd.to_datetime(top[f"f_{y}"]).to_numpy())].groupby("ruc_entidad").size()
                ent_a = aud[aud[y].to_numpy() & (pd.to_datetime(aud["f_auditor"]).to_numpy() < pd.to_datetime(aud[f"f_{y}"]).to_numpy())].groupby("ruc_entidad").size()
                u = ent_l.index.union(ent_a.index)
                al, aa = top[top[y]].groupby("ruc_entidad").size(), aud[aud[y]].groupby("ruc_entidad").size()
                ua = al.index.union(aa.index)
                fila[v] = {**c, "razon_descubre": c["descubre"] / A["descubre"] if A["descubre"] else None, "ic90": ic(ent_l.reindex(u), ent_a.reindex(u)), "por_revision": c["descubre"] / max(c["casos"], 1),
                           "razon_aciertos": c["aciertos"] / A["aciertos"] if A["aciertos"] else None, "ic90_aciertos": ic(al.reindex(ua), aa.reindex(ua))}
            res[ambito][lista] = fila
            print(ambito, lista, "casos", len(top), {v: (f["aciertos"], f["descubre"], f["prov_anio_descubre"], None if f["razon_aciertos"] is None else round(f["razon_aciertos"], 2), f["ic90_aciertos"] and [round(x, 2) for x in f["ic90_aciertos"]]) for v, f in fila.items()}, flush=True)
    (SALIDA / "auditor_corregido.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
