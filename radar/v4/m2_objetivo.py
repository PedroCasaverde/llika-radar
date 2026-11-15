"""Prueba de modelos 2 (HALLAZGOS 160, sección 20): con qué etiqueta aprende D1, midiendo siempre con la ventana (a). Fijado antes de correr.

    python -I -X utf8 m2_objetivo.py elegir       # solo 2018 → 2019, por proveedores separados: resultados/m2_elegir.json
    python -I -X utf8 m2_objetivo.py confirmar    # una vez, en 2020, 2021 y 2023: puntajes_m2_<año> y m2_confirmar.md
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d15_variables import NUEVAS, lista_121, nivel  # noqa: E402
from etiquetas import K, Hechos  # noqa: E402
from evaluacion import armar_corte, cargar_x, familia_a_la_decision, universo  # noqa: E402
from m1_d1_configuracion import ajustar, preparar_te  # noqa: E402
from medidas import fmt, medir  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4" / "resultados"
ARBOLES = (50, 100, 200)
CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
OBJETIVOS = ("T0", "T1", "T2", "T3")
T0_RELOJ = time.time()


def paso(t):
    print(f"[{(time.time() - T0_RELOJ) / 60:5.1f} min] {t}", flush=True)


def etiqueta_ajuste(H, tr, objetivo, corte):
    """y y peso de ajuste. Las filas son las mismas para todas: las excluye lo ya conocido al cierre, igual que la prueba."""
    if objetivo == "T0":
        e = H.etiqueta_amplia(tr, corte)
        return e["y"].to_numpy(), e["peso"].to_numpy()
    if objetivo == "T1":
        e = H.etiqueta_amplia(tr, corte)
        y = nivel(e["fuentes"]) >= 2
        return y, np.where(y, e["peso"].to_numpy(), 0.0)
    e = H.etiqueta_amplia(tr, corte, ventana_anios=None)
    y, peso = e["y"].to_numpy().copy(), e["peso"].to_numpy().copy()
    if objetivo == "T3":                # lo del año siguiente, repartido entre los casos del proveedor en el año
        am = H.amplio[H.amplio["fecha_conocido"] <= pd.Timestamp(corte)]
        sig = set(zip(am["ruc"].astype(str), am["anio_hechos"].astype(int) - 1))
        base = pd.DataFrame({"ruc": tr["ruc"].astype(str).to_numpy(), "anio": tr["anio"].astype(int).to_numpy()})
        k = base.groupby(["ruc", "anio"])["ruc"].transform("size").to_numpy()
        tiene = np.array([x in sig for x in zip(base["ruc"], base["anio"])])
        peso = np.where(tiene & ~y, 1.0 / k, peso)
        y = y | tiene
    return y, peso


def mitad(ruc):
    return np.array([zlib.crc32(str(r).encode()) % 2 for r in ruc])


def elegir():
    H = Hechos(conocimiento="publicacion")
    c116 = [c for c in lista_121() if c not in NUEVAS]
    x = familia_a_la_decision(cargar_x([2018, 2019]), 2019)
    hist = H.historia(x)
    x = pd.concat([x, hist], axis=1)
    x = x[universo(x, hist)].copy()
    tr, te = x[x["anio"] == 2018].copy(), x[x["anio"] == 2019].copy()
    et_tr = H.etiqueta_amplia(tr, "2020-12-31")
    tr = tr[~et_tr["conocido_al_cierre"]].reset_index(drop=True)
    te = te.join(H.etiqueta_amplia(te, "2021-12-31"))
    te = preparar_te(te[~te["conocido_al_cierre"]].reset_index(drop=True))
    m_tr, m_te = mitad(tr["ruc"]), mitad(te["ruc"])
    paso(f"2018: {len(tr):,} casos; 2019: {len(te):,} casos, {int(te['y'].sum()):,} positivos, en sanciones {int(te['y_S'].sum()):,}")
    filas = {}
    for obj in OBJETIVOS:
        y, peso = etiqueta_ajuste(H, tr, obj, "2021-12-31" if obj in ("T2", "T3") else "2020-12-31")
        t = tr.assign(y=y, peso=peso)
        paso(f"{obj}: {int(y.sum()):,} positivos de ajuste en 2018")
        partes = {a: {} for a in ARBOLES}
        for f in (0, 1):
            cand = ajustar(t[m_tr == f], te[m_te != f], c116, ARBOLES, True)
            sub = te[m_te != f]
            for a in ARBOLES:
                partes[a][f] = sub.assign(s_m=cand[a].mean(axis=0))
        for a in ARBOLES:
            cortes = {f + 1: partes[a][f] for f in (0, 1)}
            r = {yy: medir(cortes, yy, {"m": "s_m", "tamano": "s_tamano"}, ["tamano"], n_boot=200)["metodos"] for yy in ("y", "y_S", "y_P")}
            py, ps = r["y"]["m"]["precision"], r["y_S"]["m"]["precision"]
            filas[f"{obj}|{a}"] = {"precision_y": py, "precision_S": ps, "precision_P": r["y_P"]["m"]["precision"], "media_geometrica": float(np.sqrt(py * ps)),
                                   "y_frente_tamano": fmt(r["y"]["m"]["frente_a"]["tamano"]), "S_frente_tamano": fmt(r["y_S"]["m"]["frente_a"]["tamano"]),
                                   "P_frente_tamano": fmt(r["y_P"]["m"]["frente_a"]["tamano"])}
            paso(f"   {obj}, {a} árboles: {filas[f'{obj}|{a}']}")
    gana = max(filas, key=lambda k: filas[k]["media_geometrica"])
    (OUT / "m2_elegir.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "m2_elegido.json").write_text(json.dumps({"objetivo": gana.split("|")[0], "arboles": int(gana.split("|")[1])}), encoding="utf-8")
    paso(f"ELEGIDO por la regla: {gana}")


def confirmar():
    assert not (OUT / "m2_confirmar.json").exists(), "la confirmación se corre una sola vez"
    el = json.loads((OUT / "m2_elegido.json").read_text(encoding="utf-8"))
    assert el["objetivo"] != "T0", "ganó T0: no se confirma (sería casi la prueba 1)"
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cortes = {}
    for a, anios in CORTES.items():
        tr, te = armar_corte(H, anios, a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        y, peso = etiqueta_ajuste(H, tr, el["objetivo"], f"{a}-12-31")
        paso(f"corte {anios} → {a}: ajuste {len(tr):,} casos, {int(y.sum()):,} positivos con {el['objetivo']}; prueba {len(te):,}")
        P = ajustar(tr.assign(y=y, peso=peso), te, c116, (el["arboles"],), True)[el["arboles"]]
        d = te[K].astype({"ruc_entidad": str, "ruc": str, "anio": int}).copy()
        d["s_m2"] = P.mean(axis=0)
        d.to_parquet(OUT / f"puntajes_m2_{a}.parquet", index=False)
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet").astype({"ruc_entidad": str, "ruc": str, "anio": int})
        n0 = len(b)
        b = b.merge(d, on=K, how="inner")
        assert len(b) == n0, "la prueba no son los mismos casos"
        b["y"] = b["y"].astype(bool)
        b["y_P"], b["y_S"], b["y_ABC"] = b["y"] & (b["nivel"] == 1), b["y"] & (b["nivel"] == 2), b["y"] & (b["nivel"] == 3)
        cortes[a] = b
    M = {"m2": "s_m2", "D1_linea_base": "s_D1_binario", "tamano": "s_tamano_proveedor"}
    res = {"elegido": el}
    L = [f"# Prueba de modelos 2: D1 entrenado con {el['objetivo']}, {el['arboles']} árboles y 10 semillas", ""]
    for nom, y in (("Señal adversa", "y"), ("Solo penalidades", "y_P"), ("Sanciones del Tribunal", "y_S"), ("A, B o C en la ventana", "y_ABC")):
        r = medir(cortes, y, M, ["tamano", "D1_linea_base"])
        res[y] = r
        L += [f"## {nom}", "", "| Método | Precisión en 5 | Frente al tamaño | Por año | Frente a D1 de la línea base | Por año |", "|---|---|---|---|---|---|"]
        for m, d in r["metodos"].items():
            ta, ba = d["frente_a"].get("tamano"), d["frente_a"].get("D1_linea_base")
            pa = lambda q: ", ".join("nd" if x["razon"] is None else f"{x['razon']:.2f}" for x in q["por_anio"].values()) if q else "—"  # noqa: E731
            L.append(f"| {m} | {d['precision']:.4f} | {fmt(ta) if ta else '—'} | {pa(ta)} | {fmt(ba) if ba else '—'} | {pa(ba)} |")
        L.append("")
    gy, gs = res["y"]["metodos"]["m2"]["frente_a"]["D1_linea_base"], res["y_S"]["metodos"]["m2"]["frente_a"]["D1_linea_base"]
    sube = lambda g, u: sum(1 for x in g["por_anio"].values() if x["razon"] and x["razon"] >= u)  # noqa: E731
    via_s = gs["razon"] >= 1.05 and sube(gs, 1.05) >= 2 and gy["razon"] >= 0.95
    via_y = gy["razon"] >= 1.01 and sube(gy, 1.01) >= 2 and gs["razon"] >= 0.95
    res["adopta"] = {"y": gy["razon"], "S": gs["razon"], "cortes_y": sube(gy, 1.01), "cortes_S": sube(gs, 1.05), "via_S": bool(via_s), "via_y": bool(via_y), "se_adopta": bool(via_s or via_y)}
    L += [f"**Regla de adopción:** {res['adopta']}"]
    (OUT / "m2_confirmar.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "m2_confirmar.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    {"elegir": elegir, "confirmar": confirmar}[sys.argv[1]]()
    paso("listo")
