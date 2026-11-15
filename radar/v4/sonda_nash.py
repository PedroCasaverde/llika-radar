"""El precio de sortear (HALLAZGOS 160, sección 81; Pedro, 7/11: «¿podrías probar el equilibrio de Nash para ver cómo mejora LLIKA?»)

No es un equilibrio de Nash: los boletos de los sorteos son elegidos, no salen del juego, porque faltan lo que gana el
proveedor que incumple (G) y lo que pierde si lo sancionan (F). Con datos pasados tampoco se mide la disuasión.
Se mide el precio de mezclar (un puesto por sorteo, o los cinco) y qué tan cerca del corte están los aciertos.
Solo lectura y solo agregados: mismas listas y modelos de lista_unica.py, en los años de exploración (2020, 2021 y 2023).
No toca 2022, 2024 ni la lista congelada de 2025. No guarda RUC ni nombres. Debe reproducir la lista (177, 229 y 240)
y el monto de la sección 80 (298 en total); si no, se detiene.

    python -I -X utf8 sonda_nash.py
Salida: resultados/lista_unica/sonda_nash.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_a, sys.argv = sys.argv, ["x"]
import lista_unica as LU  # noqa: E402
from casos import DIR, eventos, top  # noqa: E402
from etiquetas import HOY_EXPLORACION, Hechos  # noqa: E402

sys.argv = _a

SORTEOS = 200
ACTUAL = "actual"
VARIANTES = ("A. sorteo dentro de la bandera", "B. sorteo por riesgo", "C. sorteo parejo", "D. los cinco por sorteo")
ANTES = {2020: 177, 2021: 229, 2023: 240}      # la lista aprobada, por año (sección 77)
DESCARTAR, SUSTENTAR, HORA = (0.75, 1.0), (16, 24), 65
VALOR = {"10 % del promedio sin el 1 % mayor": 6541, "20 % del promedio sin el 1 % mayor": 13082, "10 % de la mediana": 569}
PUESTOS = 10
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def tramos(gid):
    """gid ya ordenado: inicio y largo de cada grupo."""
    ini = np.r_[0, np.flatnonzero(gid[1:] != gid[:-1]) + 1]
    return ini, np.diff(np.r_[ini, len(gid)])


def preparar_sorteo(g, elegible, peso):
    """Un puesto por grupo, con boletos según el peso. Si un grupo no tiene peso positivo, todos llevan el mismo boleto."""
    pos = np.flatnonzero(elegible)
    pos = pos[np.argsort(g[pos], kind="stable")]
    ini, lon = tramos(g[pos])
    w = peso[pos].astype(float)
    lw = np.full(len(pos), -np.inf)
    lw[w > 0] = np.log(w[w > 0])
    sin = np.repeat(~np.isfinite(np.maximum.reduceat(lw, ini)), lon)
    lw[sin] = 0.0
    return pos, ini, lon, lw, np.repeat(np.arange(len(ini)), lon)


def sortear_uno(prep, rng):
    """Máximo de log(peso) + Gumbel en cada grupo: sale uno, con probabilidad proporcional al peso."""
    pos, ini, lon, lw, seg = prep
    k = lw + rng.gumbel(size=len(pos))
    es = np.flatnonzero(k == np.maximum.reduceat(k, ini)[seg])
    _, pr = np.unique(seg[es], return_index=True)
    return pos[es[pr]]


def prob_uno(prep, n):
    """Probabilidad exacta de salir en el sorteo de un puesto."""
    pos, ini, lon, lw, seg = prep
    w = np.exp(lw - np.maximum.reduceat(lw, ini)[seg])
    p = np.zeros(n)
    p[pos] = w / np.add.reduceat(w, ini)[seg]
    return p


def sortear_cinco(gD, posD, primero, lw, rng, k=5):
    """Los k de cada grupo: primero el vínculo, en su orden; luego, sin reposición, con boletos según el peso; los de peso cero, al final."""
    u = rng.gumbel(size=len(posD))
    clave = np.where(primero > -np.inf, 1e6 + primero, np.where(np.isfinite(lw), lw + u, -1e6 + u))
    o = np.lexsort((-clave, gD))
    ini, lon = tramos(gD[o])
    cc = np.arange(len(o)) - np.repeat(ini, lon)
    return posD[o[cc < k]]


def aciertos(sel, Y, rc, nr):
    """Proveedores distintos con cada verdad, y filas con señal."""
    r = {}
    for v, y in Y.items():
        m = np.zeros(nr, bool)
        m[rc[sel[y[sel]]]] = True
        r[v] = int(m.sum())
    r["por_fila"] = int(Y["señal adversa"][sel].sum())
    return r


def horas(filas, h):
    en_vano = (filas - h) / h
    return [round(SUSTENTAR[i] + DESCARTAR[i] * en_vano, 1) for i in (0, 1)]


def corte_de_equilibrio():
    """Parte 1, sin correr nada: el umbral del auditor, p* = c_d / (V − c_s + c_d), contra lo que acierta cada tramo (sección 80)."""
    k = json.loads((LU.SALIDA_L / "sonda_corte_k.json").read_text(encoding="utf-8"))
    tr, previo = [], None
    for K, f in k["cortes"].items():
        if previo is None:
            tr.append({"puestos": f"1 a {K}", "por_100": round(100 * f["aciertos"]["lista única"] / f["filas"], 2)})
        else:
            tr.append({"puestos": f"{previo + 1} a {K}", "por_100": f["frente_al_corte_anterior"]["por_100_filas_adicionales"]})
        previo = int(K)
    r = {"tramos": tr, "umbral": {}}
    for nombre, V in VALOR.items():
        r["umbral"][nombre] = []
        for i in (0, 1):
            cd, cs = DESCARTAR[i] * HORA, SUSTENTAR[i] * HORA
            p = 100 * cd / (V - cs + cd) if V - cs + cd > 0 else None
            hasta = None
            for t in tr:
                if p is None or t["por_100"] < p:
                    break
                hasta = t["puestos"]
            r["umbral"][nombre].append({"descartar_y_sustentar_soles": [cd, cs], "umbral_por_100": None if p is None else round(p, 2), "se_paga_hasta": hasta})
    return r, {int(a): v for a, v in k["cortes"]["5"]["por_anio_monto"].items()}


def main():
    corte, monto_anio = corte_de_equilibrio()
    paso(f"ETAPA corte de equilibrio: {corte['umbral']}")
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    E = eventos()
    Pa = pd.read_parquet(DIR / "proveedor_anio.parquet").astype({"ruc": str, "anio": int})
    cprov, nombres = LU.comunes()

    todas = (ACTUAL,) + VARIANTES
    tot = {ACTUAL: {}, "monto": {}, "sin_el_quinto": {}}
    filas = 0
    por_sorteo = {v: [] for v in VARIANTES}
    marco0 = {v: [] for v in VARIANTES}
    marco_actual, marco_monto = [], []
    perfil = {"filas": np.zeros(PUESTOS + 1), "aciertos_fila": np.zeros(PUESTOS + 1), **{v: np.zeros(PUESTOS + 1) for v in todas}}
    alcance = {v: {"casos": 0, "con_probabilidad": 0, "fuera": 0, "fuera_con_1_por_100": 0} for v in todas}
    fuera = {"filas_con_senal": 0, **{v: 0.0 for v in VARIANTES}}
    sorteado = {"actual (el quinto puesto)": [0.0, 0], **{v: [0.0, 0] for v in VARIANTES[:3]}}
    margen, margen_bandera = [], []
    grupos = grandes = 0
    cols = ["ruc", "anio", "y"]

    for a, anios in LU.EXPLORACION.items():
        d = LU.armar_anio(a, anios, H, E, Pa, cprov, nombres, HOY_EXPLORACION)
        q = LU.preparar_lista(d, a)
        del d
        assert isinstance(q.index, pd.RangeIndex) and q.index.start == 0 and q["anio"].nunique() == 1
        base = LU.lista(q)
        N = len(q)
        en = np.zeros(N, bool)
        en[base.index.to_numpy()] = True
        quinto = np.zeros(N, bool)
        quinto[base.index[base["es_quinto"].to_numpy()].to_numpy()] = True
        g = pd.factorize(q["grupo"])[0]
        G = int(g.max()) + 1
        tam = np.bincount(g, minlength=G)
        grande_g = tam > 5
        grande = grande_g[g]
        A4 = en & grande & ~quinto
        assert en[~grande].all() and (np.bincount(g[A4], minlength=G)[grande_g] == 4).all() and (np.bincount(g[quinto], minlength=G)[grande_g] == 1).all(), "la lista no tiene 4 + 1 por entidad"
        R4 = grande & ~A4
        s = q["s_lista"].to_numpy(float)
        punt = q["puntaje"].to_numpy(float)
        monto = q["c_monto"].to_numpy(float)
        bandera = (q["f30"] | q["f_cerca"]).to_numpy(bool)
        vinc = (q["grado"] >= 2).to_numpy()
        rc = pd.factorize(q["ruc"])[0]
        nr = int(rc.max()) + 1
        Y = {v: q[c].astype(bool).to_numpy() for v, c in LU.VERDADES.items()}
        y = Y["señal adversa"]

        sel0 = np.flatnonzero(en)
        h0 = aciertos(sel0, Y, rc, nr)
        assert h0["señal adversa"] == ANTES[a], f"{a}: la lista da {h0['señal adversa']} y debía dar {ANTES[a]}"
        selM = top(q, "s_monto").index.to_numpy()
        hM = aciertos(selM, Y, rc, nr)
        assert hM["señal adversa"] == monto_anio[a] and len(selM) == len(sel0), f"{a}: el monto da {hM['señal adversa']} y debía dar {monto_anio[a]}"
        h5 = aciertos(np.flatnonzero(en & ~quinto), Y, rc, nr)
        for n_, h_ in ((ACTUAL, h0), ("monto", hM), ("sin_el_quinto", h5)):
            for k_, v_ in h_.items():
                tot[n_][k_] = tot[n_].get(k_, 0) + v_
        filas += len(sel0)
        marco_actual.append(q.loc[sel0, cols].astype({"y": bool}))
        marco_monto.append(q.loc[selM, cols].astype({"y": bool}))
        paso(f"ETAPA {a}: lista actual {h0['señal adversa']} y monto {hM['señal adversa']}, como antes; sin el quinto puesto {h5['señal adversa']}")

        # Puesto de cada caso por el puntaje, dentro de su entidad (el empate, por orden de llegada, como nlargest)
        o = np.lexsort((np.arange(N), -s, g))
        ini, lon = tramos(g[o])
        puesto = np.empty(N, np.int64)
        puesto[o] = np.arange(N) - np.repeat(ini, lon)
        distinto = int((A4 != (grande & (puesto < 4))).sum())
        s5 = np.full(G, np.nan)
        m5 = lon > 4
        s5[g[o][ini[m5]]] = s[o[ini[m5] + 4]]

        # Cuánto tendría que bajar su puntaje un acierto de los puestos 1 a 4, sin vínculo, para quedar bajo el quinto
        mm = A4 & ~vinc & y & (s > 0)
        margen.extend((1 - s5[g[mm]] / s[mm]).tolist())
        margen_bandera.extend(bandera[mm].tolist())

        con_b = (np.bincount(g[R4 & bandera], minlength=G) > 0)[g]
        prep = {VARIANTES[0]: preparar_sorteo(g, R4 & (bandera | ~con_b), monto), VARIANTES[1]: preparar_sorteo(g, R4, punt), VARIANTES[2]: preparar_sorteo(g, R4, np.ones(N))}
        for v, p_ in prep.items():
            assert len(p_[1]) == int(grande_g.sum()), f"{v}: hay entidades sin candidato"
        fijo = np.flatnonzero(~grande | A4)
        chico = np.flatnonzero(~grande)
        posD = np.flatnonzero(grande)
        gD = g[posD]
        wD = punt[posD]
        lwD = np.full(len(posD), -np.inf)
        lwD[wD > 0] = np.log(wD[wD > 0])
        primD = np.where(vinc[posD], s[posD], -np.inf)

        pi = {ACTUAL: en.astype(float)}
        for v in VARIANTES[:3]:
            pi[v] = np.where(~grande | A4, 1.0, prob_uno(prep[v], N))
            assert np.allclose(np.bincount(g, weights=pi[v], minlength=G)[grande_g], 5), f"{v}: no suma 5 por entidad"
        cntD = np.zeros(N)

        res = {v: [] for v in VARIANTES}
        paso(f"ETAPA {a}: empiezan los sorteos; {distinto} filas con empate en el puesto 4")
        for j in range(SORTEOS):
            for i, v in enumerate(VARIANTES):
                rng = np.random.default_rng([i, a, j])
                if i < 3:
                    sel = np.concatenate([fijo, sortear_uno(prep[v], rng)])
                else:
                    el = sortear_cinco(gD, posD, primD, lwD, rng)
                    cntD[el] += 1
                    sel = np.concatenate([chico, el])
                if j == 0:
                    assert len(sel) == len(sel0) and len(np.unique(sel)) == len(sel), f"{v}: no da cinco por entidad"
                    marco0[v].append(q.loc[sel, cols].astype({"y": bool}))
                res[v].append(aciertos(sel, Y, rc, nr))
            if j % 50 == 49:
                paso(f"ETAPA {a}: {j + 1} sorteos; media hasta aquí {[round(float(np.mean([r_['señal adversa'] for r_ in res[v]])), 1) for v in VARIANTES]}")
        for v in VARIANTES:
            por_sorteo[v].append(res[v])
        pi[VARIANTES[3]] = np.where(~grande, 1.0, cntD / SORTEOS)

        # Probabilidad de revisión por puesto, solo en entidades con más de 5 casos
        b = np.minimum(puesto, PUESTOS)[grande]
        perfil["filas"] += np.bincount(b, minlength=PUESTOS + 1)
        perfil["aciertos_fila"] += np.bincount(b, weights=y[grande].astype(float), minlength=PUESTOS + 1)
        for v in todas:
            perfil[v] += np.bincount(b, weights=pi[v][grande], minlength=PUESTOS + 1)

        # Qué parte de los casos tiene alguna probabilidad (en D, la teórica)
        pw = np.zeros(N, bool)
        pw[posD] = (wD > 0) | vinc[posD]
        npw = np.bincount(g[pw], minlength=G)
        positivo = {v: pi[v] > 0 for v in todas}
        positivo[VARIANTES[3]] = ~grande | pw | (grande & (npw[g] < 5))
        for v in todas:
            alcance[v]["casos"] += N
            alcance[v]["con_probabilidad"] += int(positivo[v].sum())
            alcance[v]["fuera"] += int((~en).sum())
            alcance[v]["fuera_con_1_por_100"] += int(((pi[v] >= 0.01) & ~en).sum())

        mf = y & ~en
        fuera["filas_con_senal"] += int(mf.sum())
        for v in VARIANTES:
            fuera[v] += float(pi[v][mf].sum())
        sorteado["actual (el quinto puesto)"][0] += float(y[quinto].sum())
        sorteado["actual (el quinto puesto)"][1] += int(quinto.sum())
        for v in VARIANTES[:3]:
            sorteado[v][0] += float((pi[v] * (R4 & y)).sum())
            sorteado[v][1] += int(grande_g.sum())
        grupos += G
        grandes += int(grande_g.sum())
        paso(f"ETAPA {a}: listo")
        del q, base

    assert tot[ACTUAL]["señal adversa"] == 646 and tot["monto"]["señal adversa"] == 298, "no reproduce 646 y 298"
    paso("reproduce 646 y 298")
    h_act, h_mon = tot[ACTUAL]["señal adversa"], tot["monto"]["señal adversa"]
    MA, MM = pd.concat(marco_actual, ignore_index=True), pd.concat(marco_monto, ignore_index=True)
    r = {"anios": list(LU.EXPLORACION), "sorteos": SORTEOS, "entidades_anio": grupos, "entidades_con_mas_de_5_casos": grandes, "filas": filas, "corte_de_equilibrio": corte}
    r["actual"] = {**tot[ACTUAL], "frente_al_monto": LU.ic_proveedores(MA, MM), "horas_por_acierto": horas(filas, h_act)}
    r["monto"] = tot["monto"]
    r["techo"] = {"sin_el_quinto_puesto": tot["sin_el_quinto"]["señal adversa"], "solo_del_quinto_puesto": h_act - tot["sin_el_quinto"]["señal adversa"], "pct": round(100 * (h_act - tot["sin_el_quinto"]["señal adversa"]) / h_act, 1)}
    r["variantes"] = {}
    for v in VARIANTES:
        T = {k: np.array([sum(por_sorteo[v][t][j][k] for t in range(len(por_sorteo[v]))) for j in range(SORTEOS)]) for k in por_sorteo[v][0][0]}
        h = T["señal adversa"]
        A0 = pd.concat(marco0[v], ignore_index=True)
        f = {"aciertos": {"media": round(float(h.mean()), 1), "p5": float(np.percentile(h, 5)), "p95": float(np.percentile(h, 95)), "min": int(h.min()), "max": int(h.max())},
             "cambio_frente_a_la_actual_pct": round(100 * (float(h.mean()) / h_act - 1), 1),
             "por_fila_media": round(float(T["por_fila"].mean()), 1),
             "por_verdad_media": {k: round(float(T[k].mean()), 1) for k in LU.VERDADES},
             "frente_al_monto_media": round(float(h.mean()) / h_mon, 2),
             "sorteo_0": {"aciertos": int(h[0]), "frente_al_monto": LU.ic_proveedores(A0, MM), "frente_a_la_actual": LU.ic_proveedores(A0, MA)},
             "horas_por_acierto": horas(filas, float(h.mean()))}
        f["pasa_la_regla"] = bool(f["cambio_frente_a_la_actual_pct"] >= -5 and f["frente_al_monto_media"] >= 2)
        r["variantes"][v] = f
        paso(f"{v}: {f['aciertos']}; cambio {f['cambio_frente_a_la_actual_pct']} %; frente al monto {f['frente_al_monto_media']}; pasa la regla: {f['pasa_la_regla']}")
    nom = [str(i + 1) for i in range(PUESTOS)] + [f"{PUESTOS + 1} o más"]
    r["por_puesto"] = {nom[i]: {"filas": int(perfil["filas"][i]), "aciertos_por_100_filas": round(100 * float(perfil["aciertos_fila"][i] / perfil["filas"][i]), 2), "probabilidad_de_revision": {v: round(float(perfil[v][i] / perfil["filas"][i]), 3) for v in todas}} for i in range(PUESTOS + 1)}
    r["alcance"] = {v: {"casos_con_alguna_probabilidad_pct": round(100 * x["con_probabilidad"] / x["casos"], 1), "fuera_de_la_lista_actual_con_1_por_100_o_mas_pct": round(100 * x["fuera_con_1_por_100"] / x["fuera"], 1)} for v, x in alcance.items()}
    r["aciertos_que_la_lista_actual_no_mira"] = {"filas": fuera["filas_con_senal"], "filas_esperadas_por_sorteo": {v: round(fuera[v], 1) for v in VARIANTES}}
    r["puesto_sorteado_aciertos_por_100_filas"] = {v: round(100 * x[0] / x[1], 2) for v, x in sorteado.items()}
    x, bb = np.array(margen), np.array(margen_bandera)
    r["margen_para_salir"] = {"filas": int(len(x)), "mediana_pct": round(100 * float(np.median(x)), 1), "p25_pct": round(100 * float(np.percentile(x, 25)), 1), "p75_pct": round(100 * float(np.percentile(x, 75)), 1), "hasta_10_pct": round(100 * float((x <= 0.10).mean()), 1), "hasta_25_pct": round(100 * float((x <= 0.25).mean()), 1), "hasta_50_pct": round(100 * float((x <= 0.50).mean()), 1), "con_bandera_pct": round(100 * float(bb.mean()), 1)}
    r["fragil"] = bool(np.median(x) <= 0.25)
    sal = LU.SALIDA_L / "sonda_nash.json"
    sal.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float), flush=True)
    paso("FIN")


if __name__ == "__main__":
    main()
