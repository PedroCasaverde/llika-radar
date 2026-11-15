"""Fase 2 (HALLAZGOS 160, sección 39): escalera de modelos, casos 1 a 4. Fijado antes de correr.

    python -X utf8 p2_escalera.py puntajes    # ajusta por corte: resultados/por_caso/puntajes_p2_<año>.parquet
    python -X utf8 p2_escalera.py medir       # elige con 2020 y 2021, confirma una vez: resultados/por_caso/p2_resultado.json y .md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from casos import DIR, OUT, PRUEBA, SALIDA, TIPOS, dos_columnas, etiquetas_caso, eventos, ic_prov_anio  # noqa: E402
from d15_variables import NUEVAS, lista_121  # noqa: E402
from etiquetas import HOY_EXPLORACION, K, SELLADO_ANIO, Hechos  # noqa: E402
from evaluacion import armar_corte  # noqa: E402
from m1_d1_configuracion import ajustar  # noqa: E402
from m6_proveedor import NUEVAS_P, Fuentes, modelo_proveedor  # noqa: E402
from m7_variantes import lambdarank  # noqa: E402
from medidas import fmt, medir  # noqa: E402

CORTES = {2020: [2018], 2021: [2018, 2019], 2023: [2018, 2019, 2020, 2021]}
GAMMAS, THETAS = (1.0, 0.7, 0.4), (0.5, 1.0)
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


class Tasa:
    """Eventos esperados en el año = E^θ · (a + eventos previos con descuento) / (b + exposición previa con descuento). Todo a la fecha: la historia es de años anteriores y conocida al cierre."""

    def __init__(self, E, tipo, expo):
        self.e = E[E["fuente"] == tipo][["ruc", "anio_hechos", "fecha_conocido"]].astype({"ruc": str})
        self.expo = expo            # {año: Series por ruc}

    def historia(self, rucs, t, gamma, theta):
        fin = pd.Timestamp(f"{t}-12-31")
        e = self.e[(self.e["fecha_conocido"] <= fin) & (self.e["anio_hechos"] < t)]
        ev = (gamma ** (t - e["anio_hechos"])).groupby(e["ruc"]).sum().reindex(rucs).fillna(0.0).to_numpy()
        ex = np.zeros(len(rucs))
        for s, v in self.expo.items():
            if s < t and s != SELLADO_ANIO:
                ex += gamma ** (t - s) * (v.reindex(rucs).fillna(0.0).to_numpy() ** theta)
        return ev, ex

    def filas(self, prov, gamma, theta, corte=None):
        """prov: DataFrame con ruc y anio, sin repetir. Devuelve historia, exposición del año y eventos conocidos a `corte`."""
        H_, X_, E_, Y_ = np.zeros(len(prov)), np.zeros(len(prov)), np.zeros(len(prov)), np.zeros(len(prov))
        for t, g in prov.groupby("anio"):
            i = g.index.to_numpy()
            r = g["ruc"].to_numpy()
            H_[i], X_[i] = self.historia(pd.Index(r), int(t), gamma, theta)
            E_[i] = self.expo[int(t)].reindex(r).fillna(0.0).to_numpy() ** theta
            if corte is not None:
                e = self.e[(self.e["anio_hechos"] == t) & (self.e["fecha_conocido"] <= pd.Timestamp(corte))]
                Y_[i] = e.groupby("ruc").size().reindex(r).fillna(0).to_numpy()
        return H_, X_, E_, Y_

    def ajustar(self, prov, corte):
        mejor = None
        for g in GAMMAS:
            for th in THETAS:
                h, x, ex, y = self.filas(prov, g, th, corte)
                u = ex > 0
                h, x, ex, y = h[u], x[u], ex[u], y[u]

                def nll(lp):
                    a, b = np.exp(lp)
                    r, q = a + h, (b + x) / (b + x + ex)
                    return -(gammaln(y + r) - gammaln(r) + r * np.log(q) + y * np.log1p(-q)).sum()
                o = minimize(nll, np.log([0.01, 10.0]), method="L-BFGS-B", bounds=[(-12, 6), (-6, 14)])
                if mejor is None or o.fun < mejor[0]:
                    mejor = (o.fun, g, th, *np.exp(o.x))
        self.par = dict(zip(("nll", "gamma", "theta", "a", "b"), mejor))
        return self.par

    def puntaje(self, prov):
        p = self.par
        h, x, ex, _ = self.filas(prov, p["gamma"], p["theta"])
        return ex * (p["a"] + h) / (p["b"] + x)


def variables_logit(t):
    L = lambda c: np.log1p(t[c].astype(float).clip(lower=0).fillna(0))  # noqa: E731
    return pd.DataFrame({"monto": L("p_monto"), "ordenes": L("p_ordenes"), "ganados": L("p_exp_ocds_ganados"), "catalogo": L("p_cat_monto"), "entidades": L("p_entidades"), "anios": t["p_anios_previos_activo"].astype(float).fillna(0),
                         "penalidades": L("h_prov_penalidades"), "sanciones": L("h_prov_sanciones_otras"), "inhabilitaciones": L("h_prov_inhabilitaciones"), "multas": L("h_prov_multas"), "ejecucion": L("p_ejec_contratos"),
                         "cred_pen": np.log(t["h_prov_cred_pen"].astype(float) + 1e-6), "cred_san": np.log(t["h_prov_cred_san"].astype(float) + 1e-6), "nuevo": t["p_nuevo"].astype(float).fillna(0)})


def logit(ptr, y, pte):
    X = variables_logit(ptr)
    mu, sd = X.mean(), X.std().replace(0, 1)
    m = LogisticRegression(C=1.0, max_iter=1000).fit((X - mu) / sd, y)
    return m.decision_function((variables_logit(pte) - mu) / sd)


def puntajes():
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    E = eventos()
    P = pd.read_parquet(DIR / "proveedor_anio.parquet").astype({"ruc": str, "anio": int})
    F = Fuentes()
    cnt = F.o.rename(columns={"n": "x"})
    con = F.c.assign(anio=F.c["firma"].dt.year).groupby(["ruc", "anio"]).size().rename("x").reset_index()
    ex = pd.concat([cnt, con]).groupby(["ruc", "anio"])["x"].sum().reset_index()
    expo = {int(a): g.set_index("ruc")["x"].astype(float) for a, g in ex.groupby("anio")}
    c116 = [c for c in lista_121() if c not in NUEVAS]
    cprov = [c for c in c116 if c.startswith(("p_", "h_prov_"))] + NUEVAS_P
    nuevas = P[["ruc", "anio"] + NUEVAS_P]
    params = {}
    for a, anios in CORTES.items():
        corte = f"{a}-12-31"
        tr, te = armar_corte(H, anios, a)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        tr, te = tr.astype({"ruc": str, "ruc_entidad": str, "anio": int}), te.astype({"ruc": str, "ruc_entidad": str, "anio": int})
        tr, te = tr.merge(nuevas, on=["ruc", "anio"], how="left"), te.merge(nuevas, on=["ruc", "anio"], how="left")
        v = etiquetas_caso(E, tr, corte, ventana=None)
        y_tr = {"S": v["y_S"].to_numpy(), "P": v["y_P"].to_numpy()}
        k = tr.groupby(["ruc", "anio"])["ruc"].transform("size").to_numpy()
        d = te[K].copy()
        ptr, pte = tr[["ruc", "anio"]].drop_duplicates().reset_index(drop=True), te[["ruc", "anio"]].drop_duplicates().reset_index(drop=True)
        ptrx, ptex = ptr.merge(P, on=["ruc", "anio"], how="left"), pte.merge(P, on=["ruc", "anio"], how="left")
        paso(f"corte {a}: ajuste {len(tr):,} casos y {len(ptr):,} proveedores-año; prueba {len(te):,} y {len(pte):,}")
        for tipo in ("S", "P"):
            y = y_tr[tipo]
            yp = tr.assign(_y=y).groupby(["ruc", "anio"])["_y"].max().reindex(pd.MultiIndex.from_frame(ptr)).to_numpy()
            T = Tasa(E, tipo, expo)
            params[f"{a}_{tipo}"] = T.ajustar(ptr, corte)
            paso(f"   {tipo}: {int(yp.sum()):,} proveedores-año positivos; tasa {params[f'{a}_{tipo}']}")
            s = pte.assign(_t=T.puntaje(pte), _l=logit(ptrx, yp, ptex))
            d = d.merge(s.rename(columns={"_t": f"s_{tipo}_tasa", "_l": f"s_{tipo}_logit"}), on=["ruc", "anio"], how="left")
            paso(f"   {tipo}: tasa y logística listas")
            d[f"s_{tipo}_prov"] = modelo_proveedor(tr, y, te, cprov, (100,))[100][0]
            paso(f"   {tipo}: modelo por proveedor listo")
            d[f"s_{tipo}_d1"] = ajustar(tr.assign(y=y, peso=1.0 / k), te, c116, (50,), True)[50].mean(axis=0)
            paso(f"   {tipo}: D1 con objetivo propio listo")
            if tipo == "S":
                d["s_S_rank"] = lambdarank(tr, y.astype(int), te, c116, (50,))[50]
                paso("   S: lambdarank listo")
        d.to_parquet(SALIDA / f"puntajes_p2_{a}.parquet", index=False)
        paso(f"corte {a} guardado")
        del tr, te, d, ptrx, ptex
    (SALIDA / "p2_parametros.json").write_text(json.dumps(params, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def cargar():
    E = eventos()
    partes = {}
    for a in PRUEBA:
        b = pd.read_parquet(OUT / f"puntajes_d18_{a}.parquet", columns=K + ["grupo", "c_monto", "alerta", "s_monto", "s_tamano_proveedor", "s_alerta_monto"]).astype(TIPOS)
        n0 = len(b)
        b = b.merge(pd.read_parquet(OUT / f"puntajes_m2_{a}.parquet", columns=K + ["s_m2"]).astype(TIPOS), on=K).merge(pd.read_parquet(SALIDA / f"puntajes_p2_{a}.parquet").astype(TIPOS), on=K)
        assert len(b) == n0, f"la prueba no son los mismos casos en {a}"
        partes[a] = b.join(etiquetas_caso(E, b, HOY_EXPLORACION))
    return partes


def linea(n, r):
    return f"| {n} | {r['aciertos']:,} de {r['revisiones']:,} ({r['precision_caso']:.3f}) | {r['prov_anio_acierto']:,} de {r['prov_anio_revisados']:,} ({r['precision_prov_anio']:.3f}) | " + " | ".join(f"{v['precision_prov_anio']:.3f}" for v in r["por_anio"].values()) + " |"


CAB = ["| Lista | Por caso | Por proveedor-año | 2020 | 2021 | 2023 |", "|---|---|---|---|---|---|"]


def medir_():
    f_res = SALIDA / "p2_resultado.json"
    assert not f_res.exists(), "la confirmación se corre una sola vez"
    cortes = cargar()
    d = pd.concat(cortes.values(), ignore_index=True)
    res, L = {}, ["# Fase 2: escalera por caso (HALLAZGOS 160, sección 39)", ""]
    # caso 1
    M1 = {"alerta y monto": "s_alerta_monto", "monto": "s_monto", "tamaño": "s_tamano_proveedor", "D1 vigente": "s_m2"}
    t = dos_columnas(d, "y_imp_real", M1)
    r = medir(cortes, "y_imp_real", {"alerta": "s_alerta_monto", "monto": "s_monto"}, ["monto"])["metodos"]["alerta"]["frente_a"]["monto"]
    pa = {k: v["razon"] for k, v in r["por_anio"].items()}
    pasa = bool(r["razon"] >= 1.05 and sum(1 for v in pa.values() if v and v >= 1.05) >= 2 and r["ic90_entidad"][0] > 1)
    res["caso_1"] = {"columnas": t, "razon": r["razon"], "ic90_entidad": r["ic90_entidad"], "por_anio": pa, "pasa": pasa, "prov_anio": ic_prov_anio(d, "y_imp_real", "s_alerta_monto", "s_monto")}
    paso(f"caso 1: alerta frente al monto {r['razon']:.2f} {r['ic90_entidad']}; por año {pa}; PASA {pasa}")
    L += ["## Caso 1. Impedimento", ""] + CAB + [linea(n, x) for n, x in t.items()] + ["", f"Alerta frente al monto, por caso: {r['razon']:.2f} {r['ic90_entidad']}; por año {pa}. Pasa: **{pasa}**. Por proveedor-año: {res['caso_1']['prov_anio']}.", ""]
    # casos 3 y 4
    for nom, y, tipo, cand in (("caso_3", "y_S", "S", ["tasa", "logit", "prov", "d1", "vig", "rank"]), ("caso_4", "y_P", "P", ["tasa", "logit", "prov", "d1", "vig"])):
        M = {"tamaño": "s_tamano_proveedor", "monto": "s_monto"}
        for c in cand:
            M[c] = "s_m2" if c == "vig" else f"s_{tipo}_{c}"
        ini = dos_columnas(d[d["anio"].isin([2020, 2021])], y, M)
        p0 = ini["tamaño"]["precision_prov_anio"]
        mejor = max(ini[c]["precision_prov_anio"] for c in cand)
        elegida = next(c for c in cand if ini[c]["precision_prov_anio"] >= 0.95 * mejor)
        pasa_ini = ini[elegida]["precision_prov_anio"] >= 1.05 * p0
        paso(f"{nom}: al elegir, tamaño {p0:.4f}; " + "; ".join(f"{c} {ini[c]['precision_prov_anio']:.4f}" for c in cand) + f"; elegida {elegida}; pasa a confirmar {pasa_ini}")
        t = dos_columnas(d, y, M)
        fila = {"elegir": {c: ini[c]["precision_prov_anio"] for c in ["tamaño", "monto"] + cand}, "elegida": elegida, "pasa_a_confirmar": bool(pasa_ini), "columnas": t, "pasa": False}
        fila["vigente_frente_tamano"] = ic_prov_anio(d, y, "s_m2", "s_tamano_proveedor")
        fila["vigente_frente_monto"] = ic_prov_anio(d, y, "s_m2", "s_monto")
        fila["tamano_frente_monto"] = ic_prov_anio(d, y, "s_tamano_proveedor", "s_monto")
        if pasa_ini:
            col = M[elegida]
            g = ic_prov_anio(d, y, col, "s_tamano_proveedor")
            pa = {a: ic_prov_anio(d[d["anio"] == a], y, col, "s_tamano_proveedor", n=1)["razon"] for a in PRUEBA}
            ok = bool(g["razon"] >= 1.05 and sum(1 for v in pa.values() if v and v >= 1.05) >= 2 and g["ic90"][0] > 1)
            fila |= {"frente_tamano": g, "por_anio": pa, "pasa": ok, "frente_vigente": ic_prov_anio(d, y, col, "s_m2") if elegida != "vig" else None,
                     "por_caso": fmt(medir(cortes, y, {"m": col, "t": "s_tamano_proveedor"}, ["t"])["metodos"]["m"]["frente_a"]["t"])}
            paso(f"{nom}: {elegida} frente al tamaño {g}; por año {pa}; PASA {ok}")
        else:
            paso(f"{nom}: nada pasa a confirmar; queda el tamaño; PASA False")
        res[nom] = fila
        L += [f"## {nom}: {y}", ""] + CAB + [linea(n, x) for n, x in t.items()] + ["", f"Al elegir: {fila['elegir']}. Elegida: **{elegida}**. Pasa a confirmar: {pasa_ini}. Pasa: **{fila['pasa']}**. {fila.get('frente_tamano', '')} {fila.get('por_anio', '')}", f"D1 vigente frente al tamaño: {fila['vigente_frente_tamano']}; frente al monto: {fila['vigente_frente_monto']}. Tamaño frente al monto: {fila['tamano_frente_monto']}.", ""]
    # descriptivos
    M = {"monto": "s_monto", "tamaño": "s_tamano_proveedor", "alerta y monto": "s_alerta_monto", "D1 vigente": "s_m2", "tasa S": "s_S_tasa", "logística S": "s_S_logit", "proveedor S": "s_S_prov", "D1 S": "s_S_d1", "lambdarank S": "s_S_rank"}
    for nom, y, tit in (("caso_2", "y_C_otra", "Caso 2. Otra sanción por la orden (solo se describe)"), ("P_misma_entidad", "y_P_ent", "Penalidad en la misma entidad (solo se describe)"), ("S_misma_entidad", "y_S_ent", "Sanción en la misma entidad (solo se describe)")):
        t = dos_columnas(d, y, M)
        res[nom] = {"columnas": t, "positivos": int(d[y].sum())}
        L += [f"## {tit}: {int(d[y].sum())} casos", ""] + CAB + [linea(n, x) for n, x in t.items()] + [""]
    t = dos_columnas(d, "y", {"D1 vigente": "s_m2", "tamaño": "s_tamano_proveedor", "monto": "s_monto"})
    res["vista_general"] = {"columnas": t, "D1_frente_monto": ic_prov_anio(d, "y", "s_m2", "s_monto"), "D1_frente_tamano": ic_prov_anio(d, "y", "s_m2", "s_tamano_proveedor")}
    L += ["## Vista general: señal adversa", ""] + CAB + [linea(n, x) for n, x in t.items()] + ["", f"D1 frente al monto: {res['vista_general']['D1_frente_monto']}; frente al tamaño: {res['vista_general']['D1_frente_tamano']}.", ""]
    f_res.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (SALIDA / "p2_resultado.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    {"puntajes": puntajes, "medir": medir_}[sys.argv[1]]()
    paso("listo")
