"""Capa 2, posible padre o madre e hijo con un funcionario de decisión de la entidad que compra (HALLAZGOS 136; Pedro:
«aprovecha para calcular parientes entre padre e hijos»).

En el Perú el hijo lleva el paterno del padre y el paterno de la madre (como su materno). Entre el proveedor (persona
natural, RUC 10) y un funcionario con cargo de decisión de la entidad que le compra, mientras tiene el cargo:
- mismo_paterno: paterno del proveedor = paterno del funcionario (padre e hijo en cualquier sentido);
- cruzado: materno del proveedor = paterno del funcionario (ella es su madre), o paterno del proveedor = materno del
  funcionario (el proveedor es su madre o padre por la línea materna).
Se excluyen los que ya coinciden en los dos apellidos (hermanos: alerta de familiar de radar/02).

Probabilidad bayesiana como en radar/02, con un apellido:
- azar esperado del par: λ = funcionarios de decisión de la entidad en el año × frecuencia de ese apellido entre los
  proveedores personas del departamento, encogida hacia la nacional (1 000 personas de peso);
- exceso sobre el azar: observados menos la media de 100 permutaciones por año (funcionarios repartidos entre entidades
  de su departamento); sin exceso no hay alerta;
- probabilidad = r / (r + λ), con r tal que la suma de probabilidades iguala el exceso.
Visibilidad a la fecha: la primera declaración presentada por alguno de los funcionarios que la generan (como 132).
Solo imprime agregados.

    python -I -X utf8 capa2_padre_hijo.py
Salida: datos/procesados/v4/x_padre_hijo.parquet, resultados/padre_hijo_azar.md y filas en diccionario_x.csv.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
from comun import ANIOS, SALIDA, funcionarios, norm_serie  # noqa: E402
from capa2_familia import presentaciones  # noqa: E402

spec = importlib.util.spec_from_file_location("t25", AQUI.parent / "25_tarjeta_territorial.py")
T25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T25)

V4 = AQUI.parents[1] / "datos" / "procesados" / "v4"
K = ["ruc_entidad", "ruc", "anio"]
N_PERM, M_PREVIO = 100, 1000
ANIOS_DECISION = range(2022, 2026)
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def main():
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "depto", "ruc", "nombre", "fecha", "anio"])
    o = o[o["ruc"].str.startswith("10")].copy()
    o["nombre"] = norm_serie(o["nombre"])
    nombres = pd.Series(o["nombre"].unique())
    partes = np.array([T25.apellidos(x) for x in nombres], dtype=object)
    mapa_ap, mapa_am = dict(zip(nombres, partes[:, 0])), dict(zip(nombres, partes[:, 1]))
    o["ap"], o["am"] = o["nombre"].map(mapa_ap), o["nombre"].map(mapa_am)
    paso(f"órdenes a personas naturales: {len(o):,}; nombres distintos: {len(nombres):,}")
    f = funcionarios()
    f = f[f["decide"]].merge(presentaciones(), on=["persona", "ruc_entidad"], how="left")
    personas_func = set(f["persona"])
    cand = o[~o["nombre"].isin(personas_func)]
    # frecuencia de cada apellido (como paterno) entre los proveedores personas: local (departamento) y nacional
    per = cand.drop_duplicates("ruc")[["ruc", "depto", "ap"]]
    p_nac = per.groupby("ap").size() / len(per)
    n_loc = per.groupby(["depto", "ap"]).size()
    n_dep = per.groupby("depto").size()

    def p_loc(depto, apellido):
        nl = np.array([n_loc.get((d, a), 0) for d, a in zip(depto, apellido)], dtype=float)
        pn = apellido.map(p_nac).fillna(1 / len(per)).to_numpy()
        return (nl + M_PREVIO * pn) / (depto.map(n_dep).fillna(0).to_numpy() + M_PREVIO)

    alertas, filas = [], []
    for y in ANIOS:
        ini, fin = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        s = f[(f["inicio"] <= fin) & (f["cese"] >= ini)].reset_index(drop=True)
        pv = cand[cand["anio"] == y][["ruc_entidad", "depto", "ruc", "nombre", "ap", "am"]].drop_duplicates(["ruc_entidad", "ruc"]).reset_index(drop=True)
        ents = sorted(set(s["ruc_entidad"]) | set(pv["ruc_entidad"]))
        ce = {r: i for i, r in enumerate(ents)}
        apells = sorted(set(s["ap"]) | set(s["am"]) | set(pv["ap"]) | set(pv["am"]))
        ca = {a: i for i, a in enumerate(apells)}
        B = np.int64(len(ca) + 1)
        pe = pv["ruc_entidad"].map(ce).to_numpy(np.int64)
        pap, pam = pv["ap"].map(ca).to_numpy(np.int64), pv["am"].map(ca).to_numpy(np.int64)
        se = s["ruc_entidad"].map(ce).to_numpy(np.int64)
        sap, sam = s["ap"].map(ca).to_numpy(np.int64), s["am"].map(ca).to_numpy(np.int64)
        dep = s["depto"].fillna("SIN DEPTO").to_numpy()
        grupos = [np.flatnonzero(dep == x) for x in np.unique(dep)]

        def aciertos(a):
            ap_f, am_f = np.unique(a * B + sap), np.unique(a * B + sam)
            herm = np.isin((pe * B + pap) * B + pam, np.unique((a * B + sap) * B + sam))
            mismo = np.isin(pe * B + pap, ap_f) & ~herm
            cruz = (np.isin(pe * B + pam, ap_f) | np.isin(pe * B + pap, am_f)) & ~herm & ~mismo
            return mismo, cruz

        obs = aciertos(se)
        perm = {"mismo_paterno": [], "cruzado": []}
        for k in range(N_PERM):
            a = se.copy()
            rng = np.random.default_rng(k + 1000 * y)
            for g in grupos:
                a[g] = se[g][rng.permutation(len(g))]
            m, c = aciertos(a)
            perm["mismo_paterno"].append(int(m.sum()))
            perm["cruzado"].append(int(c.sum()))
        n_e = s.groupby("ruc_entidad").size()
        for patron, hit in zip(("mismo_paterno", "cruzado"), obs):
            pr = np.array(perm[patron])
            exceso = max(int(hit.sum()) - pr.mean(), 0.0)
            h = pv[hit].copy()
            # apellido que calza: el paterno en mismo_paterno; en cruzado, el que esté entre los del funcionario
            if patron == "mismo_paterno":
                h["apellido"] = h["ap"]
            else:
                ap_f = set(zip(s["ruc_entidad"], s["ap"]))
                h["apellido"] = np.where([(e, a) in ap_f for e, a in zip(h["ruc_entidad"], h["am"])], h["am"], h["ap"])
            lam = (h["ruc_entidad"].map(n_e).fillna(0).to_numpy() * p_loc(h["depto"], h["apellido"])).clip(min=1e-12)
            if exceso > 0 and len(h):
                r = brentq(lambda r: (r / (r + lam)).sum() - exceso, 1e-12, 1e4)
                h["prob"] = r / (r + lam)
            else:
                h["prob"] = 0.0
            filas.append({"año": y, "patrón": patron, "funcionarios": len(s), "observados": int(hit.sum()),
                          "azar_media": round(pr.mean(), 1), "azar_p95": float(np.percentile(pr, 95)),
                          "veces": round(int(hit.sum()) / max(pr.mean(), 1), 2), "p": round((1 + (pr >= hit.sum()).sum()) / (1 + N_PERM), 3),
                          "prob_media": round(float(h["prob"].mean()), 4) if len(h) else 0.0,
                          "prob_80_o_mas": int((h["prob"] >= 0.8).sum()) if len(h) else 0})
            h = h[h["prob"] > 0]
            if h.empty:
                continue
            # órdenes del año de esas personas en esa entidad mientras algún funcionario que calza tenía el cargo
            llaves = [("ap", "ap")] if patron == "mismo_paterno" else [("am", "ap"), ("ap", "am")]
            for lp, lf in llaves:
                mm = h[["ruc_entidad", "ruc", lp, "prob"]].merge(s[["ruc_entidad", lf, "inicio", "cese", "presentacion"]].rename(columns={lf: "_k"}),
                                                                 left_on=["ruc_entidad", lp], right_on=["ruc_entidad", "_k"])
                oo = o[o["anio"] == y][["id", "ruc_entidad", "ruc", "fecha"]].merge(mm[["ruc_entidad", "ruc", "prob", "inicio", "cese", "presentacion"]],
                                                                                  on=["ruc_entidad", "ruc"])
                oo = oo[(oo["fecha"] >= oo["inicio"]) & (oo["fecha"] <= oo["cese"])]
                alertas.append(oo.assign(patron=patron, anio=y)[["id", "ruc_entidad", "ruc", "anio", "patron", "prob", "presentacion"]])
        paso(f"   {y}: " + "; ".join(f"{x['patrón']} {x['observados']:,} observados frente a {x['azar_media']:,} por azar "
                                     f"({x['veces']} veces, p {x['p']})" for x in filas[-2:]))
    t = pd.DataFrame(filas)
    (V4 / "resultados" / "padre_hijo_azar.md").write_text(
        "# Padre o madre e hijo: observados frente al azar (permutación dentro del departamento, 100 por año)\n\n" + t.to_markdown(index=False),
        encoding="utf-8")
    a = pd.concat(alertas, ignore_index=True)
    a = a.groupby(["id", "ruc_entidad", "ruc", "anio"]).agg(prob=("prob", "max"), visible=("presentacion", "min")).reset_index()
    a = a.astype({"ruc_entidad": str, "ruc": str, "anio": int})
    paso(f"órdenes con alerta de padre o madre e hijo: {len(a):,}; con probabilidad de 50 % o más: {(a['prob'] >= 0.5).sum():,}; "
         f"visibles por año: {a['visible'].dt.year.value_counts().sort_index().to_dict()}")
    x = a[K].drop_duplicates().set_index(K)
    for y in ANIOS_DECISION:
        v = a[a["visible"] <= pd.Timestamp(f"{y}-12-31")].groupby(K)
        x[f"f_ph_prob_hasta_{y}"] = v["prob"].max()
        x[f"f_ph_ordenes_hasta_{y}"] = v["id"].size()
    x = x.fillna(0).reset_index()
    x.to_parquet(V4 / "x_padre_hijo.parquet", index=False)
    paso("casos con alerta visible por año de decisión: "
         + "; ".join(f"{y}: {int((x[f'f_ph_ordenes_hasta_{y}'] > 0).sum()):,}" for y in ANIOS_DECISION))
    d = pd.read_csv(V4 / "diccionario_x.csv")
    nuevas = ["f_padre_hijo_prob", "f_padre_hijo_ordenes"]
    d = d[~d["variable"].isin(nuevas)]
    fuente = "declaraciones juradas presentadas hasta el 31/12 del año de decisión; un apellido en posición de padre o madre e hijo (HALLAZGOS 136)"
    filas_d = [{"variable": c, "grupo": "alertas", "tipo": "proporción" if c.endswith("prob") else "conteo o monto",
                "clase_temporal": "familia a la fecha de decisión", "fuente": fuente, "lleno_pct": 100.0} for c in nuevas]
    pd.concat([d, pd.DataFrame(filas_d)], ignore_index=True).to_csv(V4 / "diccionario_x.csv", index=False)
    print(t.to_string(index=False))
    paso("listo")


if __name__ == "__main__":
    main()
