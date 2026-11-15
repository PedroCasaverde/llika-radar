"""Probabilidad bayesiana de padre o madre e hijo con autoridades del JNE, con edad y sexo (HALLAZGOS 136).

`capa2_padre_hijo_jne.py` mostró que, con edad y sexo, las coincidencias superan al azar ~1,13 veces desde 2020 (sobre el
percentil 95 en 2020, 2021, 2023, 2024 y 2025). Aquí se reparte ese exceso entre los casos, como en radar/02:
- λ del caso = autoridades del territorio en ese nivel y año × frecuencia del apellido que calza entre los proveedores
  personas del departamento, encogida hacia la nacional (1 000 personas de peso); si calza con varias, el menor λ;
- probabilidad = r / (r + λ), con r tal que la suma iguala el exceso del año (observados − media del azar);
- solo en los años que superan el percentil 95; en los demás no hay alerta (2018, 2019 y 2022).
Queda por caso como t_padre_hijo_prob (clase «territorial a la fecha»: la lista del JNE y el DNI del RUC 10 son públicos
en su momento). Solo imprime agregados.

    python -I -X utf8 capa2_padre_hijo_jne_prob.py
Salida: datos/procesados/v4/x_padre_hijo_jne.parquet y una fila en diccionario_x.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
from comun import SALIDA, entidades, norm_serie  # noqa: E402
from capa2_padre_hijo_jne import T25, cohortes, prob_edad  # noqa: E402

V4 = AQUI.parents[1] / "datos" / "procesados" / "v4"
K = ["ruc_entidad", "ruc", "anio"]
M_PREVIO = 1000
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def detalle(pv, au, cortes, nacs):
    """Como capa2_padre_hijo_jne.aciertos, pero con el territorio, el nivel y el apellido que calza."""
    filas = []
    for nivel, llave in (("distrito", ["R", "P", "D"]), ("provincia", ["R", "P"])):
        q = au[au["nivel"] == nivel]
        izq = pv.rename(columns={"eR": "R", "eP": "P", "eD": "D"})
        for lp, la, genero, aut_mayor in [("ap", "ap", "M", True), ("am", "ap", "F", True), ("ap", "ap", None, False), ("ap", "am", None, False)]:
            qq = q if genero is None else q[q["GENERO"] == genero]
            if aut_mayor:
                m = izq.merge(qq[llave + ["ap", "am", "nac"]].rename(columns={"ap": "_k", "am": "_otro", "nac": "_nac"}),
                              left_on=llave + [lp], right_on=llave + ["_k"])
                herm = (m["ap"] == m["_k"]) & (m["am"] == m["_otro"])
            else:
                m = izq.merge(qq[llave + ["ap", "am", "nac"]].rename(columns={la: "_k", "ap" if la == "am" else "am": "_otro", "nac": "_nac"}),
                              left_on=llave + ["ap"], right_on=llave + ["_k"])
                herm = (m["ap"] == m["_k"]) & (m["am"] == m["_otro"]) if la == "ap" else pd.Series(False, index=m.index)
            m = m[~herm]
            if m.empty:
                continue
            w = prob_edad(m["_nac"].to_numpy(float), m["dni"].to_numpy(np.int64), cortes, nacs, aut_mayor)
            m = m[w >= 0.5]
            unidad = m[llave].astype(str).agg("|".join, axis=1)
            filas.append(pd.DataFrame({"i": m["i"].to_numpy(), "nivel": nivel, "unidad": unidad.to_numpy(), "apellido": m["_k"].to_numpy(),
                                       "depto": m["R"].to_numpy()}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["i", "nivel", "unidad", "apellido", "depto"])


def main():
    L = [f for f in (V4 / "resultados" / "padre_hijo_jne_azar.md").read_text(encoding="utf-8").splitlines() if f.startswith("|") and "---" not in f]
    cab = [c.strip() for c in L[0].strip("|").split("|")]
    tabla = pd.DataFrame([[c.strip() for c in f.strip("|").split("|")] for f in L[1:]], columns=cab)
    tabla = tabla.astype({"año": int, "observados": float, "azar_media": float})
    tabla["supera_p95"] = tabla["supera_p95"] == "True"
    exceso = {int(r["año"]): r["observados"] - r["azar_media"] for _, r in tabla.iterrows() if r["supera_p95"]}
    paso(f"años con señal y su exceso: { {k: round(v) for k, v in exceso.items()} }")
    cortes, nacs = cohortes()
    au = T25.autoridades().rename(columns={"eR": "R", "eP": "P", "eD": "D"})
    au["nac"] = pd.to_numeric(au["ANIOELECCION"], errors="coerce") - pd.to_numeric(au["EDAD"], errors="coerce")
    au = au.dropna(subset=["nac"])
    au = au[au["nivel"].isin(["distrito", "provincia"])]
    e = entidades()[["RUC", "DEPARTAMENTO", "PROVINCIA", "DISTRITO"]].rename(columns={"RUC": "ruc_entidad"})
    e["eR"], e["eP"], e["eD"] = (norm_serie(e[c].fillna("")) for c in ("DEPARTAMENTO", "PROVINCIA", "DISTRITO"))
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["ruc_entidad", "ruc", "nombre", "anio"])
    o = o[o["ruc"].str.startswith("10")].drop_duplicates(["ruc_entidad", "ruc", "anio"]).merge(e, on="ruc_entidad", how="left")
    o["nombre"] = norm_serie(o["nombre"])
    nombres = pd.Series(o["nombre"].unique())
    partes = np.array([T25.apellidos(x) for x in nombres], dtype=object)
    o["ap"], o["am"] = o["nombre"].map(dict(zip(nombres, partes[:, 0]))), o["nombre"].map(dict(zip(nombres, partes[:, 1])))
    o["dni"] = o["ruc"].str[2:10].astype(np.int64)
    per = o.drop_duplicates("ruc")[["ruc", "eR", "ap"]]
    p_nac = per.groupby("ap").size() / len(per)
    n_loc, n_dep = per.groupby(["eR", "ap"]).size(), per.groupby("eR").size()
    salidas = []
    for y, ex in exceso.items():
        ini, fin = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        a = au[(au["ini"] <= fin) & (au["hasta"] >= ini)].drop_duplicates(["dni_aut", "nivel", "R", "P", "D"]).reset_index(drop=True)
        pv = o[o["anio"] == y].reset_index(drop=True).assign(i=lambda x: np.arange(len(x)))
        h = detalle(pv, a, cortes, nacs)
        n_aut = pd.concat([a[a["nivel"] == "distrito"][["R", "P", "D"]].astype(str).agg("|".join, axis=1).value_counts(),
                           a[a["nivel"] == "provincia"][["R", "P"]].astype(str).agg("|".join, axis=1).value_counts()])
        nl = np.array([n_loc.get((d, s), 0) for d, s in zip(h["depto"], h["apellido"])], dtype=float)
        pn = h["apellido"].map(p_nac).fillna(1 / len(per)).to_numpy()
        h["lam"] = h["unidad"].map(n_aut).fillna(1).to_numpy() * (nl + M_PREVIO * pn) / (h["depto"].map(n_dep).fillna(0).to_numpy() + M_PREVIO)
        h = h.groupby("i")["lam"].min().clip(lower=1e-12)
        r = brentq(lambda r: (r / (r + h.to_numpy())).sum() - ex, 1e-12, 1e4)
        prob = r / (r + h)
        casos = pv.loc[prob.index, ["ruc_entidad", "ruc", "anio"]].assign(t_padre_hijo_prob=prob.to_numpy())
        salidas.append(casos)
        paso(f"   {y}: {len(h):,} casos que calzan; exceso {ex:,.0f}; probabilidad media {prob.mean():.3f}; con 50 % o más: {(prob >= 0.5).sum():,}")
    x = pd.concat(salidas, ignore_index=True).astype({"ruc_entidad": str, "ruc": str, "anio": int})
    x.to_parquet(V4 / "x_padre_hijo_jne.parquet", index=False)
    d = pd.read_csv(V4 / "diccionario_x.csv")
    d = d[d["variable"] != "t_padre_hijo_prob"]
    d = pd.concat([d, pd.DataFrame([{"variable": "t_padre_hijo_prob", "grupo": "alertas", "tipo": "proporción",
                                     "clase_temporal": "territorial a la fecha",
                                     "fuente": "JNE (DNI, género y edad) y DNI del RUC 10; padre o madre e hijo con edad y sexo (HALLAZGOS 136)",
                                     "lleno_pct": 100.0}])], ignore_index=True)
    d.to_csv(V4 / "diccionario_x.csv", index=False)
    paso(f"casos con probabilidad por año: {x.groupby('anio').size().to_dict()}; listo")


if __name__ == "__main__":
    main()
