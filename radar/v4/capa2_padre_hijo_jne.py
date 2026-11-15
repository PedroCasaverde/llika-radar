"""Padre o madre e hijo con autoridades electas, con edad y sexo (HALLAZGOS 136; Pedro: «tendrás en cuenta las edades,
por sentido común»).

Con las declaraciones juradas, un solo apellido no se distinguió del azar, y ahí no hay edad ni sexo del funcionario.
La lista del JNE sí trae DNI, género y edad al elegirse, así que aquí se exige además que la edad cuadre:
- año de nacimiento de la autoridad = año de la elección − edad;
- del proveedor (RUC 10) solo se conoce el DNI; su año de nacimiento se aproxima con la distribución del año de
  nacimiento de las autoridades que tienen un DNI del mismo decil (Spearman 0,80 entre DNI y año de nacimiento en 38 117
  autoridades): los DNI bajos son de nacidos antes de 1977 y los de 45 millones en adelante, de nacidos hacia 1990;
- un padre o una madre le lleva al hijo entre 18 y 50 años; se calcula la probabilidad de que la diferencia caiga ahí.

Patrones (se excluyen los que coinciden en los dos apellidos, que son la tarjeta de pariente de radar/25):
- la autoridad es su padre: autoridad varón, paterno del proveedor = paterno de la autoridad, autoridad mayor;
- la autoridad es su madre: autoridad mujer, materno del proveedor = paterno de la autoridad, autoridad mayor;
- el proveedor es padre o madre de la autoridad: paterno de la autoridad = paterno del proveedor, o materno de la
  autoridad = paterno del proveedor, proveedor mayor.
Cuenta como coincidencia si la probabilidad de edad es 0,5 o más. Territorio: distrito (alcalde y regidores distritales)
y provincia (provinciales), en su mandato o hasta 12 meses después, como la tarjeta territorial.
Prueba: los mismos conteos con 100 permutaciones por año del territorio de las autoridades dentro de su departamento y
nivel. Se emite solo si lo observado supera el percentil 95 del azar (la vara de la H1). Solo imprime agregados.

    python -I -X utf8 capa2_padre_hijo_jne.py
Salida: resultados/padre_hijo_jne_azar.md y, si hay señal, datos/procesados/v4/x_padre_hijo_jne.parquet.
"""
import importlib.util
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from comun import ANIOS, CRUDOS, SALIDA, entidades, norm_serie  # noqa: E402

spec = importlib.util.spec_from_file_location("t25", AQUI.parent / "25_tarjeta_territorial.py")
T25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T25)

V4 = AQUI.parents[1] / "datos" / "procesados" / "v4"
N_PERM, BRECHA = 100, (18, 50)
PROCESOS = 5
G = {}
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def cohortes():
    """Decil de DNI → años de nacimiento observados en las autoridades de ese decil (todas, no solo locales)."""
    a = pd.read_excel(CRUDOS / "jne_autoridades" / "Autoridades_Electas_20251113.xls", dtype=str)
    a = a[a["DOCUMENTOIDENTIDAD"].str.fullmatch(r"\d{8}", na=False)].copy()
    a["dni"] = a["DOCUMENTOIDENTIDAD"].astype(np.int64)
    a["nac"] = pd.to_numeric(a["ANIOELECCION"], errors="coerce") - pd.to_numeric(a["EDAD"], errors="coerce")
    a = a.dropna(subset=["nac"]).drop_duplicates("dni")
    cortes = np.unique(np.quantile(a["dni"], np.linspace(0, 1, 11)))
    a["d"] = np.clip(np.searchsorted(cortes, a["dni"], side="right") - 1, 0, len(cortes) - 2)
    return cortes, {d: g["nac"].to_numpy() for d, g in a.groupby("d")}


def prob_edad(nac_aut, dni_prov, cortes, nacs, autoridad_mayor):
    """Probabilidad de que la diferencia de edad caiga entre 18 y 50 años en el sentido pedido."""
    d = np.clip(np.searchsorted(cortes, dni_prov, side="right") - 1, 0, len(cortes) - 2)
    out = np.zeros(len(d))
    for k in np.unique(d):
        m = d == k
        b = nacs[k]
        dif = (b[None, :] - nac_aut[m][:, None]) if autoridad_mayor else (nac_aut[m][:, None] - b[None, :])
        out[m] = ((dif >= BRECHA[0]) & (dif <= BRECHA[1])).mean(axis=1)
    return out


def aciertos(pv, au, cortes, nacs):
    """Proveedor-entidad con alguna autoridad del territorio que cuadra como padre o madre (con edad y sexo)."""
    hits = []
    for nivel, llave in (("distrito", ["R", "P", "D"]), ("provincia", ["R", "P"])):
        q = au[au["nivel"] == nivel]
        izq = pv.rename(columns={"eR": "R", "eP": "P", "eD": "D"})
        reglas = [("ap", "ap", "M", True), ("am", "ap", "F", True), ("ap", "ap", None, False), ("ap", "am", None, False)]
        for lp, la, genero, aut_mayor in reglas:
            qq = q if genero is None else q[q["GENERO"] == genero]
            if aut_mayor:          # la autoridad es el padre o la madre: el apellido del proveedor (lp) = paterno de la autoridad
                m = izq.merge(qq[llave + ["ap", "am", "nac"]].rename(columns={"ap": "_k", "am": "_am_a", "nac": "_nac"}),
                              left_on=llave + [lp], right_on=llave + ["_k"])
                herm = (m["ap"] == m["_k"]) & (m["am"] == m["_am_a"])
            else:                  # el proveedor es el padre o la madre: el paterno del proveedor = apellido de la autoridad (la)
                m = izq.merge(qq[llave + ["ap", "am", "nac"]].rename(columns={la: "_k", "ap" if la == "am" else "am": "_otro", "nac": "_nac"}),
                              left_on=llave + ["ap"], right_on=llave + ["_k"])
                herm = (m["ap"] == m["_k"]) & (m["am"] == m["_otro"]) if la == "ap" else pd.Series(False, index=m.index)
            m = m[~herm]
            if m.empty:
                continue
            w = prob_edad(m["_nac"].to_numpy(float), m["dni"].to_numpy(np.int64), cortes, nacs, aut_mayor)
            hits.append(m.loc[w >= 0.5, ["i"]].assign(w=w[w >= 0.5]))
    return pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(columns=["i", "w"])


def iniciar(pv, a, cortes, nacs, y):
    G.update(pv=pv, a=a, cortes=cortes, nacs=nacs, y=y)


def una_permutacion(k):
    rng = np.random.default_rng(k + 1000 * G["y"])
    b = G["a"].copy()
    for (nivel, r), g in b.groupby(["nivel", "R"]).groups.items():
        cols = ["P", "D"] if nivel == "distrito" else ["P"]
        b.loc[g, cols] = b.loc[g, cols].to_numpy()[rng.permutation(len(g))]
    return aciertos(G["pv"], b, G["cortes"], G["nacs"])["i"].nunique()


def main():
    cortes, nacs = cohortes()
    paso(f"deciles de DNI y nacimiento de las autoridades: medianas {[int(np.median(nacs[k])) for k in sorted(nacs)]}")
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
    paso(f"relaciones de personas naturales por año: {len(o):,}; autoridades distritales y provinciales: {len(au):,}")
    filas = []
    for y in ANIOS:
        ini, fin = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        a = au[(au["ini"] <= fin) & (au["hasta"] >= ini)].drop_duplicates(["dni_aut", "nivel", "R", "P", "D"]).reset_index(drop=True)
        pv = o[o["anio"] == y].reset_index(drop=True).assign(i=lambda x: np.arange(len(x)))
        obs = aciertos(pv, a, cortes, nacs)["i"].nunique()
        # cada permutación tiene su semilla: repartirlas da exactamente lo mismo que en serie (160, sección 12)
        with get_context("spawn").Pool(PROCESOS, initializer=iniciar, initargs=(pv, a, cortes, nacs, y)) as pool:
            perm = np.array(pool.map(una_permutacion, range(N_PERM)))
        filas.append({"año": y, "autoridades": len(a), "observados": int(obs), "azar_media": round(perm.mean(), 1),
                      "azar_p95": float(np.percentile(perm, 95)), "veces": round(obs / max(perm.mean(), 1), 2),
                      "p": round((1 + (perm >= obs).sum()) / (1 + N_PERM), 3)})
        paso(f"   {y}: {obs:,} observados frente a {perm.mean():,.1f} por azar ({filas[-1]['veces']} veces, p {filas[-1]['p']})")
    t = pd.DataFrame(filas)
    t["supera_p95"] = t["observados"] > t["azar_p95"]
    (V4 / "resultados" / "padre_hijo_jne_azar.md").write_text(
        "# Padre o madre e hijo con autoridades del JNE, con edad y sexo: observados frente al azar\n\n" + t.to_markdown(index=False),
        encoding="utf-8")
    print(t.to_string(index=False))
    paso(f"años que superan el percentil 95 del azar: {t.loc[t['supera_p95'], 'año'].tolist() or 'ninguno'}")


if __name__ == "__main__":
    main()
