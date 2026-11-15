"""Capa 2, impedimentos que la alerta no veía (HALLAZGOS 148, aprobada por Pedro el 2/11).

Tres señales nuevas, todas a la fecha:
1. Empresa de un pariente de autoridad (literales h, i y k del art. 11): una empresa le vende a una entidad cuyo
   territorio gobierna una autoridad electa (JNE), mientras ejerce o hasta 12 meses después, y uno de sus socios,
   representantes u órganos (conformación del OECE del corte más reciente al 31/12 de ese año) tiene los mismos dos
   apellidos que esa autoridad, con un par raro (100 personas o menos en el padrón). Si el integrante es la autoridad
   misma, ya lo cubre «empresa de autoridad» y no se repite. Antes del primer corte (11/11/2021) se usa ese corte, con
   marca.
2. Empresa de un funcionario de la entidad (literales e, f, i y k): uno de sus integrantes tiene el mismo nombre completo
   que un funcionario con cargo de decisión de la entidad que compra, durante su cargo. Se hace visible con la primera
   declaración jurada presentada por ese funcionario (desde 2022), como la familia (132).
3. Funcionario propio (literales e y f): la persona natural que vende es un funcionario de la entidad durante su cargo
   (alerta de radar/02), visible con su primera declaración presentada.

Guarda por caso: t_empresa_pariente_aut_ordenes, t_empresa_pariente_aut_corte_posterior (clase «territorial a la
fecha»), y f_empfunc_ordenes_hasta_<Y> y f_funcprop_ordenes_hasta_<Y> (Y de 2022 a 2025), que `familia_a_la_decision`
convierte en f_empresa_funcionario_ordenes y f_funcionario_propio_ordenes según el año de decisión. Solo agregados.

    python -I -X utf8 capa2_impedidos.py
Salida: datos/procesados/v4/x_impedidos.parquet y filas en diccionario_x.csv.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
from comun import PROC, SALIDA, entidades, funcionarios, norm_serie  # noqa: E402
from capa2_familia import presentaciones  # noqa: E402

spec = importlib.util.spec_from_file_location("t25", AQUI.parent / "25_tarjeta_territorial.py")
T25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T25)

V4 = AQUI.parents[1] / "datos" / "procesados" / "v4"
K = ["ruc_entidad", "ruc", "anio"]
ANIOS = range(2018, 2026)
ANIOS_DECISION = range(2022, 2026)
RARO = 100
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def apellidos_de(nombres):
    u = pd.Series(nombres.unique())
    p = np.array([T25.apellidos(x) for x in u], dtype=object)
    return nombres.map(dict(zip(u, p[:, 0]))), nombres.map(dict(zip(u, p[:, 1])))


def main():
    e = entidades()[["RUC", "DEPARTAMENTO", "PROVINCIA", "DISTRITO"]].rename(columns={"RUC": "ruc_entidad"})
    e["eR"], e["eP"], e["eD"] = (norm_serie(e[c].fillna("")) for c in ("DEPARTAMENTO", "PROVINCIA", "DISTRITO"))
    o = pd.read_parquet(SALIDA / "ordenes.parquet", columns=["id", "ruc_entidad", "ruc", "nombre", "fecha", "monto"])
    o["anio"] = o["fecha"].dt.year
    emp = o[~o["ruc"].str.startswith("10")].merge(e[["ruc_entidad", "eR", "eP", "eD"]], on="ruc_entidad", how="left")
    au = T25.autoridades()
    au["par"] = au["ap"] + " " + au["am"]
    frec = pd.read_parquet(SALIDA / "padron_pares.parquet").groupby("par")["n"].sum()
    f = funcionarios()
    f = f.merge(presentaciones(), on=["persona", "ruc_entidad"], how="left")
    fd = f[f["decide"]]
    cortes = sorted((PROC / "acc3_conformacion").glob("conformacion_corte*.parquet"))
    fechas = {c: pd.Timestamp(c.stem[-8:]) for c in cortes}
    cols = ["dni_aut", "CARGO", "nivel", "R", "P", "D", "ini", "fin", "hasta", "todo_el_pais", "par"]
    aa = au.rename(columns={"eR": "R", "eP": "P", "eD": "D"})[cols]
    paso(f"órdenes a empresas: {len(emp):,}; autoridades: {len(au):,}; funcionarios de decisión con declaración: {len(fd):,}")
    pariente, empfunc = [], []
    cache = {}
    for y in ANIOS:
        fin = pd.Timestamp(f"{y}-12-31")
        validos = [c for c in cortes if fechas[c] <= fin]
        c = validos[-1] if validos else cortes[0]
        if c not in cache:
            s = pd.read_parquet(c, columns=["ruc_proveedor", "nombre_integrante", "dni_integrante", "usable_para_cruce"])
            s = s[s["usable_para_cruce"].fillna(False).astype(bool) & s["dni_integrante"].str.fullmatch(r"\d{8}", na=False)]
            s = s.rename(columns={"ruc_proveedor": "ruc"}).drop_duplicates(["ruc", "dni_integrante"])
            s["nom"] = norm_serie(s["nombre_integrante"].fillna(""))
            s["ap"], s["am"] = apellidos_de(s["nom"])
            s["par"] = s["ap"] + " " + s["am"]
            cache = {c: s}
        s = cache[c]
        oy = emp[emp["anio"] == y].merge(s[["ruc", "dni_integrante", "nom", "par"]], on="ruc")
        # 1) empresa de un pariente de autoridad: mismo par raro de apellidos en el territorio y el tiempo
        filas = []
        for nivel, llave in (("distrito", ["R", "P", "D"]), ("provincia", ["R", "P"]), ("region", ["R"])):
            q = aa[aa["nivel"] == nivel]
            k = oy.rename(columns={"eR": "R", "eP": "P", "eD": "D"}).merge(q, on=llave + ["par"], suffixes=("", "_a"))
            k = k[k["dni_integrante"] != k["dni_aut"]]
            k["eR"], k["eP"], k["eD"] = k["R"], k["P"], k["D"]
            for col in ("P", "D"):
                if col not in llave:
                    k[col] = k[f"{col}_a"]
            filas.append(T25.ventana(k))
        m = pd.concat(filas)
        m = m[(m["momento"] != "fuera_de_su_territorio_durante") & (m["par"].map(frec).fillna(0) <= RARO)]
        pariente.append(m[["id", "ruc_entidad", "ruc"]].drop_duplicates("id").assign(anio=y, corte_posterior=fechas[c] > fin))
        # 2) empresa de un funcionario de decisión de la entidad: mismo nombre completo, dentro del cargo
        g = oy.merge(fd[["persona", "ruc_entidad", "inicio", "cese", "presentacion"]], left_on=["nom", "ruc_entidad"],
                     right_on=["persona", "ruc_entidad"])
        g = g[(g["fecha"] >= g["inicio"]) & (g["fecha"] <= g["cese"])]
        empfunc.append(g.groupby("id").agg(ruc_entidad=("ruc_entidad", "first"), ruc=("ruc", "first"),
                                           visible=("presentacion", "min")).reset_index().assign(anio=y))
        paso(f"   {y}: corte {fechas[c].date()}{' (posterior)' if fechas[c] > fin else ''}; empresa de pariente de autoridad "
             f"{pariente[-1]['id'].nunique():,} órdenes; empresa de funcionario {empfunc[-1]['id'].nunique():,} órdenes")
    # 3) funcionario propio (alerta de radar/02) con su fecha de visibilidad
    fp = pd.read_parquet(SALIDA / "alerta_funcionario_propio.parquet", columns=["id"])
    fp = fp.merge(o[["id", "ruc_entidad", "ruc", "nombre", "fecha", "anio"]], on="id")
    fp["nom"] = norm_serie(fp["nombre"])
    fp = fp.merge(f[["persona", "ruc_entidad", "inicio", "cese", "presentacion"]], left_on=["nom", "ruc_entidad"],
                  right_on=["persona", "ruc_entidad"])
    fp = fp[(fp["fecha"] >= fp["inicio"]) & (fp["fecha"] <= fp["cese"])]
    fp = fp.groupby("id").agg(ruc_entidad=("ruc_entidad", "first"), ruc=("ruc", "first"), anio=("anio", "first"),
                              visible=("presentacion", "min")).reset_index()
    paso(f"funcionario propio con fecha de visibilidad: {len(fp):,} órdenes")
    p = pd.concat(pariente, ignore_index=True).astype({"ruc_entidad": str, "ruc": str, "anio": int})
    x = p.groupby(K).agg(t_empresa_pariente_aut_ordenes=("id", "nunique"), t_empresa_pariente_aut_corte_posterior=("corte_posterior", "max"))
    x["t_empresa_pariente_aut_corte_posterior"] = x["t_empresa_pariente_aut_corte_posterior"].astype(float)
    for nombre, d in (("empfunc", pd.concat(empfunc, ignore_index=True)), ("funcprop", fp)):
        d = d.astype({"ruc_entidad": str, "ruc": str, "anio": int})
        for yy in ANIOS_DECISION:
            v = d[d["visible"] <= pd.Timestamp(f"{yy}-12-31")].groupby(K)["id"].nunique().rename(f"f_{nombre}_ordenes_hasta_{yy}")
            x = x.join(v, how="outer")
    x = x.fillna(0).reset_index()
    x.to_parquet(V4 / "x_impedidos.parquet", index=False)
    paso("casos con señal nueva por año: " + str({a: int(((x["anio"] == a) & (x.drop(columns=K).sum(axis=1) > 0)).sum()) for a in ANIOS}))
    d = pd.read_csv(V4 / "diccionario_x.csv")
    nuevas = {"t_empresa_pariente_aut_ordenes": ("territorial a la fecha", "conformación del OECE (corte del año) y autoridades del JNE: par raro de apellidos (HALLAZGOS 148)"),
              "t_empresa_pariente_aut_corte_posterior": ("territorial a la fecha", "el corte de socios es posterior al año (2018 a 2020)"),
              "f_empresa_funcionario_ordenes": ("familia a la fecha de decisión", "conformación del OECE y declaraciones juradas presentadas hasta el año de decisión (HALLAZGOS 148)"),
              "f_funcionario_propio_ordenes": ("familia a la fecha de decisión", "radar/02 y declaraciones juradas presentadas hasta el año de decisión (HALLAZGOS 148)")}
    d = d[~d["variable"].isin(nuevas)]
    filas = [{"variable": k, "grupo": "alertas", "tipo": "binaria" if k.endswith("posterior") else "conteo o monto",
              "clase_temporal": v[0], "fuente": v[1], "lleno_pct": 100.0} for k, v in nuevas.items()]
    pd.concat([d, pd.DataFrame(filas)], ignore_index=True).to_csv(V4 / "diccionario_x.csv", index=False)
    paso("listo")


if __name__ == "__main__":
    main()
