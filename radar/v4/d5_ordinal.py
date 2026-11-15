"""Ronda completa con etiquetas ordinales por gravedad (HALLAZGOS 137; Pedro: «terminada con la etiqueta amplia, das una
ronda completa con la versión 137»). Configuración fijada antes de correr.

Nivel de cada caso (el más grave que tenga, a la fecha, etiqueta amplia de 130):
    3 = hallazgo del control o impedimento sancionado (A, B o C núcleo)
    2 = sanción del Tribunal por otra causa (S)
    1 = penalidad o multa (P)
    0 = ninguna
Modelo ordinal de Frank y Hall (2001): un modelo sí/no por umbral (nivel ≥ 1, ≥ 2 y = 3), cada uno con sus positivos.
Puntaje = gravedad esperada = P(≥1) + P(≥2) + P(≥3).

Candidatas, todas con la tarjeta territorial y la familia a la fecha (128 y 132):
- D1_ord y D1_ord_grupo: boosting de D1 en cada umbral, global y con el ajuste por tipo de entidad de 131;
- A_ord y A_ord_grupo: suma ponderada en cada umbral, global y con pesos por grupo de 131;
- B_ord: A_ord más las variables de la red de PASCO en los umbrales que pasan la compuerta de homofilia (con sus etiquetas);
- C_ord: B_ord con MaskCluster solo en el umbral 3, el de pocos positivos (maestro sin monto, a lo más 0,3 % por
  comunidad y año con medio peso, promedio de maestro y alumno, como en 127);
- referencias: monto, tamaño del proveedor en el Estado (135) y D1 binario con la etiqueta amplia (130).
Cortes 2020, 2021 y 2023; 2022 sellado.

    python -I -X utf8 d5_ordinal.py
Salida: resultados/d5_ordinal.json y .md, y puntajes de la prueba en resultados/puntajes_d5_<año>.parquet.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_linea_base import CORTES, HIST, d1  # noqa: E402
from d3_capas import MONTO, PESO_PSEUDO, RED_COLS, TOPE, Lineal, auc_dentro  # noqa: E402
from d4_grupos import LinealGrupos, d1_y_grupos, grupo  # noqa: E402
from etiquetas import Hechos, K  # noqa: E402
from evaluacion import armar_corte, columnas, diferencia_conjunta, metricas, metricas_k, razon_conjunta  # noqa: E402

warnings.filterwarnings("ignore")
V4 = Path(__file__).resolve().parents[2] / "datos" / "procesados" / "v4"
OUT, RED = V4 / "resultados", V4 / "red"
UMBRALES = (1, 2, 3)
TAREAS = {"amplia": "y", "nivel_2_o_mas": "y_n2", "estricta_nivel_3": "y_estricta", "revision_AB": "y_rev",
          "impedimentos_C": "y_imp", "proveedor_SP": "y_sp"}
PRUEBA = "--prueba" in sys.argv
T0 = time.time()


def paso(t):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {t}", flush=True)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def nivel(fuentes):
    f = fuentes.fillna("").astype(str)
    return np.select([f.str.contains("A|B|C"), f.str.contains("S"), f.str.contains("P")], [3, 2, 1], 0)


def peso_umbral(tr, yk):
    return np.where(yk, tr["peso"].clip(lower=0.25), 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H = Hechos(conocimiento="publicacion")
    H.etiqueta = H.etiqueta_amplia
    cols = columnas() + HIST + columnas(clases=("territorial a la fecha",)) + columnas(clases=("familia a la fecha de decisión",))
    paso(f"etiqueta ordinal (3 hallazgo o impedimento, 2 sanción, 1 penalidad); {len(cols)} variables")
    red = pd.concat([pd.read_parquet(RED / f"comunidades_{a}.parquet") for a in (2018, 2019, 2020, 2021, 2023)], ignore_index=True)
    red = red.astype({"ruc_entidad": str, "ruc": str, "anio": int})
    res, puntajes = {"cortes": []}, {}
    for tr_a, te_a in CORTES:
        paso(f"armando el corte {tr_a} → {te_a}")
        tr, te = armar_corte(H, tr_a, te_a)
        if PRUEBA:                                   # prueba de humo: 5 % de las entidades, solo el primer corte
            ents = set(te["ruc_entidad"].drop_duplicates().sample(frac=0.05, random_state=0))
            tr, te = tr[tr["ruc_entidad"].isin(ents)].reset_index(drop=True), te[te["ruc_entidad"].isin(ents)].reset_index(drop=True)
        tr = tr.astype({"ruc_entidad": str, "ruc": str, "anio": int}).merge(red, on=K, how="left")
        te = te.astype({"ruc_entidad": str, "ruc": str, "anio": int}).merge(red, on=K, how="left")
        tr["nivel"], te["nivel"] = nivel(tr["fuentes"]), nivel(te["fuentes"])
        fu = te["fuentes"].fillna("").astype(str)
        te["y_n2"] = te["nivel"] >= 2
        te["y_rev"], te["y_imp"], te["y_sp"] = (fu.str.contains(p).to_numpy() for p in ("A|B", "C", "S|P"))
        te["con_antecedentes"] = (te[[c for c in HIST if c.startswith("h_prov_")]] > 0).any(axis=1).to_numpy()
        g_tr, g_te = grupo(tr), grupo(te)
        paso(f"   entrenamiento {len(tr):,}, por nivel {pd.Series(tr['nivel']).value_counts().sort_index().to_dict()}; "
             f"prueba {len(te):,}, por nivel {pd.Series(te['nivel']).value_counts().sort_index().to_dict()}")
        s = {"monto": te["c_monto"].to_numpy(dtype=float),
             "tamano_proveedor": np.log1p(te[["p_monto", "p_exp_ocds_monto", "p_cat_monto"]].fillna(0).clip(lower=0).sum(axis=1)).to_numpy()}
        _, s["D1_binario"], _ = d1(tr, te, cols, valid_anio=tr_a[-1])
        p = {n: np.zeros(len(te)) for n in ("D1_ord", "D1_ord_grupo", "A_ord", "A_ord_grupo", "B_ord", "C_ord")}
        compuerta = {}
        for k in UMBRALES:
            yk = (tr["nivel"] >= k).to_numpy()
            wk = peso_umbral(tr, yk)
            trk = tr.assign(y=yk)
            sg, sgr, it, _ = d1_y_grupos(trk, te, cols, tr_a[-1], g_tr, g_te)
            p["D1_ord"] += sig(sg)
            p["D1_ord_grupo"] += sig(sgr)
            A = Lineal(tr, cols, yk, wk)
            sa = A.puntaje(te)
            p["A_ord"] += sig(sa)
            p["A_ord_grupo"] += sig(LinealGrupos(tr, cols, yk, wk, g_tr).puntaje(te, g_te))
            auc, ic = auc_dentro(yk, tr["r_com_hist_otras"].fillna(0).to_numpy(), tr["grupo"].to_numpy(), tr["ruc_entidad"].to_numpy())
            pasa = bool(ic[0] > 0.5)
            compuerta[k] = {"auc": round(auc, 4), "ic90": [round(v, 4) for v in ic], "pasa": pasa}
            cols_b = cols + RED_COLS if pasa else cols
            Bk = Lineal(tr, cols_b, yk, wk) if pasa else A
            sb = Bk.puntaje(te) if pasa else sa
            p["B_ord"] += sig(sb)
            if k == 3:                    # MaskCluster solo donde faltan positivos
                maestro = Lineal(tr, [c for c in cols_b if not c.startswith(MONTO)], yk, wk)
                sm = maestro.puntaje(tr)
                d = pd.DataFrame({"c": tr["comunidad"].fillna(-1).astype(int).to_numpy(), "a": tr["anio"].to_numpy(), "s": sm, "y": yk})
                pseudo = np.zeros(len(tr), dtype=bool)
                for _, gg in d.groupby(["c", "a"]):
                    pseudo[gg[~gg["y"]].nlargest(max(1, int(TOPE * len(gg))), "s").index] = True
                alumno = Lineal(tr, cols_b, yk | pseudo, np.where(yk, wk, np.where(pseudo, PESO_PSEUDO, 1.0)))
                p["C_ord"] += sig(0.5 * sb + 0.5 * alumno.puntaje(te))
                paso(f"   umbral 3: MaskCluster propuso {int(pseudo.sum()):,} frente a {int(yk.sum()):,} reales")
            else:
                p["C_ord"] += sig(sb)
            paso(f"   umbral ≥{k}: {int(yk.sum()):,} positivos de entrenamiento; D1 {it} árboles; compuerta de la red "
                 f"{compuerta[k]['auc']} {'pasa' if pasa else 'no pasa'}")
        s.update(p)
        fila = {"entrena": tr_a, "prueba": te_a, "compuerta_por_umbral": compuerta,
                "niveles_prueba": pd.Series(te["nivel"]).value_counts().sort_index().to_dict()}
        for n, v in s.items():
            fila[n] = {k_: round(x, 4) if isinstance(x, float) else x for k_, x in metricas(te, v).items()}
            fila[n]["precision_k"] = {t: {k_: round(x, 4) for k_, x in metricas_k(te, v, y=c).items()} for t, c in TAREAS.items()}
            top = te.assign(_s=v).sort_values(["grupo", "_s"], ascending=[True, False]).groupby("grupo").head(5)
            ac = top[top["y"]]
            fila[n]["aciertos_top5_sin_antecedentes"] = round(float((~ac["con_antecedentes"]).mean()), 4) if len(ac) else None
            pk = fila[n]["precision_k"]
            paso(f"   {n:<18} P@5 amplia {pk['amplia']['P@5_ent']:.4f} | nivel 3 {pk['estricta_nivel_3']['P@5_ent']:.4f} | "
                 f"revisión {pk['revision_AB']['P@5_ent']:.4f} | MAP amplia {fila[n]['MAP_ent']:.4f}")
        res["cortes"].append(fila)
        if PRUEBA:
            paso("prueba de humo terminada (no se guarda)")
            return
        puntajes[te_a] = (te[["grupo", "y", "y_n2", "y_estricta", "y_rev", "y_imp", "y_sp", "c_monto"]].copy(), s)
        te[["grupo", "ruc_entidad", "ruc", "anio", "nivel", "y", "y_n2", "y_estricta", "y_rev", "y_imp", "y_sp", "con_antecedentes", "c_monto"]] \
            .assign(**{f"s_{n}": v for n, v in s.items()}).to_parquet(OUT / f"puntajes_d5_{te_a}.parquet", index=False)
    paso("diferencias y razones conjuntas (IC al 90 %)")
    candidatas = ["D1_binario", "D1_ord", "D1_ord_grupo", "A_ord", "A_ord_grupo", "B_ord", "C_ord"]
    res["comparaciones"] = {}
    for a in candidatas:
        for b in ("monto", "tamano_proveedor"):
            res["comparaciones"][f"{a} − {b}"] = diferencia_conjunta([(t, s_[a], s_[b]) for t, s_ in puntajes.values()])
    for a, b in (("D1_ord", "D1_binario"), ("D1_ord_grupo", "D1_ord"), ("A_ord_grupo", "A_ord"), ("B_ord", "A_ord"), ("C_ord", "B_ord")):
        res["comparaciones"][f"{a} − {b}"] = diferencia_conjunta([(t, s_[a], s_[b]) for t, s_ in puntajes.values()])
    for k_, d in res["comparaciones"].items():
        print(f"   {k_}: {d}", flush=True)
    res["razones"] = {}
    for n in candidatas:
        for ref in ("monto", "tamano_proveedor"):
            for t in (("amplia", "estricta_nivel_3", "revision_AB", "proveedor_SP") if ref == "monto" else ("amplia", "estricta_nivel_3")):
                for medida, q in (("P@5", False), ("AP", True)):
                    clave = f"{n} / {ref} | {t} | {medida}{' en rangos de monto' if q else ''}"
                    r = razon_conjunta([(t_, s_[n], s_[ref]) for t_, s_ in puntajes.values()], medida=medida, quintil=q, y=TAREAS[t])
                    res["razones"][clave] = r
                    print(f"   {clave:<70} {r['modelo']:.4f} / {r['monto']:.4f} = {r['razon']:.2f} [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}]", flush=True)
    (OUT / "d5_ordinal.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# Ronda ordinal por gravedad (137): precisión en 5 por tarea", "",
         "| Corte | Puntaje | Amplia | Nivel 2 o más | Nivel 3 (estricta) | Revisión (A o B) | Impedimentos (C) | Proveedor (S o P) | MAP amplia | Aciertos sin antecedentes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for f in res["cortes"]:
        for n in ["monto", "tamano_proveedor"] + candidatas:
            pk = f[n]["precision_k"]
            L.append(f"| {f['prueba']} | {n} | " + " | ".join(f"{pk[t]['P@5_ent']:.4f}" for t in TAREAS) +
                     f" | {f[n]['MAP_ent']:.4f} | {f[n]['aciertos_top5_sin_antecedentes']} |")
    L += ["", "| Comparación | Diferencia de MAP (amplia) | IC al 90 % | Por corte |", "|---|---|---|---|"]
    for k_, d in res["comparaciones"].items():
        L.append(f"| {k_} | {d['dif_MAP']:+.4f} | [{d['ic90'][0]:+.4f}, {d['ic90'][1]:+.4f}] | {d['por_corte']} |")
    L += ["", "| Candidata / referencia, tarea y medida | Candidata | Referencia | Razón | IC al 90 % |", "|---|---|---|---|---|"]
    for k_, r in res["razones"].items():
        L.append(f"| {k_} | {r['modelo']:.4f} | {r['monto']:.4f} | {r['razon']:.2f} | [{r['ic90'][0]:.2f}, {r['ic90'][1]:.2f}] |")
    (OUT / "d5_ordinal.md").write_text("\n".join(L), encoding="utf-8")
    paso("listo")


if __name__ == "__main__":
    main()
