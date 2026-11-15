"""Data v4, capa 3: fábrica de etiquetas y de historia conocida (HALLAZGOS 118). Todo lo que depende de los hechos sale
de aquí, siempre «a la fecha». El sellado vive aquí y en ningún otro lugar.

Hechos que cuentan como etiqueta (HALLAZGOS 114 a 117):
- A: orden citada en un informe; solo si la orden es de una entidad candidata del informe (entidad_coincide).
- C núcleo: resolución sancionadora con la orden hallada; el año del hecho es el de esa orden.
- B: el informe nombra al proveedor como beneficiado en esa entidad. Se reparte entre los años del periodo del hecho en
  los que la relación tuvo órdenes (soporte); cada año recibe la fracción 1/k (peso), nunca en proporción al monto.
- La C sin orden queda solo como antecedente (historia), no como etiqueta.
Fecha de conocimiento: la publicación (por defecto: lo que ve cualquiera en la web) o la emisión (lo que sabría la
propia Contraloría), a elegir.

Uso típico:
    from etiquetas import Hechos
    H = Hechos(conocimiento="publicacion")                 # abrir_sellado=True solo en la corrida final pre-registrada
    y = H.etiqueta(casos, fecha_corte="2020-12-31", horizonte=0)
    hist = H.historia(casos)                               # variables de historia conocida al cierre del año de cada fila
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from comun import PROC  # noqa: E402

BASE = PROC / "v4" / "base"
SELLADO_ANIO = 2022
SELLADO_DESDE = pd.Timestamp("2026-01-01")
HOY_EXPLORACION = pd.Timestamp("2025-12-31")      # en exploración, lo más que se sabe de la prueba
K = ["ruc_entidad", "ruc", "anio"]
MAX_ANIOS_B = 4
VENTANA_ANIOS = 2         # etiqueta amplia: lo conocido hasta dos años después del cierre del año del hecho (HALLAZGOS 160, C5)


def cierre(anios):
    return pd.to_datetime(pd.Series(anios).astype(int).astype(str) + "-12-31").to_numpy()


def dentro_de_ventana(e, anios):
    """Hechos conocidos a más tardar `anios` años después del cierre del año de los hechos."""
    return e[e["fecha_conocido"].to_numpy() <= cierre(e["anio_hechos"].to_numpy() + anios)]


class Hechos:
    def __init__(self, conocimiento="publicacion", abrir_sellado=False, fuentes=("A", "B", "C"), sin_dudosas=False, sin_acreedores=False):
        """sin_dudosas y sin_acreedores son sensibilidades: quitan los cruces A marcados y los acreedores de B (160, C7 y C8)."""
        assert conocimiento in ("publicacion", "emision")
        self.conocimiento, self.abrir_sellado, self.fuentes = conocimiento, abrir_sellado, tuple(fuentes)
        self.sin_dudosas, self.sin_acreedores = sin_dudosas, sin_acreedores
        inf = pd.read_parquet(BASE / "informes.parquet", columns=["informe", "fecha_emision", "fecha_publicacion"]).set_index("informe")
        res = pd.read_parquet(BASE / "resoluciones.parquet", columns=["resolucion", "fecha_resolucion", "fecha_publicacion"])
        res = res.drop_duplicates("resolucion").set_index("resolucion")
        f_inf = inf["fecha_publicacion"].fillna(inf["fecha_emision"]) if conocimiento == "publicacion" else inf["fecha_emision"]
        f_res = res["fecha_publicacion"] if conocimiento == "publicacion" else res["fecha_resolucion"]
        ho = pd.read_parquet(BASE / "hechos_orden.parquet")
        A = ho[(ho["fuente"] == "A") & ho["entidad_coincide"].astype(bool)].copy()
        if sin_dudosas:
            A = A[~A["dudosa"].fillna(False).astype(bool)]
        A["fecha_conocido"] = A["documento"].map(f_inf)
        C = ho[ho["fuente"] == "C"].copy()
        C["fecha_conocido"] = C["documento"].map(f_res)
        orden = pd.concat([A, C])[["fuente", "documento", "hecho", "ruc_entidad", "ruc", "anio", "fecha_conocido", "es_impedimento"]]
        orden = orden.rename(columns={"anio": "anio_hechos"}).assign(peso=1.0)
        hr = pd.read_parquet(BASE / "hechos_relacion.parquet")
        b = hr[hr["fuente"] == "B"].dropna(subset=["ruc_entidad", "desde"]).copy()
        if sin_acreedores:
            b = b[~b["acreedor"].astype(bool)]
        b["d"] = b["desde"].dt.year.astype(int)
        b["h"] = b["hasta"].dt.year.fillna(b["d"]).astype(int).clip(upper=b["d"] + MAX_ANIOS_B - 1)
        o = pd.read_parquet(BASE / "ordenes.parquet", columns=["ruc_entidad", "ruc", "anio"]).drop_duplicates()
        soporte = set(zip(o["ruc_entidad"], o["ruc"], o["anio"]))
        filas = []
        for doc, hecho, e, r, d, h in zip(b["documento"], b["hecho"], b["ruc_entidad"], b["ruc"], b["d"], b["h"]):
            anios = [a for a in range(d, h + 1) if (e, r, a) in soporte]          # solo años con compras de la relación
            for a in anios:
                filas.append((doc, hecho, e, r, a, 1.0 / len(anios)))
        B = pd.DataFrame(filas, columns=["documento", "hecho", "ruc_entidad", "ruc", "anio_hechos", "peso"]).assign(fuente="B", es_impedimento=False)
        B["fecha_conocido"] = B["documento"].map(f_inf)
        ev = pd.concat([orden, B], ignore_index=True).dropna(subset=["fecha_conocido"])
        ev["es_impedimento"] = ev["es_impedimento"].fillna(False).astype(bool)          # los hechos A y B no son impedimentos
        ev = ev[ev["fuente"].isin(self.fuentes)]
        ev["anio_hechos"] = ev["anio_hechos"].astype(int)
        self.sin_sellar = ev
        self.ev = ev if abrir_sellado else ev[(ev["anio_hechos"] != SELLADO_ANIO) & (ev["fecha_conocido"] < SELLADO_DESDE)]
        # antecedentes que no son etiqueta (sanción sin orden, sanciones del registro), para la historia
        cs = hr[hr["fuente"] == "C_sin_orden"].copy()
        cs["fecha_conocido"] = cs["documento"].map(f_res)
        cs["anio_hechos"] = cs["desde"].dt.year
        self.antecedentes = cs if abrir_sellado else cs[(cs["anio_hechos"] != SELLADO_ANIO) & (cs["fecha_conocido"] < SELLADO_DESDE)]
        # registros que no son nuestras etiquetas pero sí historia fechada: penalidades y multas (eventos del paso 29) e
        # inhabilitaciones del registro de sanciones (inicio y fin)
        rg = pd.read_parquet(PROC / "benchmark" / "eventos.parquet", columns=["ruc", "tipo", "fecha_hechos", "fecha_conocido"])
        rg = rg[rg["tipo"].isin(["penalidad", "multa"])].copy()
        rg["anio_hechos"] = pd.to_datetime(rg["fecha_hechos"], errors="coerce").dt.year
        self.registros = rg if abrir_sellado else rg[(rg["anio_hechos"] != SELLADO_ANIO) & (rg["fecha_conocido"] < SELLADO_DESDE)]
        s = pd.read_parquet(BASE / "sanciones.parquet", columns=["ruc", "inicio", "fin"]).dropna(subset=["inicio"])
        self.sanciones = s if abrir_sellado else s[s["inicio"] < SELLADO_DESDE]
        # etiqueta amplia, «señal adversa documentada» (HALLAZGOS 130 y 160, C5 y C6): sanción del Tribunal por cualquier
        # causa sin la orden hallada (S; con la entidad si la resolución la nombra) y penalidades (P), por hechos del año
        # del caso. Las multas quedan solo en la historia: su fecha es la de la resolución, no la de los hechos. La
        # inhabilitación es la sanción misma del Tribunal: ya está en S y en la C núcleo.
        S = self.antecedentes.assign(fuente="S")[["fuente", "documento", "ruc_entidad", "ruc", "anio_hechos", "fecha_conocido"]]
        pen = self.registros[self.registros["tipo"] == "penalidad"]
        P = pen.assign(fuente="P", documento=pen["tipo"], ruc_entidad=np.nan)[["fuente", "documento", "ruc_entidad", "ruc", "anio_hechos", "fecha_conocido"]]
        am = pd.concat([S, P], ignore_index=True).dropna(subset=["ruc", "anio_hechos", "fecha_conocido"])
        am["anio_hechos"] = am["anio_hechos"].astype(int)
        am["fecha_conocido"] = pd.to_datetime(am["fecha_conocido"])
        self.amplio = am

    def etiqueta_amplia(self, casos, fecha_corte, horizonte=0, ventana_anios=VENTANA_ANIOS):
        """Señal adversa documentada (HALLAZGOS 130 y 160, C5): la etiqueta estricta (A, B o C núcleo) o, del proveedor, una
        sanción del Tribunal por cualquier causa (S) o una penalidad (P), por hechos del año del caso, conocida hasta
        fecha_corte y dentro de la ventana (dos años desde el cierre del año de los hechos: se mide igual en todos los cortes
        y no necesita el año sellado). Si la resolución nombra la entidad del caso, pesa 1; si es solo del proveedor, se
        reparte entre sus casos del año (1/k).
        Devuelve lo mismo que `etiqueta`, más y_estricta y fuentes_estrictas, que conservan TODO lo conocido hasta
        fecha_corte: las vistas de hallazgos e impedimentos se miden con esas, sin la ventana."""
        corte = pd.Timestamp(fecha_corte)
        completa = Hechos.etiqueta(self, casos, fecha_corte, horizonte)
        out = Hechos.etiqueta(self, casos, fecha_corte, horizonte, ventana_anios=ventana_anios).rename(columns={"y": "y_ventana"})
        out["y_estricta"], out["fuentes_estrictas"] = completa["y"].to_numpy(), completa["fuentes"].to_numpy()
        a = casos["anio"].astype(int).to_numpy() + horizonte
        e = self.amplio[self.amplio["fecha_conocido"] <= corte]
        if ventana_anios is not None:
            e = dentro_de_ventana(e, ventana_anios)
        base = pd.DataFrame({"ruc_entidad": casos["ruc_entidad"].astype(str).to_numpy(), "ruc": casos["ruc"].astype(str).to_numpy(),
                             "anio": a, "i": np.arange(len(casos))})
        k = base.groupby(["ruc", "anio"])["i"].transform("size").to_numpy()
        ee = e.rename(columns={"anio_hechos": "anio"})
        rel = ee.dropna(subset=["ruc_entidad"]).astype({"ruc_entidad": str})
        m1 = base.merge(rel[["ruc_entidad", "ruc", "anio", "fecha_conocido", "fuente"]], on=["ruc_entidad", "ruc", "anio"]).assign(peso=1.0)
        m2 = base.merge(ee[["ruc", "anio", "fecha_conocido", "fuente"]], on=["ruc", "anio"])
        f = pd.concat([m1, m2.assign(peso=1.0 / k[m2["i"].to_numpy()])], ignore_index=True)
        g = f.groupby("i").agg(fecha=("fecha_conocido", "min"), peso=("peso", "max"), fuentes=("fuente", lambda s: "".join(sorted(set(s)))))
        g = g.reindex(np.arange(len(casos)))
        tiene = g["fecha"].notna().to_numpy()
        est = out["y_ventana"].to_numpy()
        out["y"] = est | tiene
        out["peso"] = np.maximum(np.where(est, out["peso"], 0.0), g["peso"].fillna(0).to_numpy())
        f_amp = pd.to_datetime(g["fecha"]).to_numpy()
        f_est = pd.to_datetime(out["fecha_conocido"]).to_numpy()
        out["fecha_conocido"] = np.where(pd.isna(f_est), f_amp, np.where(pd.isna(f_amp), f_est, np.minimum(f_est, f_amp)))
        out["fuentes"] = out["fuentes"] + g["fuentes"].fillna("").to_numpy()
        out["conocido_al_cierre"] = out["y"] & (pd.to_datetime(out["fecha_conocido"]) <= cierre(a))
        return out[["y", "peso", "fecha_conocido", "fuentes", "conocido_al_cierre", "y_estricta", "fuentes_estrictas"]]

    # ---------------------------------------------------------------------------------------------- etiqueta
    def etiqueta(self, casos, fecha_corte, horizonte=0, ventana_anios=None):
        """Para cada caso (ruc_entidad, ruc, anio): ¿hay un hecho de la relación en el año anio + horizonte conocido hasta
        fecha_corte? Devuelve y (bool), peso (fracción de B; 1 si hay A o C), fecha del primer conocimiento, fuentes y si
        ya se conocía al cierre del año del caso (para excluirlo de la detección)."""
        corte = pd.Timestamp(fecha_corte)
        if not self.abrir_sellado:
            assert corte < SELLADO_DESDE, "en exploración ningún corte llega a 2026"
        e = self.ev[self.ev["fecha_conocido"] <= corte]
        if ventana_anios is not None:
            e = dentro_de_ventana(e, ventana_anios)
        g = e.groupby(["ruc_entidad", "ruc", "anio_hechos"]).agg(fecha=("fecha_conocido", "min"), peso=("peso", "max"),
                                                                 fuentes=("fuente", lambda s: "".join(sorted(set(s)))))
        llave = pd.MultiIndex.from_arrays([casos["ruc_entidad"].astype(str), casos["ruc"].astype(str),
                                           casos["anio"].astype(int) + horizonte])
        r = g.reindex(llave)
        out = pd.DataFrame({"y": r["fecha"].notna().to_numpy(), "peso": r["peso"].fillna(0).to_numpy(),
                            "fecha_conocido": r["fecha"].to_numpy(), "fuentes": r["fuentes"].fillna("").to_numpy()}, index=casos.index)
        out["conocido_al_cierre"] = out["y"] & (out["fecha_conocido"] <= cierre(casos["anio"].to_numpy() + horizonte))
        return out

    def comprobar_sin_futuro(self, etiquetas, fecha_corte):
        """Prueba automática: ningún hecho de entrenamiento puede conocerse después del corte."""
        malo = etiquetas["y"] & (pd.to_datetime(etiquetas["fecha_conocido"]) > pd.Timestamp(fecha_corte))
        assert not malo.any(), f"{int(malo.sum())} etiquetas de entrenamiento con conocimiento posterior al corte"

    # ---------------------------------------------------------------------------------------------- historia
    def historia(self, casos):
        """Historia conocida al cierre del año de cada caso, de todo el país: del proveedor, de la entidad y de la
        relación (hechos de años anteriores ya conocidos). Nunca usa lo conocido después del cierre del año."""
        anios = sorted(casos["anio"].astype(int).unique())
        filas_p, filas_e, filas_r = [], [], []
        for a in anios:
            fin = pd.Timestamp(f"{a}-12-31")
            e = self.ev[self.ev["fecha_conocido"] <= fin]
            filas_p.append(e.groupby("ruc").agg(h_prov_hechos=("documento", "nunique"), h_prov_entidades=("ruc_entidad", "nunique"),
                                                h_prov_impedimento=("es_impedimento", "max")).assign(anio=a))
            filas_e.append(e.groupby("ruc_entidad").agg(h_ent_hechos=("documento", "nunique"),
                                                        h_ent_proveedores=("ruc", "nunique")).assign(anio=a))
            filas_r.append(e.groupby(["ruc_entidad", "ruc"]).agg(h_rel_hechos=("documento", "nunique")).assign(anio=a))
            c = self.antecedentes[self.antecedentes["fecha_conocido"] <= fin]
            rg = self.registros[self.registros["fecha_conocido"] <= fin]
            s = self.sanciones[self.sanciones["inicio"] <= fin]
            extra = pd.concat([c.groupby("ruc").agg(h_prov_sanciones_otras=("documento", "nunique")),
                               rg[rg["tipo"] == "penalidad"].groupby("ruc").size().rename("h_prov_penalidades"),
                               rg[rg["tipo"] == "multa"].groupby("ruc").size().rename("h_prov_multas"),
                               s.groupby("ruc").size().rename("h_prov_inhabilitaciones"),
                               s[s["fin"].isna() | (s["fin"] >= fin)].groupby("ruc").size().gt(0).rename("h_prov_inhabilitado_al_cierre")], axis=1)
            filas_p[-1] = filas_p[-1].join(extra, how="outer").assign(anio=a)
        hp = pd.concat(filas_p).reset_index().rename(columns={"index": "ruc"})
        he = pd.concat(filas_e).reset_index()
        hr = pd.concat(filas_r).reset_index()
        x = casos[K].astype({"anio": int}).merge(hp, on=["ruc", "anio"], how="left").merge(he, on=["ruc_entidad", "anio"], how="left") \
            .merge(hr, on=K, how="left")
        cols = [c for c in x.columns if c.startswith("h_")]
        x[cols] = x[cols].astype(float).fillna(0)
        return x.set_index(casos.index)[cols]

    # ---------------------------------------------------------------------------------------------- madurez
    def madurez(self, anios_ref=(2012, 2018), meses=range(0, 85, 6)):
        """Parte de los hechos de cada fuente ya conocida a los d meses del cierre de su año, con los hechos de los años de
        referencia. Insumo de D2 (filas inmaduras) y de la calibración. Usa solo lo no sellado. Limitación documentada
        (HALLAZGOS 121): la forma de la demora se estima con lo conocido hasta 2025; no revela qué casos son positivos."""
        e = self.ev[self.ev["anio_hechos"].between(*anios_ref)]
        dem = (e["fecha_conocido"] - pd.to_datetime(e["anio_hechos"].astype(str) + "-12-31")).dt.days / 30.44
        return pd.DataFrame({f: {m: float((dem[e["fuente"] == f] <= m).mean()) for m in meses} for f in e["fuente"].unique()})
