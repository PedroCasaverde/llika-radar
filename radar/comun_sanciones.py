"""Regla común para saber si una sanción estaba vigente y publicada el día de una orden."""
import pandas as pd

from comun import SALIDA

LLAVE = ["ruc", "inicio", "fin", "resolucion"]


def vigentes_publicadas(x):
    """x trae una fila por (orden, sanción) con 'fecha' y las columnas de LLAVE. Devuelve las filas en que la sanción
    cubre la fecha y el registro publicado en las 6 semanas siguientes todavía la mostraba (antes de la primera versión
    guardada, del 29/09/2021, basta con que aparezca en esa primera versión)."""
    ap = pd.read_parquet(SALIDA / "sanciones_apariciones.parquet")
    ap["fin"] = ap["fin"].where(ap["fin"] < pd.Timestamp("2100-01-01"))
    primera = ap["corte"].min()
    x = x[(x["fecha"] >= x["inicio"]) & (x["fin"].isna() | (x["fecha"] <= x["fin"]))]
    y = x[["fecha"] + LLAVE].drop_duplicates().merge(ap, on=LLAVE)
    ok = ((y["corte"] >= y["fecha"]) & (y["corte"] <= y["fecha"] + pd.Timedelta(days=45))) | \
         ((y["fecha"] < primera) & (y["corte"] == primera))
    return x.merge(y.loc[ok, ["fecha"] + LLAVE].drop_duplicates(), on=["fecha"] + LLAVE)
