"""
Verifica que el entrenamiento y la proyección usen la misma alineación
temporal y que el ingenuo estacional tome el mes correcto.

Usa una serie cuyo valor es el número de mes (1, 2, 3, ...), de modo que
cualquier desfase se ve a simple vista.

    python -m pruebas.prueba_alineacion
"""
import numpy as np
import pandas as pd

from src import caracteristicas
from src.modelos import IngenuoEstacional

fechas = pd.date_range("2000-01-01", periods=60, freq="MS")
serie = pd.Series(np.arange(1, 61, dtype=float), index=fechas)

for h in (1, 3, 6, 12):
    X, y = caracteristicas.construir(serie, horizonte=h)
    salto = y.iloc[-1] - X.iloc[-1]["rezago_1"]
    assert salto == h, f"h={h}: el entrenamiento proyecta {salto} meses, no {h}"

    fila = caracteristicas.variables_para_prediccion(serie.iloc[:40], horizonte=h)
    assert fila.iloc[0]["rezago_1"] == 40
    assert fila.index[0] == fechas[39] + pd.DateOffset(months=h)

    # El mes codificado en la proyección coincide con el del entrenamiento.
    X40, _ = caracteristicas.construir(serie.iloc[:40 + h], horizonte=h)
    fila_equivalente = X40.iloc[-1]
    for columna in ("rezago_1", "media_movil_3", "variacion_12", "mes_seno", "mes_coseno"):
        assert np.isclose(fila_equivalente[columna], fila.iloc[0][columna]), (h, columna)

    ingenuo = IngenuoEstacional().ajustar(serie.iloc[:40], h).proyectar()
    assert ingenuo == 40 + h - 12, f"h={h}: ingenuo devolvió {ingenuo}"

print("Alineación correcta en los horizontes 1, 3, 6 y 12.")
