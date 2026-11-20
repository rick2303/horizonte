"""Prueba de humo: verifica que todos los módulos corren de extremo a extremo."""
import numpy as np, pandas as pd
from src import carga, caracteristicas, modelos, evaluacion

rng = np.random.default_rng(7)
fechas = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
n = len(fechas)
t = np.arange(n)

# IPC sintético: tendencia suave, estacionalidad leve, poco ruido.
ipc = 50 * np.exp(0.0045 * t) + 0.6*np.sin(2*np.pi*t/12) + rng.normal(0, .25, n)
ipc = pd.Series(ipc, index=fechas, name="IPC")

# Remesas sintéticas: crecimiento multiplicativo, estacionalidad marcada, ruido alto.
rem = 180*np.exp(0.0125*t) * (1 + .18*np.sin(2*np.pi*(t-2)/12)) * np.exp(rng.normal(0,.06,n))
rem = pd.Series(rem, index=fechas, name="Remesas")

print("=== DIAGNOSTICO ===")
print(carga.diagnostico(ipc).to_string()); print()
print(carga.diagnostico(rem).to_string()); print()

print("=== CARACTERISTICAS (h=1) ===")
X, y = caracteristicas.construir(ipc, horizonte=1)
print("X:", X.shape, "| y:", y.shape)
print("columnas:", list(X.columns)); print()

print("=== CARACTERISTICAS (h=3, log) ===")
X3, y3 = caracteristicas.construir(rem, horizonte=3, logaritmo=True)
print("X:", X3.shape, "| primera fecha:", X3.index[0].date(), "| ultima:", X3.index[-1].date())
print("fila de prediccion:"); print(caracteristicas.variables_para_prediccion(rem, logaritmo=True).round(3).to_string())
