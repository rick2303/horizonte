import numpy as np, pandas as pd
from src import experimento

rng = np.random.default_rng(7)
fechas = pd.date_range("2000-01-01","2025-12-01",freq="MS"); n=len(fechas); t=np.arange(n)
ipc = pd.Series(50*np.exp(0.0045*t)+0.6*np.sin(2*np.pi*t/12)+rng.normal(0,.25,n), index=fechas, name="IPC")

out = experimento.ejecutar(ipc, horizontes=(1,3), minimo_entrenamiento=200)
print("--- TABLA DE DESEMPENO ---"); print(out["desempeno"].to_string(index=False))
print("\n--- DIEBOLD-MARIANO ---"); print(out["diebold_mariano"].to_string(index=False))
print("\n--- SOBREAJUSTE ---"); print(out["sobreajuste"])
