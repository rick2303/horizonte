import numpy as np, pandas as pd
from src import experimento

rng = np.random.default_rng(7)
fechas = pd.date_range("2000-01-01","2025-12-01",freq="MS"); n=len(fechas); t=np.arange(n)
rem = 180*np.exp(0.0125*t)*(1+.18*np.sin(2*np.pi*(t-2)/12))*np.exp(rng.normal(0,.06,n))
rem = pd.Series(rem, index=fechas, name="Remesas")

out = experimento.ejecutar(rem, horizontes=(1,), logaritmo=True, minimo_entrenamiento=200)
print("--- DESEMPENO ---"); print(out["desempeno"].to_string(index=False))
print("\n--- DM ---"); print(out["diebold_mariano"].to_string(index=False))
print("\n--- SOBREAJUSTE ---"); print(out["sobreajuste"])
