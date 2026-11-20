import time, numpy as np, pandas as pd
from src import modelos, evaluacion

rng = np.random.default_rng(7)
fechas = pd.date_range("2000-01-01","2025-12-01",freq="MS"); n=len(fechas); t=np.arange(n)
ipc = pd.Series(50*np.exp(0.0045*t)+0.6*np.sin(2*np.pi*t/12)+rng.normal(0,.25,n), index=fechas, name="IPC")

ini = time.time()
arima = modelos.Arima().seleccionar(ipc.iloc[:187])
print(f"Seleccion ARIMA: {arima.especificacion()}  AIC={arima.aic:.2f}  ({time.time()-ini:.1f}s)")

for M in [modelos.IngenuoEstacional, modelos.BosqueAleatorio, modelos.Xgboost]:
    m = M().ajustar(ipc.iloc[:187], 1)
    print(f"{M.nombre:22s} proyeccion h=1: {m.proyectar():8.3f}   (real {ipc.iloc[187]:.3f})")
m = modelos.Arima(arima.orden, arima.orden_estacional).ajustar(ipc.iloc[:187],1)
print(f"{'ARIMA':22s} proyeccion h=1: {m.proyectar():8.3f}   (real {ipc.iloc[187]:.3f})")
