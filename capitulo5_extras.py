"""
Cifras complementarias del Capítulo 5 que no imprime ejecutar_todo.py.

    python capitulo5_extras.py               todo, salvo la réplica
    python capitulo5_extras.py --replica     incluye la réplica sobre la serie
                                             desagregada de remesas (tarda más)

Deja los resultados en resultados/capitulo5_extras.json y las figuras 1 y 2
en resultados/. Usa las mismas series de datos/procesados/ que el
experimento principal (2000-01 a 2025-12).
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from src import carga, modelos, explicabilidad, experimento

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).parent
PROC = RAIZ / "datos" / "procesados"
RES = RAIZ / "resultados"

BLOQUES = {
    "Valores recientes": ["rezago_1", "rezago_2", "rezago_3", "rezago_6", "rezago_12"],
    "Promedios móviles": ["media_movil_3", "media_movil_6", "media_movil_12"],
    "Cambios recientes": ["variacion_1", "variacion_12"],
    "Componente estacional": ["mes_seno", "mes_coseno"],
}


def descriptivos(s):
    v = s.pct_change().dropna() * 100
    return {
        "observaciones": int(len(s)),
        "periodo": f"{s.index[0]:%Y-%m} a {s.index[-1]:%Y-%m}",
        "media": round(float(s.mean()), 2),
        "desviacion": round(float(s.std()), 2),
        "cv_pct": round(float(s.std() / s.mean() * 100), 1),
        "minimo": round(float(s.min()), 2),
        "maximo": round(float(s.max()), 2),
        "fecha_maximo": f"{s.idxmax():%Y-%m}",
        "var_mensual_media_pct": round(float(v.mean()), 2),
        "var_mensual_desv_pct": round(float(v.std()), 2),
    }


def estacionales(s):
    f = seasonal_decompose(s, model="multiplicative", period=12).seasonal
    m = f.groupby(f.index.month).mean()
    m = m / m.mean()
    return {int(k): round(float(x), 3) for k, x in m.items()}


def fuerza_estacional(s, log=False):
    x = np.log(s) if log else s
    x = x.diff().dropna()
    p = STL(x, period=12, robust=True).fit()
    return round(float(max(0.0, 1 - np.var(p.resid) / np.var(p.seasonal + p.resid))), 3)


def raiz_unitaria(x):
    x = x.dropna()
    adf = adfuller(x, autolag="AIC")
    k = kpss(x, regression="c", nlags="auto")
    return {"adf": round(float(adf[0]), 3), "adf_p": round(float(adf[1]), 4),
            "kpss": round(float(k[0]), 3), "kpss_p": round(float(k[1]), 3)}


def granger(ipc, rem):
    d = pd.DataFrame({"ipc": ipc.diff(), "rem": np.log(rem).diff()}).dropna()
    out = {}
    for dep, cau in [("ipc", "rem"), ("rem", "ipc")]:
        r = grangercausalitytests(d[[dep, cau]], maxlag=12)
        ps = {int(lag): float(r[lag][0]["ssr_ftest"][1]) for lag in r}
        out[f"{cau}_causa_{dep}"] = {
            "p_por_rezago": {k: round(v, 4) for k, v in ps.items()},
            "p_minimo": round(min(ps.values()), 4),
            "rezago_p_minimo": int(min(ps, key=ps.get)),
            "rechazos_5pct": int(sum(v < 0.05 for v in ps.values())),
        }
    return out


def shap_por_horizonte(serie, log):
    filas = {}
    for h in (1, 3, 6, 12):
        m = modelos.Xgboost(logaritmo=log).ajustar(serie, h)
        _, tabla, _ = explicabilidad.valores_entrenamiento(m)
        imp = tabla.abs().mean()
        tot = imp.sum()
        bloques = {b: round(float(imp[c].sum() / tot * 100), 2) for b, c in BLOQUES.items()}
        top = imp.sort_values(ascending=False)
        filas[int(h)] = {"bloques": bloques,
                    "principal": top.index[0],
                    "principal_pct": round(float(top.iloc[0] / tot * 100), 2)}
    return filas


def figura_series(ipc, rem):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
    ax[0].plot(ipc.index, ipc.values, color="#1F5FAE", lw=1.6)
    ax[0].set_title("A. Índice de precios al consumidor (dic. 1999 = 100)", loc="left", fontsize=10)
    ax[1].plot(rem.index, rem.values, color="#B4500C", lw=1.2)
    ax[1].set_title("B. Remesas familiares (millones de US$)", loc="left", fontsize=10)
    for a in ax:
        a.grid(alpha=.3); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "figura1_series.png", dpi=200)


def figura_mase():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    colores = {"Modelo ingenuo estacional": "#8A93A0", "ARIMA de referencia": "#1F5FAE", "Aprendizaje automático": "#B4500C"}
    nombres = {"Modelo ingenuo estacional": "Ingenuo estacional", "ARIMA de referencia": "SARIMA", "Aprendizaje automático": "XGBoost"}
    for a, (clave, tit) in zip(ax, [("ipc", "A. Índice de precios"), ("remesas", "B. Remesas familiares")]):
        d = pd.read_csv(RES / f"desempeno_{clave}.csv")
        hs = sorted(d.Horizonte.unique()); x = np.arange(len(hs)); w = .26
        for i, (m, c) in enumerate(colores.items()):
            v = d[d.Modelo == m].sort_values("Horizonte").MASE.values
            a.bar(x + (i - 1) * w, v, w, color=c, label=nombres[m])
        a.set_xticks(x, [f"{h} m" for h in hs]); a.set_title(tit, loc="left", fontsize=10)
        a.set_ylabel("MASE"); a.grid(axis="y", alpha=.3); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "figura2_mase.png", dpi=200)


def replica():
    s = carga.desde_remesas_departamento(str(RAIZ / "datos" / "crudos" / "remesas_departamento.xlsx"), "Remesas (departamento)")
    s = s.loc["2018-01":"2025-12"]
    sal = experimento.ejecutar(s, logaritmo=True, minimo_entrenamiento=60, verboso=True)
    return {"observaciones": int(len(s)),
            "especificacion": sal["especificacion_arima"],
            "desempeno": sal["desempeno"].to_dict(orient="records"),
            "diebold_mariano": sal["diebold_mariano"].to_dict(orient="records")}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--replica", action="store_true"); a = p.parse_args()
    ipc = carga.leer(str(PROC / "ipc.csv"), nombre="IPC")
    rem = carga.leer(str(PROC / "remesas.csv"), nombre="Remesas")
    r = {
        "descriptivos": {"ipc": descriptivos(ipc), "remesas": descriptivos(rem)},
        "indices_estacionales": {"ipc": estacionales(ipc), "remesas": estacionales(rem)},
        "fuerza_estacional_stl": {"ipc": fuerza_estacional(ipc), "remesas": fuerza_estacional(rem, True)},
        "raiz_unitaria": {
            "ipc_nivel": raiz_unitaria(ipc), "ipc_dif": raiz_unitaria(ipc.diff()),
            "rem_nivel": raiz_unitaria(rem), "rem_dif": raiz_unitaria(rem.diff()),
            "rem_log": raiz_unitaria(np.log(rem)), "rem_dif_log": raiz_unitaria(np.log(rem).diff()),
        },
        "granger": granger(ipc, rem),
        "shap_por_horizonte": {"ipc": shap_por_horizonte(ipc, False), "remesas": shap_por_horizonte(rem, True)},
    }
    figura_series(ipc, rem)
    if (RES / "desempeno_ipc.csv").exists():
        figura_mase()
    if a.replica:
        r["replica_desagregada"] = replica()
    (RES / "capitulo5_extras.json").write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
