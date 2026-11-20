"""
Datos de la interfaz del prototipo.

Reúne en un único diccionario, serializable como JSON, todo lo que muestra la
pantalla principal: historia de cada serie, proyecciones de ambos modelos con
su intervalo, explicación de la proyección del primer mes y tablas de
desempeño. La interfaz (app/plantilla.html) solo dibuja; ningún número se
calcula en el navegador.

Dos explicaciones distintas, una por modelo:

  - XGBoost se explica con valores SHAP (apartado 3.4.2), porque es opaco.
  - SARIMA se explica descomponiendo su propia ecuación, que es lineal y
    conocida. Esa descomposición es exacta y no necesita SHAP.

Cuando la serie se modela en logaritmos (remesas), los términos de ambas
explicaciones se suman en la escala logarítmica y se traducen a millones de
dólares encadenándolos en el orden en que se muestran, de modo que la suma de
las barras coincide exactamente con la proyección.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from . import caracteristicas, modelos

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
HORIZONTES = (1, 3, 6, 12)
MAXIMO = 12
NOMBRES_MODELO = {
    "Modelo ingenuo estacional": "ingenuo",
    "ARIMA de referencia": "sarima",
    "Aprendizaje automático": "xgb",
}


def _mes(fecha) -> str:
    return f"{MESES[fecha.month - 1]} {fecha.year}"


def _encadenar(inicio_modelo: float, pasos: list[float], logaritmo: bool) -> tuple[float, list[float]]:
    """
    Convierte una suma de términos en la escala del modelo en contribuciones
    en la escala de la serie. Devuelve el punto de partida y la contribución
    de cada término, que suman exactamente la proyección.
    """
    if not logaritmo:
        return float(inicio_modelo), [float(p) for p in pasos]
    acumulado = np.cumsum([inicio_modelo] + list(pasos))
    niveles = np.exp(acumulado)
    return float(niveles[0]), [float(x) for x in np.diff(niveles)]


# ---------------------------------------------------------------------------
# Explicación del SARIMA: descomposición exacta de su ecuación
# ---------------------------------------------------------------------------

def _polinomio(params: pd.Series, prefijo: str, signo: float, periodo: int, estacional: bool) -> np.ndarray:
    """Coeficientes de 1 + signo*sum(c_k B^k) a partir de los parámetros de statsmodels."""
    coef = {}
    for nombre, valor in params.items():
        if not nombre.startswith(prefijo):
            continue
        es_estacional = ".S.L" in nombre
        if es_estacional != estacional:
            continue
        rezago = int(nombre.split("L")[-1])
        coef[rezago] = valor
    grado = max(coef) if coef else 0
    poly = np.zeros(grado + 1)
    poly[0] = 1.0
    for rezago, valor in coef.items():
        poly[rezago] = signo * valor
    return poly


def explicar_sarima(arima: "modelos.Arima", serie: pd.Series, nombre_corto: str) -> dict:
    """
    Descompone la proyección a un mes del SARIMA ajustado.

    Con (1-B)^d (1-B^12)^D z_t = w_t y w_t siguiendo un ARMA estacional:

        z_{T+1} = punto de partida + cambio estacional del año anterior
                  + términos autorregresivos + términos de media móvil

    Los términos se agrupan por su origen en la ecuación. Si la suma no
    reproduce el pronóstico de statsmodels se devuelve None.
    """
    resultado = arima.resultado
    params = resultado.params
    z = np.log(serie.values) if arima.logaritmo else serie.values.astype(float)
    d = arima.orden[1]
    D = arima.orden_estacional[1]
    s = arima.orden_estacional[3]
    if d > 1 or D > 1:
        return None

    # Polinomio de diferenciación (1-B)^d (1-B^s)^D = 1 + c_1 B + ...
    dif = np.array([1.0])
    if d:
        dif = np.convolve(dif, [1.0, -1.0])
    if D:
        dif = np.convolve(dif, np.r_[1.0, np.zeros(s - 1), -1.0])

    n = len(z)
    w = np.array([sum(dif[k] * z[t - k] for k in range(len(dif))) for t in range(len(dif) - 1, n)])

    # AR: phi(B) Phi(B^s) = 1 - a_1 B - ...   MA: theta(B) Theta(B^s) = 1 + m_1 B + ...
    ar_ns = _polinomio(params, "ar.", -1.0, s, False)
    ar_s_corto = _polinomio(params, "ar.", -1.0, s, True)
    ma_ns = _polinomio(params, "ma.", 1.0, s, False)
    ma_s_corto = _polinomio(params, "ma.", 1.0, s, True)
    ar = np.convolve(ar_ns, ar_s_corto)
    ma = np.convolve(ma_ns, ma_s_corto)
    errores = np.asarray(resultado.resid)

    # Deriva: en statsmodels es una tendencia lineal sobre la serie sin
    # diferenciar; tras una diferencia equivale a un crecimiento constante
    # por mes, y los términos autorregresivos actúan sobre w menos la deriva.
    deriva = float(params.get("drift", 0.0))
    terminos = {"ar_ns": 0.0, "ar_s": 0.0, "ma_ns": 0.0, "ma_s": 0.0}
    for k in range(1, len(ar)):
        if ar[k] != 0:
            terminos["ar_ns" if k < s else "ar_s"] += -ar[k] * (w[-k] - deriva)
    for k in range(1, len(ma)):
        if ma[k] != 0:
            terminos["ma_ns" if k < s else "ma_s"] += ma[k] * errores[-k]
    constante = float(params.get("intercept", 0.0)) + deriva

    # Parte determinada por la diferenciación: z_{T+1} = w_{T+1} - sum c_k z_{T+1-k}
    if d:
        inicio = z[-1]
        estacional = sum(-dif[k] * z[n - k] for k in range(2, len(dif))) if D else 0.0
    else:
        inicio = z[n - s] if D else 0.0
        estacional = 0.0

    total = inicio + estacional + constante + sum(terminos.values())
    esperado = float(resultado.forecast(1).iloc[0])
    if not np.isclose(total, esperado, rtol=0, atol=1e-6 * max(1.0, abs(esperado))):
        # Con coeficientes de media móvil cercanos a -1 el filtro de Kalman
        # aún no converge a la recursión simple y los residuos no reproducen
        # el pronóstico. En ese caso los términos de media móvil se reportan
        # juntos, calculados como lo que falta para llegar al pronóstico: es
        # exactamente el aporte del componente de media móvil filtrado.
        terminos["ma_ns"] = 0.0
        terminos["ma_s"] = esperado - (inicio + estacional + constante + terminos["ar_ns"] + terminos["ar_s"])
        terminos["ma_juntos"] = True

    ultimo = serie.index[-1]
    objetivo = ultimo + pd.DateOffset(months=1)
    hace_un_anio = objetivo - pd.DateOffset(months=12)
    filas = []
    if D and d:
        filas.append((
            f"Cambio de {MESES[objetivo.month - 1]} del año pasado",
            f"Patrón estacional: {_mes(hace_un_anio)} frente a {_mes(hace_un_anio - pd.DateOffset(months=1))}",
            estacional,
        ))
    if constante:
        filas.append(("Constante de la ecuación", "Deriva: crecimiento de fondo", constante))
    etiquetas = {
        "ar_ns": ("Ajuste por tendencia reciente", "Términos autorregresivos (φ)"),
        "ar_s": ("Ajuste estacional", "Autorregresivo estacional (Φ)"),
        "ma_ns": ("Corrección por error reciente", "Media móvil (θ)"),
        "ma_s": ("Corrección por error estacional", "Media móvil estacional (Θ)"),
    }
    if terminos.pop("ma_juntos", False):
        etiquetas["ma_s"] = ("Corrección por errores pasados", "Media móvil reciente y estacional (θ, Θ)")
    for clave, (nombre, descripcion) in etiquetas.items():
        if abs(terminos[clave]) > 0:
            filas.append((nombre, descripcion, terminos[clave]))

    base, contribuciones = _encadenar(inicio, [f[2] for f in filas], arima.logaritmo)
    return {
        "metodo": "ecuacion",
        "base": base,
        "base_etiqueta": f"Punto de partida ({nombre_corto} {_mes(ultimo)})",
        "fecha": _mes(objetivo),
        "terminos": [
            {"nombre": f[0], "descripcion": f[1], "valor": v}
            for f, v in zip(filas, contribuciones)
        ],
        "proyeccion": base + sum(contribuciones),
        "estacional": contribuciones[0] if (D and d) else None,
    }


# ---------------------------------------------------------------------------
# Explicación del XGBoost con SHAP
# ---------------------------------------------------------------------------

def _describir(variable: str, ultimo, objetivo, nombre_corto: str) -> tuple[str, str]:
    legibles = caracteristicas.nombres_legibles()
    if variable.startswith("rezago_"):
        k = int(variable.split("_")[1])
        fecha = ultimo - pd.DateOffset(months=k - 1)
        return legibles[variable], f"{nombre_corto} de {_mes(fecha)}"
    if variable.startswith("media_movil_"):
        v = int(variable.split("_")[2])
        inicio = ultimo - pd.DateOffset(months=v - 1)
        return legibles[variable], f"Media móvil {MESES[inicio.month - 1]}–{MESES[ultimo.month - 1]}"
    if variable == "variacion_1":
        return legibles[variable], f"{_mes(ultimo)} frente a {_mes(ultimo - pd.DateOffset(months=1))}"
    if variable == "variacion_12":
        return legibles[variable], f"{_mes(ultimo)} frente a {_mes(ultimo - pd.DateOffset(months=12))}"
    return "Mes del año", f"Efecto de {MESES[objetivo.month - 1]}"


def explicar_xgboost(modelo: "modelos.ModeloArbol", nombre_corto: str, mostrar: int = 6) -> dict:
    """Valores SHAP de la proyección a un mes, agrupando el mes del año."""
    fila = modelo.fila_prediccion()
    explicador = shap.TreeExplainer(modelo.estimador)
    valores = pd.Series(np.ravel(explicador.shap_values(fila)), index=fila.columns).astype(float)
    base_modelo = float(np.ravel(explicador.expected_value)[0]) + float(modelo.nivel_de_partida(fila)[0])

    # Seno y coseno del mes son una sola idea; los valores SHAP son aditivos.
    agrupado = valores.drop(["mes_seno", "mes_coseno"])
    agrupado["mes"] = valores["mes_seno"] + valores["mes_coseno"]
    agrupado = agrupado.reindex(agrupado.abs().sort_values(ascending=False).index)

    principales = agrupado.iloc[:mostrar]
    resto = float(agrupado.iloc[mostrar:].sum())
    pasos = list(principales.values) + ([resto] if len(agrupado) > mostrar else [])
    base, contribuciones = _encadenar(base_modelo, pasos, modelo.logaritmo)

    ultimo = modelo.serie.index[-1]
    objetivo = fila.index[0]
    terminos = []
    for variable, valor in zip(principales.index, contribuciones):
        nombre, descripcion = _describir(variable, ultimo, objetivo, nombre_corto)
        terminos.append({"nombre": nombre, "descripcion": descripcion, "valor": valor})
    if len(agrupado) > mostrar:
        terminos.append({
            "nombre": "Otras variables",
            "descripcion": f"{len(agrupado) - mostrar} variables de menor peso",
            "valor": contribuciones[-1],
        })

    return {
        "metodo": "shap",
        "base": base,
        "base_etiqueta": (
            "Punto de partida (último dato + cambio promedio)"
            if modelo.variacion else "Valor base (promedio del modelo)"
        ),
        "fecha": _mes(objetivo),
        "terminos": terminos,
        "proyeccion": base + sum(contribuciones),
    }


# ---------------------------------------------------------------------------
# Intervalo del XGBoost a partir de sus errores fuera de muestra
# ---------------------------------------------------------------------------

def _cuantiles_xgb(resultados: Path, clave: str) -> dict[int, tuple[float, float]]:
    """
    XGBoost no produce intervalos por sí mismo. Se construyen con los errores
    relativos que cometió en la validación de origen móvil: percentiles 2.5 y
    97.5 por horizonte. Es un intervalo empírico, no paramétrico.
    """
    cuantiles = {}
    for h in HORIZONTES:
        ruta = resultados / f"pronosticos_{clave}_h{h}_aprendizaje_automatico.csv"
        if not ruta.exists():
            continue
        tabla = pd.read_csv(ruta).dropna()
        relativo = tabla["error"] / tabla["proyectado"]
        cuantiles[h] = (float(relativo.quantile(0.025)), float(relativo.quantile(0.975)))
    return cuantiles


def _interpolar(cuantiles: dict, h: int) -> tuple[float, float] | None:
    if not cuantiles:
        return None
    hs = sorted(cuantiles)
    bajo = np.interp(h, hs, [cuantiles[x][0] for x in hs])
    alto = np.interp(h, hs, [cuantiles[x][1] for x in hs])
    return float(bajo), float(alto)


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------

def datos_serie(
    serie: pd.Series,
    clave: str,
    logaritmo: bool,
    resultados: Path,
    nombre_corto: str,
    posterior: pd.Series | None = None,
) -> dict:
    resultados = Path(resultados)
    resumen_ruta = resultados / f"resumen_{clave}.json"
    resumen = json.loads(resumen_ruta.read_text(encoding="utf-8")) if resumen_ruta.exists() else {}
    variacion = bool(resumen.get("variacion", False))

    # ARIMA: misma especificación que en el experimento, reestimada con toda la serie.
    arima = modelos.Arima(logaritmo=logaritmo)
    if resumen.get("especificacion_arima"):
        numeros = [int(x) for x in resumen["especificacion_arima"].split(" ")[0].replace("SARIMA", "")
                   .replace(")(", ",").replace(")[", ",").strip("()[]").split(",")]
        arima.orden, arima.orden_estacional = tuple(numeros[:3]), tuple(numeros[3:7])
    else:
        arima.seleccionar(serie)
    arima.ajustar(serie, 1)
    trayectoria = arima.trayectoria(MAXIMO)

    xgb = {h: modelos.Xgboost(logaritmo=logaritmo, variacion=variacion).ajustar(serie, h)
           for h in range(1, MAXIMO + 1)}
    cuantiles = _cuantiles_xgb(resultados, clave)

    proyeccion_xgb = []
    for h in range(1, MAXIMO + 1):
        valor = xgb[h].proyectar()
        rango = _interpolar(cuantiles, h)
        # error = observado - proyectado, así que observado = proyectado (1 + relativo)
        proyeccion_xgb.append({
            "valor": valor,
            "inferior": valor * (1 + rango[0]) if rango else None,
            "superior": valor * (1 + rango[1]) if rango else None,
        })

    desempeno, dm = {}, []
    tabla = resultados / f"desempeno_{clave}.csv"
    if tabla.exists():
        df = pd.read_csv(tabla)
        for h, grupo in df.groupby("Horizonte"):
            desempeno[str(h)] = {
                NOMBRES_MODELO.get(f["Modelo"], f["Modelo"]): {
                    m: float(f[m]) for m in ("MAE", "RMSE", "MAPE", "MASE")
                }
                for _, f in grupo.iterrows()
            }
    tabla = resultados / f"dm_{clave}.csv"
    if tabla.exists():
        for _, f in pd.read_csv(tabla).iterrows():
            dm.append({
                "horizonte": f["Horizonte"],
                "estadistico": float(f["Estadístico"]),
                "valor_p": float(f["Valor p"]),
                "favorecido": {"Modelo de referencia": "sarima", "Modelo alternativo": "xgb"}
                .get(f["Modelo favorecido"], "ninguno"),
            })

    fechas_proy = [serie.index[-1] + pd.DateOffset(months=h) for h in range(1, MAXIMO + 1)]

    # Verificación: meses ya publicados que el modelo no vio al entrenarse.
    verificacion = None
    if posterior is not None and len(posterior):
        filas = []
        for i, fecha in enumerate(fechas_proy):
            if fecha not in posterior.index or pd.isna(posterior.loc[fecha]):
                continue
            real = float(posterior.loc[fecha])
            a, x = trayectoria.iloc[i], proyeccion_xgb[i]
            filas.append({
                "mes": fecha.strftime("%Y-%m"),
                "real": real,
                "sarima": float(a.proyectado),
                "inferior": float(a.inferior),
                "superior": float(a.superior),
                "xgb": float(x["valor"]),
            })
        if filas:
            verificacion = {
                "filas": filas,
                "error_sarima": float(np.mean([abs(f["real"] - f["sarima"]) / f["real"] for f in filas]) * 100),
                "error_xgb": float(np.mean([abs(f["real"] - f["xgb"]) / f["real"] for f in filas]) * 100),
                "dentro": int(sum(f["inferior"] <= f["real"] <= f["superior"] for f in filas)),
            }
    return {
        "historia": {
            "fechas": [f.strftime("%Y-%m") for f in serie.index],
            "valores": [float(v) for v in serie.values],
        },
        "fechas_proyeccion": [f.strftime("%Y-%m") for f in fechas_proy],
        "proyecciones": {
            "sarima": [
                {"valor": float(r.proyectado), "inferior": float(r.inferior), "superior": float(r.superior)}
                for r in trayectoria.itertuples()
            ],
            "xgb": proyeccion_xgb,
        },
        "explicacion": {
            "sarima": explicar_sarima(arima, serie, nombre_corto),
            "xgb": explicar_xgboost(xgb[1], nombre_corto),
        },
        "especificacion_arima": arima.especificacion(),
        "logaritmo": logaritmo,
        "variacion_xgb": variacion,
        "desempeno": desempeno,
        "diebold_mariano": dm,
        "sobreajuste": resumen.get("sobreajuste"),
        "verificacion": verificacion,
    }


def construir(
    ipc: pd.Series,
    remesas: pd.Series,
    resultados: str | Path,
    ilustrativo: bool,
    posteriores: dict[str, pd.Series] | None = None,
) -> dict:
    posteriores = posteriores or {}
    return {
        "generado": date.today().isoformat(),
        "ilustrativo": ilustrativo,
        "series": {
            "ipc": datos_serie(ipc, "ipc", False, resultados, "IPC", posteriores.get("ipc")),
            "rem": datos_serie(remesas, "remesas", True, resultados, "Remesas", posteriores.get("remesas")),
        },
    }


def generar_html(datos: dict, plantilla: str | Path, destino: str | Path) -> str:
    """Inserta los datos en la plantilla y guarda una página autónoma."""
    html = Path(plantilla).read_text(encoding="utf-8")
    contenido = json.dumps(datos, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = html.replace("/*__DATOS__*/null", contenido)
    Path(destino).write_text(html, encoding="utf-8")
    return html
