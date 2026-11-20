"""
Ingeniería de variables predictoras.

Traduce una serie de tiempo a la estructura tabular que exigen los modelos
basados en árboles, según lo expuesto en el apartado 3.2.1 y operacionalizado
en la Tabla 3 del apartado 4.2.

La dependencia temporal se incorpora dentro de las variables predictoras
mediante tres bloques:

  - rezagos de la propia serie,
  - medias móviles que resumen el comportamiento reciente,
  - codificación cíclica del mes del año, que representa la estacionalidad.

Regla que gobierna todo el módulo: ninguna variable construida para el mes t
puede utilizar información posterior a t. Las medias móviles se calculan sobre
valores ya rezagados por esa razón.

Convención de alineación
------------------------
La fila del mes t contiene información disponible hasta t-1 (rezago_1 es el
valor de t-1). Con horizonte h, esa fila se empareja con el valor de t+h-1,
que está exactamente h meses después del último dato que el modelo conoce.

En la proyección real, con datos hasta el mes T, la fila que se construye
corresponde al mes T+1 (rezago_1 es el valor de T) y proyecta el mes T+h.
Ambas funciones de este módulo siguen la misma convención; si se rompe, el
modelo aprende a proyectar h+1 meses y se le evalúa contra h meses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REZAGOS = (1, 2, 3, 6, 12)
VENTANAS = (3, 6, 12)


def _codificar_mes(meses) -> tuple[np.ndarray, np.ndarray]:
    """Seno y coseno del mes, para que diciembre y enero queden próximos."""
    meses = np.asarray(meses)
    return np.sin(2 * np.pi * meses / 12), np.cos(2 * np.pi * meses / 12)


def construir(
    serie: pd.Series,
    horizonte: int = 1,
    rezagos: tuple[int, ...] = REZAGOS,
    ventanas: tuple[int, ...] = VENTANAS,
    logaritmo: bool = False,
    variacion: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye la matriz de variables predictoras y el vector objetivo.

    Parámetros
    ----------
    serie : serie mensual depurada
    horizonte : número de meses hacia adelante que se desea proyectar,
        contados desde el último dato conocido. Es la estrategia directa de
        pronóstico multipaso: se entrena un modelo por cada horizonte, lo que
        evita la acumulación de error que produce la estrategia recursiva.
    rezagos : rezagos de la serie que se incluyen como predictores
    ventanas : longitudes de las medias móviles
    logaritmo : aplica transformación logarítmica a la serie antes de
        construir las variables. Se usa cuando la serie presenta crecimiento
        multiplicativo, como ocurre con las remesas (apartado 4.6, etapa 3).
    variacion : si es True, el objetivo deja de ser el nivel de la serie y
        pasa a ser su cambio respecto del último dato conocido (rezago_1).
        Los árboles de decisión no pueden proyectar valores fuera del rango
        que vieron al entrenar; en una serie con tendencia, como el IPC, el
        nivel futuro siempre está fuera de ese rango y el modelo se queda
        plano. El cambio, en cambio, sí se mantiene dentro de lo observado.
        La proyección del nivel se recupera sumando el cambio al último dato.

    Devuelve
    --------
    X : DataFrame de variables predictoras
    y : Serie objetivo alineada con X

    El número de filas se reduce respecto de la serie original porque las
    primeras observaciones no tienen historia suficiente para calcular los
    rezagos, y las últimas no tienen todavía un valor observado en el mes
    objetivo.
    """
    if horizonte < 1:
        raise ValueError("El horizonte debe ser un entero mayor o igual a 1.")

    if logaritmo and (serie <= 0).any():
        raise ValueError(
            "La transformación logarítmica exige que todos los valores sean "
            "positivos. Revise la serie antes de activarla."
        )

    base = np.log(serie) if logaritmo else serie.copy()

    variables = pd.DataFrame(index=base.index)

    for k in rezagos:
        variables[f"rezago_{k}"] = base.shift(k)

    # Las medias móviles se calculan sobre la serie ya rezagada un período,
    # de modo que la media del mes t nunca incluye el valor del mes t.
    rezagada = base.shift(1)
    for v in ventanas:
        variables[f"media_movil_{v}"] = rezagada.rolling(window=v).mean()

    # La variación mensual y la interanual recogen la dinámica de corto y
    # largo plazo sin repetir la información de nivel de los rezagos.
    variables["variacion_1"] = base.shift(1) - base.shift(2)
    variables["variacion_12"] = base.shift(1) - base.shift(13)

    # Se codifica el mes que se proyecta, no el mes de la fila, porque la
    # estacionalidad relevante es la del período objetivo.
    meses_objetivo = (base.index + pd.DateOffset(months=horizonte - 1)).month
    variables["mes_seno"], variables["mes_coseno"] = _codificar_mes(meses_objetivo)

    # Fila t (información hasta t-1) frente al valor de t+h-1: h meses de
    # distancia respecto del último dato conocido.
    objetivo = base.shift(-(horizonte - 1))
    if variacion:
        objetivo = objetivo - base.shift(1)
    objetivo.name = "objetivo"

    conjunto = variables.join(objetivo).dropna()
    X = conjunto.drop(columns="objetivo")
    y = conjunto["objetivo"]
    return X, y


def variables_para_prediccion(
    serie: pd.Series,
    horizonte: int = 1,
    rezagos: tuple[int, ...] = REZAGOS,
    ventanas: tuple[int, ...] = VENTANAS,
    logaritmo: bool = False,
) -> pd.DataFrame:
    """
    Construye una única fila de variables predictoras a partir del último mes
    disponible de la serie. Es la que utiliza el prototipo para generar la
    proyección que se muestra al usuario.

    Con datos hasta T, la fila equivale a la del mes T+1 de construir(): el
    rezago_1 es el valor de T y las medias móviles terminan en T. El índice
    de la fila es el mes que se proyecta, T+h.
    """
    base = np.log(serie) if logaritmo else serie.copy()
    ultimo = base.index[-1]
    fila = {}

    for k in rezagos:
        fila[f"rezago_{k}"] = base.iloc[-k]

    for v in ventanas:
        fila[f"media_movil_{v}"] = base.iloc[-v:].mean()

    fila["variacion_1"] = base.iloc[-1] - base.iloc[-2]
    fila["variacion_12"] = base.iloc[-1] - base.iloc[-13]

    fecha_objetivo = ultimo + pd.DateOffset(months=horizonte)
    seno, coseno = _codificar_mes([fecha_objetivo.month])
    fila["mes_seno"] = float(seno[0])
    fila["mes_coseno"] = float(coseno[0])

    return pd.DataFrame([fila], index=[fecha_objetivo])


def nombres_legibles() -> dict[str, str]:
    """
    Traduce los nombres técnicos de las variables a etiquetas comprensibles
    para el usuario final. Se usa en los gráficos de explicabilidad del
    prototipo, donde el lector no tiene formación estadística.
    """
    etiquetas = {
        "variacion_1": "Cambio del último mes",
        "variacion_12": "Cambio respecto al año anterior",
        "mes_seno": "Época del año (componente 1)",
        "mes_coseno": "Época del año (componente 2)",
    }
    for k in REZAGOS:
        etiquetas[f"rezago_{k}"] = (
            "Valor del último mes" if k == 1 else f"Valor de hace {k} meses"
        )
    for v in VENTANAS:
        etiquetas[f"media_movil_{v}"] = f"Promedio de los últimos {v} meses"
    return etiquetas
