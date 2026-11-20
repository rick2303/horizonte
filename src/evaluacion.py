"""
Evaluación del desempeño predictivo.

Implementa el procedimiento del apartado 3.3 y la etapa 5 del apartado 4.6:
validación de origen móvil, las cuatro métricas de error de la Tabla 2 y la
prueba de Diebold y Mariano con la corrección de Harvey et al. (1997).

Cada función documenta la fórmula que aplica y el origen de cada valor que
interviene, de modo que todo número reportado en el Capítulo 5 sea rastreable
hasta su definición.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Validación de origen móvil (apartado 3.3.1)
# ---------------------------------------------------------------------------

def validacion_origen_movil(
    serie: pd.Series,
    modelo_factoria,
    horizonte: int = 1,
    minimo_entrenamiento: int | None = None,
    ventana_expansiva: bool = True,
    verboso: bool = False,
) -> pd.DataFrame:
    """
    Genera pronósticos fuera de muestra desplazando el origen a lo largo de
    la serie.

    Procedimiento
    -------------
    Se parte de una ventana inicial de entrenamiento. El modelo se ajusta con
    la información disponible hasta el mes t, proyecta el mes t+h, se incorpora
    la observación efectivamente registrada y el ciclo se repite desplazando el
    origen un mes. Ningún pronóstico utiliza información posterior al mes que
    proyecta, que es la condición que distingue este procedimiento de la
    validación cruzada convencional (apartado 3.3.1, segundo párrafo).

    Parámetros
    ----------
    serie : serie mensual completa
    modelo_factoria : función sin argumentos que devuelve una instancia nueva
        del modelo. Se recibe como factoría y no como instancia para que cada
        origen entrene un modelo limpio, sin arrastrar el ajuste anterior.
    horizonte : meses hacia adelante que se proyectan
    minimo_entrenamiento : número de observaciones de la ventana inicial.
        Por omisión se toma el 60 % de la serie, con un mínimo de 60 meses,
        de modo que el modelo siempre disponga de al menos cinco ciclos
        estacionales completos.
    ventana_expansiva : si es True la ventana crece en cada origen; si es
        False se desplaza manteniendo longitud fija.

    Devuelve
    --------
    DataFrame indexado por la fecha proyectada, con las columnas
    observado, proyectado y error.
    """
    serie = serie.dropna()
    n = len(serie)

    if minimo_entrenamiento is None:
        minimo_entrenamiento = max(60, int(n * 0.6))

    if minimo_entrenamiento + horizonte > n:
        raise ValueError(
            f"La serie tiene {n} observaciones, insuficientes para una ventana "
            f"inicial de {minimo_entrenamiento} más un horizonte de {horizonte}."
        )

    registros = []
    for corte in range(minimo_entrenamiento, n - horizonte + 1):
        if ventana_expansiva:
            entrenamiento = serie.iloc[:corte]
        else:
            entrenamiento = serie.iloc[corte - minimo_entrenamiento: corte]

        fecha_objetivo = serie.index[corte + horizonte - 1]
        observado = float(serie.iloc[corte + horizonte - 1])

        try:
            modelo = modelo_factoria()
            modelo.ajustar(entrenamiento, horizonte)
            proyectado = modelo.proyectar()
        except Exception as error:
            if verboso:
                print(f"  origen {serie.index[corte - 1]:%Y-%m}: {error}")
            proyectado = np.nan

        registros.append(
            {
                "fecha": fecha_objetivo,
                "observado": observado,
                "proyectado": proyectado,
                "error": observado - proyectado,
            }
        )

    resultado = pd.DataFrame(registros).set_index("fecha")
    return resultado


# ---------------------------------------------------------------------------
# Métricas de error (apartado 3.3.2, Tabla 2)
# ---------------------------------------------------------------------------

def _mae(errores: np.ndarray) -> float:
    """Error absoluto medio: promedio del valor absoluto de los errores."""
    return float(np.mean(np.abs(errores)))


def _rmse(errores: np.ndarray) -> float:
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean(errores ** 2)))


def _mape(observados: np.ndarray, errores: np.ndarray) -> float:
    """
    Error porcentual absoluto medio, expresado en porcentaje.
    Se excluyen las observaciones nulas para evitar la división por cero.
    """
    validos = observados != 0
    return float(np.mean(np.abs(errores[validos] / observados[validos])) * 100)


def escala_mase(serie_entrenamiento: pd.Series, periodo: int = 12) -> float:
    """
    Denominador de la métrica MASE (Hyndman y Koehler, 2006).

    Es el error absoluto medio del modelo ingenuo estacional calculado
    **dentro de la muestra de entrenamiento**, no sobre el conjunto de prueba.
    Ese es el punto que suele confundirse: la escala debe ser independiente de
    los pronósticos que se están evaluando.

        escala = media( |y_t - y_{t-12}| )   para t = 13 ... T

    Un MASE inferior a 1 indica que el modelo supera al ingenuo estacional.
    """
    valores = serie_entrenamiento.dropna().values
    if len(valores) <= periodo:
        raise ValueError(
            "La muestra de entrenamiento debe superar el período estacional "
            "para poder calcular la escala del MASE."
        )
    diferencias = np.abs(valores[periodo:] - valores[:-periodo])
    return float(np.mean(diferencias))


def metricas(
    resultado: pd.DataFrame,
    escala: float,
    decimales: int = 4,
) -> dict[str, float]:
    """
    Calcula las cuatro métricas de la Tabla 2 sobre el resultado de la
    validación de origen móvil.

    Parámetros
    ----------
    resultado : salida de validacion_origen_movil
    escala : valor devuelto por escala_mase, calculado sobre la ventana
        inicial de entrenamiento de esa misma validación
    """
    limpio = resultado.dropna(subset=["proyectado"])
    errores = limpio["error"].values
    observados = limpio["observado"].values

    return {
        "MAE": round(_mae(errores), decimales),
        "RMSE": round(_rmse(errores), decimales),
        "MAPE": round(_mape(observados, errores), decimales),
        "MASE": round(_mae(errores) / escala, decimales),
        "n": int(len(errores)),
    }


# ---------------------------------------------------------------------------
# Prueba de Diebold y Mariano (apartado 3.3.3)
# ---------------------------------------------------------------------------

def diebold_mariano(
    errores_referencia: np.ndarray | pd.Series,
    errores_alternativo: np.ndarray | pd.Series,
    horizonte: int = 1,
    perdida: str = "cuadratica",
    correccion_harvey: bool = True,
) -> dict:
    """
    Contrasta la igualdad de exactitud esperada entre dos pronósticos
    competidores (Diebold y Mariano, 1995).

    Convención de signos
    --------------------
    El diferencial de pérdida se define como

        d_t = L(e_referencia_t) - L(e_alternativo_t)

    de manera que un estadístico **negativo indica mayor precisión del modelo
    de referencia** y uno **positivo mayor precisión del modelo alternativo**.
    Es exactamente la regla declarada en el apartado 4.3 y en la nota de la
    Tabla 4, donde el modelo de referencia es el ARIMA y el alternativo el de
    aprendizaje automático.

    Construcción del estadístico
    ----------------------------
    Sea d la media del diferencial de pérdida y n el número de pronósticos.
    Para un horizonte h, los diferenciales presentan autocorrelación hasta el
    rezago h-1, por lo que la varianza de la media se estima con las
    autocovarianzas correspondientes:

        V(d) = (1/n) [ gamma_0 + 2 * suma_{k=1}^{h-1} gamma_k ]

        DM = d / raiz( V(d) )

    Bajo la hipótesis nula el estadístico sigue una distribución normal
    estándar, de modo que un valor absoluto superior a 1.96 conduce al rechazo
    con nivel de significancia de 0.05.

    Corrección de Harvey et al. (1997)
    ----------------------------------
    Con muestras reducidas y horizontes mayores a un mes el estadístico
    original tiende a rechazar en exceso. La corrección aplica el factor

        factor = raiz( [ n + 1 - 2h + h(h-1)/n ] / n )

    y contrasta el resultado contra una distribución t de Student con n-1
    grados de libertad en lugar de la normal. Es el ajuste que el apartado
    3.3.3 anuncia como pertinente para esta investigación.

    Devuelve
    --------
    Diccionario con el estadístico, el valor p, la decisión y el modelo
    favorecido.
    """
    e1 = np.asarray(pd.Series(errores_referencia).dropna(), dtype=float)
    e2 = np.asarray(pd.Series(errores_alternativo).dropna(), dtype=float)

    if len(e1) != len(e2):
        raise ValueError(
            "Las dos series de errores deben tener la misma longitud. "
            "Verifique que ambas validaciones usaron la misma ventana inicial "
            "y el mismo horizonte."
        )

    n = len(e1)
    if n < 10:
        raise ValueError(
            f"Solo hay {n} pares de errores. La prueba no es informativa con "
            "tan pocos pronósticos."
        )

    if perdida == "cuadratica":
        d = e1 ** 2 - e2 ** 2
    elif perdida == "absoluta":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("La pérdida debe ser 'cuadratica' o 'absoluta'.")

    d_medio = float(np.mean(d))
    desviaciones = d - d_medio

    # gamma_0 es la varianza del diferencial; gamma_k sus autocovarianzas.
    gamma_0 = float(np.sum(desviaciones ** 2) / n)
    suma_autocovarianzas = 0.0
    for k in range(1, horizonte):
        gamma_k = float(np.sum(desviaciones[k:] * desviaciones[:-k]) / n)
        suma_autocovarianzas += gamma_k

    varianza = (gamma_0 + 2 * suma_autocovarianzas) / n

    if varianza <= 0:
        raise ValueError(
            "La varianza estimada del diferencial no es positiva. Suele ocurrir "
            "cuando los dos modelos producen pronósticos casi idénticos."
        )

    estadistico = d_medio / np.sqrt(varianza)

    if correccion_harvey:
        factor = np.sqrt(
            (n + 1 - 2 * horizonte + horizonte * (horizonte - 1) / n) / n
        )
        estadistico = estadistico * factor
        valor_p = 2 * (1 - stats.t.cdf(abs(estadistico), df=n - 1))
        distribucion = f"t de Student con {n - 1} grados de libertad"
    else:
        valor_p = 2 * (1 - stats.norm.cdf(abs(estadistico)))
        distribucion = "normal estándar"

    rechaza = valor_p < 0.05
    if not rechaza:
        favorecido = "Sin diferencia significativa"
    elif estadistico < 0:
        favorecido = "Modelo de referencia"
    else:
        favorecido = "Modelo alternativo"

    return {
        "estadistico": round(float(estadistico), 4),
        "valor_p": round(float(valor_p), 4),
        "n": n,
        "horizonte": horizonte,
        "perdida": perdida,
        "distribucion": distribucion,
        "correccion_harvey": correccion_harvey,
        "decision": "Se rechaza H0" if rechaza else "No se rechaza H0",
        "modelo_favorecido": favorecido,
    }


# ---------------------------------------------------------------------------
# Diagnóstico de sobreajuste (apartado 4.6, etapa 5)
# ---------------------------------------------------------------------------

def diagnostico_sobreajuste(
    modelo,
    serie: pd.Series,
    resultado_validacion: pd.DataFrame,
    horizonte: int = 1,
) -> dict:
    """
    Contrasta el error dentro de la muestra de entrenamiento frente al error
    fuera de muestra.

    Una razón cercana a 1 indica que el modelo generaliza. Una razón muy
    inferior a 1 revela que el error de entrenamiento es artificialmente bajo,
    señal de memorización y no de aprendizaje (apartado 3.2.3).
    """
    X, y = modelo.matriz_entrenamiento(horizonte=horizonte)
    # Si el modelo aprende la variación, se suma el último dato conocido a
    # ajustados y observados para comparar niveles, igual que fuera de muestra.
    partida = modelo.nivel_de_partida(X) if hasattr(modelo, "nivel_de_partida") else 0.0
    ajustados = modelo.estimador.predict(X) + partida
    observados = y.values + partida

    # Si el modelo se entrenó en logaritmos, tanto los valores ajustados como
    # los observados están en esa escala. Hay que devolverlos a la escala
    # original antes de comparar, porque el error fuera de muestra se mide
    # siempre en las unidades de la serie. Omitir este paso produce una razón
    # cercana a cero que no significa nada.
    if getattr(modelo, "logaritmo", False):
        ajustados = np.exp(ajustados)
        observados = np.exp(observados)

    error_entrenamiento = float(np.mean(np.abs(observados - ajustados)))

    limpio = resultado_validacion.dropna(subset=["proyectado"])
    error_fuera = float(np.mean(np.abs(limpio["error"].values)))

    return {
        "MAE en entrenamiento": round(error_entrenamiento, 4),
        "MAE fuera de muestra": round(error_fuera, 4),
        "Razón entrenamiento/fuera": round(error_entrenamiento / error_fuera, 4),
    }


# ---------------------------------------------------------------------------
# Presentación de resultados
# ---------------------------------------------------------------------------

def tabla_desempeno(resultados: dict[str, dict]) -> pd.DataFrame:
    """
    Arma la Tabla 3 del Capítulo 5 a partir de un diccionario cuyas claves son
    los nombres de los modelos y cuyos valores son los diccionarios devueltos
    por metricas().
    """
    tabla = pd.DataFrame(resultados).T
    tabla.index.name = "Modelo"
    return tabla[["MAE", "RMSE", "MAPE", "MASE", "n"]]


def tabla_diebold_mariano(pruebas: list[dict]) -> pd.DataFrame:
    """
    Arma la Tabla 4 del Capítulo 5 a partir de una lista de resultados
    devueltos por diebold_mariano().
    """
    filas = [
        {
            "Horizonte": f"{p['horizonte']} mes" if p["horizonte"] == 1
            else f"{p['horizonte']} meses",
            "Estadístico": p["estadistico"],
            "Valor p": p["valor_p"],
            "Decisión": p["decision"],
            "Modelo favorecido": p["modelo_favorecido"],
        }
        for p in pruebas
    ]
    return pd.DataFrame(filas)
