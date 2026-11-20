"""
Cálculo de la Escala de Usabilidad del Sistema (SUS).

Corresponde al apartado 3.5 y a la etapa 7 del apartado 4.6. Cada
participante responde diez afirmaciones en una escala de 1 a 5. Las
afirmaciones impares están redactadas en positivo y las pares en negativo,
por eso se puntúan distinto (Brooke, 1996):

    impares: respuesta - 1
    pares:   5 - respuesta
    puntaje = suma de los diez aportes x 2.5      (rango de 0 a 100)

La interpretación sigue la escala de adjetivos de Bangor et al. (2008).

Uso típico: se registran las respuestas en una hoja con una fila por
participante y diez columnas P1 a P10, se lee con pandas y se llama a
resumen().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PREGUNTAS = [
    "Creo que me gustaría utilizar este sistema con frecuencia.",
    "Encontré el sistema innecesariamente complejo.",
    "Pensé que el sistema era fácil de usar.",
    "Creo que necesitaría el apoyo de un técnico para poder utilizar este sistema.",
    "Encontré que las diversas funciones de este sistema estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en este sistema.",
    "Imagino que la mayoría de las personas aprendería a usar este sistema muy rápidamente.",
    "Encontré el sistema muy difícil de usar.",
    "Me sentí muy seguro al utilizar el sistema.",
    "Necesité aprender muchas cosas antes de poder empezar a usar este sistema.",
]

COLUMNAS = [f"P{i}" for i in range(1, 11)]


def puntaje(respuestas) -> float:
    """Puntaje SUS de un participante a partir de sus diez respuestas."""
    r = np.asarray(respuestas, dtype=float)
    if r.shape != (10,) or np.any((r < 1) | (r > 5)):
        raise ValueError("Se esperan diez respuestas entre 1 y 5.")
    impares = r[0::2] - 1
    pares = 5 - r[1::2]
    return float((impares.sum() + pares.sum()) * 2.5)


def adjetivo(valor: float) -> str:
    """
    Categoría de la escala de adjetivos de Bangor et al. Los umbrales son los
    puntajes medios asociados a cada adjetivo (85.58, 72.75, 52.01 y 35.71),
    que la literatura usa habitualmente como puntos de corte. Conviene
    verificar las cifras contra la fuente citada en la bibliografía.
    """
    if valor >= 85.58:
        return "Excelente"
    if valor >= 72.75:
        return "Buena"
    if valor >= 52.01:
        return "Aceptable"
    if valor >= 35.71:
        return "Pobre"
    return "Inaceptable"


def resumen(tabla: pd.DataFrame, columnas: list[str] = COLUMNAS) -> dict:
    """
    Puntaje por participante y estadísticos del grupo, con intervalo de
    confianza del 95 % basado en la t de Student, apropiado para diez casos.
    """
    puntajes = tabla[columnas].apply(puntaje, axis=1)
    n = len(puntajes)
    media = float(puntajes.mean())
    desviacion = float(puntajes.std(ddof=1)) if n > 1 else 0.0
    margen = float(stats.t.ppf(0.975, n - 1) * desviacion / np.sqrt(n)) if n > 1 else 0.0
    return {
        "puntajes": puntajes.rename("SUS"),
        "n": n,
        "media": round(media, 2),
        "desviacion": round(desviacion, 2),
        "ic95": (round(media - margen, 2), round(media + margen, 2)),
        "minimo": float(puntajes.min()),
        "maximo": float(puntajes.max()),
        "adjetivo": adjetivo(media),
        "supera_68": media > 68,
    }
