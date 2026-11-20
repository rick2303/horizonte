"""
Capa de explicabilidad con valores SHAP.

Corresponde al apartado 3.4.2 y a la etapa 6 del apartado 4.6. Para un modelo
basado en árboles ya entrenado, descompone cada proyección en la suma de un
valor base y de la contribución de cada variable predictora (Lundberg y Lee,
2017, sobre la noción de valor de Shapley de 1953):

    proyección = valor_base + suma_j phi_j

El módulo produce dos lecturas:

  - global: qué variables pesan más, en promedio, en todas las proyecciones
    del período de entrenamiento;
  - local: por qué la proyección del próximo mes tiene el valor que tiene.

Cuando el modelo se entrenó en logaritmos, como ocurre con las remesas, las
contribuciones están en unidades logarítmicas. Una contribución phi equivale
a multiplicar la proyección por exp(phi), es decir, a un efecto aproximado de
(exp(phi) - 1) x 100 por ciento. El módulo reporta ese porcentaje para que el
usuario no tenga que interpretar logaritmos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import caracteristicas

# Las dos componentes del mes son una sola idea para el usuario. Como los
# valores SHAP son aditivos, sumarlas es exacto y no una aproximación.
GRUPOS = {"mes_seno": "Época del año", "mes_coseno": "Época del año"}


def _etiqueta(variable: str, agrupar: bool) -> str:
    if agrupar and variable in GRUPOS:
        return GRUPOS[variable]
    return caracteristicas.nombres_legibles().get(variable, variable)


def _explicador(modelo):
    if modelo.estimador is None:
        raise ValueError("El modelo debe estar ajustado antes de explicarlo.")
    return shap.TreeExplainer(modelo.estimador)


def _valor_base(explicador) -> float:
    base = explicador.expected_value
    return float(np.ravel(base)[0])


def valores_entrenamiento(modelo) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Calcula los valores SHAP de todas las filas de entrenamiento.

    Devuelve la matriz X, la matriz de valores SHAP con las mismas filas y
    columnas, y el valor base del modelo.
    """
    explicador = _explicador(modelo)
    X, _ = modelo.matriz_entrenamiento()
    X = X[modelo.columnas]
    valores = explicador.shap_values(X)
    tabla = pd.DataFrame(valores, index=X.index, columns=X.columns).astype(float)
    return X, tabla, _valor_base(explicador)


def importancia_global(modelo, agrupar: bool = True) -> pd.DataFrame:
    """
    Importancia media de cada variable: promedio del valor absoluto de su
    contribución SHAP sobre todas las filas de entrenamiento.

    La columna 'Participación (%)' expresa cada importancia como proporción
    del total, que es la cifra más fácil de citar en el Capítulo 5.
    """
    _, tabla, _ = valores_entrenamiento(modelo)
    if agrupar:
        tabla = tabla.T.groupby(lambda v: _etiqueta(v, True)).sum().T
    else:
        tabla = tabla.rename(columns=lambda v: _etiqueta(v, False))

    importancia = tabla.abs().mean().sort_values(ascending=False)
    resultado = pd.DataFrame(
        {
            "Variable": importancia.index,
            "Importancia media |SHAP|": importancia.values,
            "Participación (%)": 100 * importancia.values / importancia.sum(),
        }
    )
    return resultado.round(4).reset_index(drop=True)


def explicacion_local(modelo, agrupar: bool = True) -> dict:
    """
    Descompone la proyección que el modelo genera con los últimos datos.

    Devuelve un diccionario con:
      fecha_objetivo : mes proyectado
      valor_base : proyección promedio del modelo, en la escala de la serie
      proyeccion : proyección final, en la escala de la serie
      contribuciones : DataFrame ordenado por magnitud con el valor de cada
          variable, su contribución y, si hay logaritmos, su efecto en %
      logaritmo : si el modelo se entrenó en logaritmos
    """
    explicador = _explicador(modelo)
    fila = modelo.fila_prediccion()
    valores = np.ravel(explicador.shap_values(fila))
    # Con variacion=True el árbol explica el cambio; el punto de partida es
    # entonces el último dato más el cambio promedio, y las contribuciones
    # explican cuánto se aparta el cambio proyectado de ese promedio.
    base = _valor_base(explicador) + float(modelo.nivel_de_partida(fila)[0])

    detalle = pd.DataFrame(
        {
            "variable": fila.columns,
            "valor_variable": fila.iloc[0].values,
            "contribucion": valores,
        }
    )
    if modelo.logaritmo:
        # El valor de la variable se devuelve a la escala original salvo en
        # las componentes del mes, que no son niveles de la serie.
        nivel = detalle["variable"].str.startswith(("rezago_", "media_movil_"))
        detalle.loc[nivel, "valor_variable"] = np.exp(detalle.loc[nivel, "valor_variable"])

    detalle["Variable"] = detalle["variable"].map(lambda v: _etiqueta(v, agrupar))
    agrupado = (
        detalle.groupby("Variable", sort=False)
        .agg(valor_variable=("valor_variable", "first"), contribucion=("contribucion", "sum"))
        .reset_index()
    )
    # Para el grupo 'Época del año' el valor mostrado es el mes proyectado.
    fecha_objetivo = fila.index[0]
    agrupado.loc[agrupado["Variable"] == "Época del año", "valor_variable"] = fecha_objetivo.month

    if modelo.logaritmo:
        agrupado["efecto_pct"] = (np.exp(agrupado["contribucion"]) - 1) * 100
        valor_base = float(np.exp(base))
        proyeccion = float(np.exp(base + valores.sum()))
    else:
        valor_base = base
        proyeccion = base + float(valores.sum())

    agrupado = agrupado.reindex(
        agrupado["contribucion"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return {
        "fecha_objetivo": fecha_objetivo,
        "valor_base": valor_base,
        "proyeccion": proyeccion,
        "contribuciones": agrupado,
        "logaritmo": bool(modelo.logaritmo),
        "variacion": bool(getattr(modelo, "variacion", False)),
    }


def texto_explicativo(explicacion: dict, unidad: str = "", principales: int = 3) -> str:
    """
    Redacta en lenguaje llano la explicación de la proyección, para que un
    usuario sin formación estadística entienda qué la determina.
    """
    mes = _MESES[explicacion["fecha_objetivo"].month - 1]
    anio = explicacion["fecha_objetivo"].year
    sufijo = f" {unidad}" if unidad else ""
    if explicacion.get("variacion"):
        punto_partida = (
            f"Su punto de partida es el último dato más el cambio que la serie suele "
            f"tener en ese plazo, {explicacion['valor_base']:,.2f}{sufijo}."
        )
    else:
        punto_partida = (
            f"Si no supiera nada del momento actual, proyectaría "
            f"{explicacion['valor_base']:,.2f}{sufijo}, que es su valor promedio."
        )
    partes = [
        f"Para {mes} de {anio} el modelo proyecta {explicacion['proyeccion']:,.2f}{sufijo}. "
        f"{punto_partida} La diferencia se explica sobre todo por:"
    ]
    for _, fila in explicacion["contribuciones"].head(principales).iterrows():
        direccion = "eleva" if fila["contribucion"] > 0 else "reduce"
        if explicacion["logaritmo"]:
            magnitud = f"{abs(fila['efecto_pct']):.1f} %"
        else:
            magnitud = f"{abs(fila['contribucion']):,.2f}{sufijo}"
        partes.append(f"- {fila['Variable']}: {direccion} la proyección en {magnitud}.")
    return "\n".join(partes)


_MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


# ---------------------------------------------------------------------------
# Gráficos para el Capítulo 5 y para la aplicación
# ---------------------------------------------------------------------------

def grafico_importancia(importancia: pd.DataFrame, titulo: str = ""):
    """Barras horizontales de la importancia global. Devuelve la figura."""
    import matplotlib.pyplot as plt

    datos = importancia.iloc[::-1]
    fig, eje = plt.subplots(figsize=(8, 0.42 * len(datos) + 1.2))
    eje.barh(datos["Variable"], datos["Participación (%)"], color="#1f4e79")
    eje.set_xlabel("Participación en la importancia total (%)")
    eje.set_title(titulo)
    eje.spines[["top", "right"]].set_visible(False)
    for posicion, valor in enumerate(datos["Participación (%)"]):
        eje.text(valor + 0.5, posicion, f"{valor:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def grafico_cascada(explicacion: dict, titulo: str = "", principales: int = 8):
    """
    Gráfico de cascada de la explicación local: parte del valor base y suma
    una a una las contribuciones hasta llegar a la proyección. Las variables
    restantes se agrupan en 'Otras variables'.
    """
    import matplotlib.pyplot as plt

    contrib = explicacion["contribuciones"]
    log = explicacion["logaritmo"]
    cabeza = contrib.head(principales)
    resto = contrib.iloc[principales:]["contribucion"].sum()

    etiquetas = list(cabeza["Variable"])
    pasos = list(cabeza["contribucion"])
    if len(contrib) > principales:
        etiquetas.append("Otras variables")
        pasos.append(resto)

    # Se trabaja en la escala del modelo y se convierte cada punto acumulado,
    # para que en remesas la cascada termine exactamente en la proyección.
    base_modelo = np.log(explicacion["valor_base"]) if log else explicacion["valor_base"]
    acumulado = [base_modelo]
    for paso in pasos:
        acumulado.append(acumulado[-1] + paso)
    escala = np.exp(acumulado) if log else np.array(acumulado)

    fig, eje = plt.subplots(figsize=(8, 0.45 * (len(pasos) + 2) + 1.2))
    filas = ["Punto de partida del modelo"] + etiquetas + ["Proyección"]
    for i in range(len(pasos)):
        inicio, fin = escala[i], escala[i + 1]
        color = "#2e7d32" if fin >= inicio else "#c62828"
        eje.barh(i + 1, fin - inicio, left=inicio, color=color)
    eje.barh(0, 0.0001, left=escala[0], color="#555")
    eje.axvline(escala[0], color="#999", lw=0.8, ls="--")
    eje.axvline(escala[-1], color="#1f4e79", lw=1.2)
    eje.set_yticks(range(len(filas)))
    eje.set_yticklabels(filas)
    eje.invert_yaxis()
    eje.set_title(titulo)
    eje.spines[["top", "right"]].set_visible(False)
    eje.text(escala[0], 0, f" {escala[0]:,.2f}", va="center", fontsize=9)
    eje.text(escala[-1], len(filas) - 1, f" {escala[-1]:,.2f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig
