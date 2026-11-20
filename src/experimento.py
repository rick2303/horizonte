"""
Orquestación del experimento.

Ejecuta, para una serie y un conjunto de horizontes, el ciclo completo que
describe la etapa 5 del apartado 4.6: validación de origen móvil de los tres
modelos, cálculo de las cuatro métricas, contraste de Diebold y Mariano y
diagnóstico de sobreajuste.

La salida son directamente las tablas 3 y 4 del Capítulo 5.
"""

from __future__ import annotations

import time

import pandas as pd

from . import evaluacion, modelos


def ejecutar(
    serie: pd.Series,
    horizontes: tuple[int, ...] = (1, 3, 6, 12),
    logaritmo: bool = False,
    minimo_entrenamiento: int | None = None,
    modelo_ml: str = "xgboost",
    variacion: bool = False,
    verboso: bool = True,
) -> dict:
    """
    Ejecuta el experimento completo sobre una serie.

    Parámetros
    ----------
    serie : serie mensual depurada
    horizontes : horizontes de proyección a evaluar
    logaritmo : aplica transformación logarítmica en los modelos de
        aprendizaje automático. Se activa para las remesas, cuya trayectoria
        es de crecimiento multiplicativo, y se deja desactivada para el IPC.
    minimo_entrenamiento : tamaño de la ventana inicial. Se fija aquí una sola
        vez y se reutiliza en todos los modelos y horizontes, de modo que las
        series de errores sean comparables entre sí, que es la condición que
        exige la prueba de Diebold y Mariano.
    modelo_ml : 'xgboost' o 'bosque'
    variacion : si es True el modelo de aprendizaje automático aprende el
        cambio respecto del último dato y no el nivel (ver
        caracteristicas.construir). Corrige la incapacidad de los árboles
        para extrapolar una tendencia.

    Devuelve
    --------
    Diccionario con las claves 'desempeno', 'diebold_mariano', 'sobreajuste',
    'especificacion_arima' y 'pronosticos'.
    """
    serie = serie.dropna()
    n = len(serie)
    if minimo_entrenamiento is None:
        minimo_entrenamiento = max(60, int(n * 0.6))

    if verboso:
        print(f"Serie: {serie.name}")
        print(f"  Observaciones: {n}  ({serie.index[0]:%Y-%m} a {serie.index[-1]:%Y-%m})")
        print(f"  Ventana inicial de entrenamiento: {minimo_entrenamiento} meses")
        print(f"  Pronósticos fuera de muestra por horizonte: "
              f"{n - minimo_entrenamiento - min(horizontes) + 1} aprox.")

    # La especificación del ARIMA se selecciona una sola vez, sobre la ventana
    # inicial, y se reutiliza en toda la validación. Ver la nota de la clase
    # Arima sobre por qué no se reselecciona en cada origen.
    inicio = time.time()
    arima_base = modelos.Arima(logaritmo=logaritmo).seleccionar(
        serie.iloc[:minimo_entrenamiento]
    )
    if verboso:
        print(f"  Especificación ARIMA seleccionada: {arima_base.especificacion()} "
              f"(AIC {arima_base.aic:.2f}, {time.time() - inicio:.0f} s)")

    escala = evaluacion.escala_mase(serie.iloc[:minimo_entrenamiento])
    if verboso:
        print(f"  Escala del MASE (MAE del ingenuo en entrenamiento): {escala:.4f}\n")

    clase_ml = modelos.CATALOGO[modelo_ml]

    factorias = {
        "Modelo ingenuo estacional": lambda: modelos.IngenuoEstacional(),
        "ARIMA de referencia": lambda: modelos.Arima(
            arima_base.orden, arima_base.orden_estacional, logaritmo=logaritmo
        ),
        "Aprendizaje automático": lambda: clase_ml(logaritmo=logaritmo, variacion=variacion),
    }

    filas_desempeno = []
    filas_dm = []
    pronosticos = {}

    for h in horizontes:
        if minimo_entrenamiento + h > n:
            if verboso:
                print(f"  Horizonte {h}: omitido, la serie no alcanza.")
            continue

        resultados_h = {}
        for nombre, factoria in factorias.items():
            inicio = time.time()
            resultado = evaluacion.validacion_origen_movil(
                serie,
                factoria,
                horizonte=h,
                minimo_entrenamiento=minimo_entrenamiento,
            )
            resultados_h[nombre] = resultado
            metricas = evaluacion.metricas(resultado, escala)
            filas_desempeno.append({"Horizonte": h, "Modelo": nombre, **metricas})
            if verboso:
                print(f"  h={h:2d}  {nombre:28s} "
                      f"MAE {metricas['MAE']:10.4f}  MASE {metricas['MASE']:7.4f}  "
                      f"({time.time() - inicio:.0f} s)")

        pronosticos[h] = resultados_h

        # El contraste enfrenta al ARIMA (referencia) contra el modelo de
        # aprendizaje automático (alternativo), en ese orden, para que el
        # signo del estadístico siga la convención del apartado 4.3.
        #
        # Los dos vectores de error se emparejan por fecha y se eliminan de
        # forma conjunta los orígenes en los que cualquiera de los dos modelos
        # no produjo pronóstico. La prueba exige errores apareados sobre los
        # mismos períodos: descartarlos por separado desalinearía las series.
        pareados = pd.DataFrame(
            {
                "referencia": resultados_h["ARIMA de referencia"]["error"],
                "alternativo": resultados_h["Aprendizaje automático"]["error"],
            }
        ).dropna()

        prueba = evaluacion.diebold_mariano(
            pareados["referencia"],
            pareados["alternativo"],
            horizonte=h,
        )
        filas_dm.append(prueba)
        if verboso:
            print(f"        Diebold-Mariano: {prueba['estadistico']:+.4f}  "
                  f"p={prueba['valor_p']:.4f}  → {prueba['modelo_favorecido']}\n")

    # Diagnóstico de sobreajuste sobre el modelo de aprendizaje automático
    # entrenado con la serie completa, en el horizonte de un mes.
    if not pronosticos:
        raise ValueError("Ningún horizonte pudo evaluarse con esta serie.")
    h_diagnostico = min(pronosticos)
    modelo_final = clase_ml(logaritmo=logaritmo, variacion=variacion).ajustar(serie, h_diagnostico)
    sobreajuste = evaluacion.diagnostico_sobreajuste(
        modelo_final,
        serie,
        pronosticos[h_diagnostico]["Aprendizaje automático"],
        h_diagnostico,
    )

    return {
        "desempeno": pd.DataFrame(filas_desempeno),
        "diebold_mariano": evaluacion.tabla_diebold_mariano(filas_dm),
        "sobreajuste": sobreajuste,
        "especificacion_arima": arima_base.especificacion(),
        "escala_mase": escala,
        "ventana_inicial": minimo_entrenamiento,
        "pronosticos": pronosticos,
        "modelo_final": modelo_final,
        "variacion": variacion,
    }


def guardar_resultados(salida: dict, carpeta: str, etiqueta: str) -> None:
    """Guarda las tablas del experimento en archivos CSV."""
    import json
    from pathlib import Path

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    salida["desempeno"].to_csv(carpeta / f"desempeno_{etiqueta}.csv", index=False)
    salida["diebold_mariano"].to_csv(carpeta / f"dm_{etiqueta}.csv", index=False)
    for h, resultados in salida["pronosticos"].items():
        for nombre, tabla in resultados.items():
            clave = nombre.lower().replace(" ", "_").replace("á", "a").replace("é", "e")
            tabla.to_csv(carpeta / f"pronosticos_{etiqueta}_h{h}_{clave}.csv")

    # Datos de contexto que la aplicación muestra junto a las tablas.
    resumen = {
        "especificacion_arima": salida["especificacion_arima"],
        "escala_mase": salida["escala_mase"],
        "ventana_inicial": salida["ventana_inicial"],
        "sobreajuste": salida["sobreajuste"],
        "variacion": salida.get("variacion", False),
    }
    with open(carpeta / f"resumen_{etiqueta}.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)
