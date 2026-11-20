"""
Ejecuta de una sola vez todo lo que alimenta el Capítulo 5 y la aplicación.

    python ejecutar_todo.py               usa datos/procesados/ipc.csv y remesas.csv
    python ejecutar_todo.py --sintetico   genera series de prueba (NO son del BCH)
    python ejecutar_todo.py --solo ipc    corre una sola serie
    python ejecutar_todo.py --variacion   el XGBoost aprende el cambio, no el nivel

Produce en resultados/:
    desempeno_<serie>.csv           Tabla de desempeño por modelo y horizonte
    dm_<serie>.csv                  Tabla de Diebold y Mariano
    resumen_<serie>.json            Especificación ARIMA, escala MASE, sobreajuste
    pronosticos_<serie>_h*_*.csv    Pronósticos fuera de muestra de cada modelo
    shap_importancia_<serie>.csv    Importancia global de las variables
    figura_shap_<serie>.png         Figura de importancia global
    figura_cascada_<serie>.png      Explicación de la proyección del próximo mes

La validación de origen móvil reestima el ARIMA cientos de veces, así que la
corrida completa tarda varios minutos por serie.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src import carga, experimento, explicabilidad, modelos

RAIZ = Path(__file__).resolve().parent
PROCESADOS = RAIZ / "datos" / "procesados"
RESULTADOS = RAIZ / "resultados"

# Configuración de cada serie. La transformación logarítmica se activa en las
# remesas por su crecimiento multiplicativo (apartado 4.6, etapa 3).
SERIES = {
    "ipc": {"archivo": "ipc.csv", "nombre": "IPC", "logaritmo": False},
    "remesas": {"archivo": "remesas.csv", "nombre": "Remesas familiares", "logaritmo": True},
}


def generar_sinteticas() -> None:
    """Series de prueba con la forma de las reales. Se marcan en el archivo."""
    rng = np.random.default_rng(7)
    fechas = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(fechas))
    ipc = 50 * np.exp(0.0045 * t) + 0.6 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.25, len(t))
    rem = 180 * np.exp(0.0125 * t) * (1 + 0.18 * np.sin(2 * np.pi * (t - 2) / 12)) * np.exp(
        rng.normal(0, 0.06, len(t))
    )
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    carga.guardar(pd.Series(ipc, index=fechas, name="IPC"), PROCESADOS / "ipc.csv")
    carga.guardar(pd.Series(rem, index=fechas, name="Remesas"), PROCESADOS / "remesas.csv")
    (PROCESADOS / "SINTETICO.txt").write_text(
        "Las series de esta carpeta son sintéticas. Reemplácelas por las del BCH.\n",
        encoding="utf-8",
    )
    print("Series sintéticas generadas en datos/procesados/.\n")


def correr(clave: str, horizontes: tuple[int, ...], variacion: bool = False) -> None:
    config = SERIES[clave]
    ruta = PROCESADOS / config["archivo"]
    if not ruta.exists():
        print(f"No existe {ruta}. Depure primero la serie con src/carga.py.")
        return

    serie = carga.leer(str(ruta), nombre=config["nombre"])
    salida = experimento.ejecutar(
        serie, horizontes=horizontes, logaritmo=config["logaritmo"], variacion=variacion
    )
    experimento.guardar_resultados(salida, RESULTADOS, clave)

    print(salida["desempeno"].to_string(index=False))
    print()
    print(salida["diebold_mariano"].to_string(index=False))
    print("\nSobreajuste:", json.dumps(salida["sobreajuste"], ensure_ascii=False))

    # Explicabilidad sobre el modelo final entrenado con toda la serie, a un mes.
    modelo = modelos.Xgboost(logaritmo=config["logaritmo"], variacion=variacion).ajustar(serie, 1)
    importancia = explicabilidad.importancia_global(modelo)
    importancia.to_csv(RESULTADOS / f"shap_importancia_{clave}.csv", index=False)

    figura = explicabilidad.grafico_importancia(
        importancia, f"Importancia de las variables, {config['nombre']}"
    )
    figura.savefig(RESULTADOS / f"figura_shap_{clave}.png", dpi=200)

    local = explicabilidad.explicacion_local(modelo)
    figura = explicabilidad.grafico_cascada(
        local, f"Explicación de la proyección de {local['fecha_objetivo']:%m/%Y}"
    )
    figura.savefig(RESULTADOS / f"figura_cascada_{clave}.png", dpi=200)

    print("\nImportancia SHAP:")
    print(importancia.to_string(index=False))
    print("\n" + explicabilidad.texto_explicativo(local))
    print("\n" + "=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sintetico", action="store_true", help="genera series de prueba")
    parser.add_argument("--solo", choices=list(SERIES), help="corre una sola serie")
    parser.add_argument(
        "--variacion", action="store_true",
        help="el XGBoost aprende el cambio respecto del último dato, no el nivel",
    )
    parser.add_argument("--horizontes", default="1,3,6,12", help="por ejemplo 1,3")
    parser.add_argument(
        "--solo-interfaz", action="store_true",
        help="regenera la pantalla con los resultados ya guardados, sin repetir el experimento",
    )
    args = parser.parse_args()

    if args.sintetico:
        generar_sinteticas()

    if not args.solo_interfaz:
        horizontes = tuple(int(h) for h in args.horizontes.split(","))
        for clave in [args.solo] if args.solo else list(SERIES):
            correr(clave, horizontes, args.variacion)

    generar_interfaz()


def generar_interfaz() -> None:
    """
    Construye app/datos.json y app/tablero.html, la pantalla principal del
    prototipo, con las series depuradas y los resultados del experimento.
    """
    from src import tablero

    rutas = {c: PROCESADOS / SERIES[c]["archivo"] for c in SERIES}
    if not all(r.exists() for r in rutas.values()):
        print("Faltan series en datos/procesados/; no se generó la interfaz.")
        return
    ipc = carga.leer(str(rutas["ipc"]), nombre=SERIES["ipc"]["nombre"])
    remesas = carga.leer(str(rutas["remesas"]), nombre=SERIES["remesas"]["nombre"])
    print("Generando la interfaz (entrena los modelos finales con cada serie completa)…")
    posteriores = {
        c: carga.leer(str(PROCESADOS / f"{c}_posterior.csv"))
        for c in SERIES
        if (PROCESADOS / f"{c}_posterior.csv").exists()
    }
    datos = tablero.construir(
        ipc, remesas, RESULTADOS,
        ilustrativo=(PROCESADOS / "SINTETICO.txt").exists(),
        posteriores=posteriores,
    )
    carpeta = RAIZ / "app"
    (carpeta / "datos.json").write_text(
        json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tablero.generar_html(datos, carpeta / "plantilla.html", carpeta / "tablero.html")
    print("Listo: app/tablero.html (se abre con doble clic) y app/datos.json para Streamlit.")


if __name__ == "__main__":
    main()
