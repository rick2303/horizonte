"""
Depura los archivos del BCH y deja las series listas en datos/procesados/.

    python preparar_datos.py --ipc datos/crudos/ipc.xlsx --remesas datos/crudos/remesas.xlsx

Opciones:
    --hoja-ipc, --hoja-remesas   nombre de la hoja si no es la primera (por ejemplo AR1)
    --desde, --hasta             período de estudio, por omisión 2000-01 a 2025-12

Reconoce solo los tres formatos habituales de los cuadros del BCH (filas
año-mes, matriz año por mes y matriz mes por año). Imprime el diagnóstico de
cada serie para verificar que no falten meses. Después ejecute
`python ejecutar_todo.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import carga

RAIZ = Path(__file__).resolve().parent
PROCESADOS = RAIZ / "datos" / "procesados"


def hoja(valor: str | None):
    if valor is None:
        return 0
    return int(valor) if valor.isdigit() else valor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ipc", required=True)
    parser.add_argument("--remesas", required=True)
    parser.add_argument("--hoja-ipc")
    parser.add_argument("--hoja-remesas")
    parser.add_argument("--desde", default="2000-01")
    parser.add_argument("--hasta", default="2025-12")
    args = parser.parse_args()

    PROCESADOS.mkdir(parents=True, exist_ok=True)
    for clave, ruta, nombre, h in (
        ("ipc", args.ipc, "IPC", args.hoja_ipc),
        ("remesas", args.remesas, "Remesas familiares", args.hoja_remesas),
    ):
        completa = carga.detectar(ruta, nombre, hoja=hoja(h))
        serie = completa.loc[args.desde: args.hasta]
        # Los meses publicados después del período de estudio no entran al
        # modelo: se guardan aparte para comprobar si las proyecciones
        # acertaron (la pantalla lo muestra como verificación).
        posterior = completa.loc[serie.index[-1] + pd.DateOffset(months=1):].dropna()
        ruta_posterior = PROCESADOS / f"{clave}_posterior.csv"
        if len(posterior):
            carga.guardar(posterior, ruta_posterior)
            print(f"  {len(posterior)} meses posteriores guardados para verificación.")
        elif ruta_posterior.exists():
            ruta_posterior.unlink()
        print(carga.diagnostico(serie).to_string())
        if serie.isna().any():
            print(f"  ATENCIÓN: {nombre} tiene meses ausentes; revíselos antes de continuar.")
        carga.guardar(serie, PROCESADOS / f"{clave}.csv")
        print()

    aviso = PROCESADOS / "SINTETICO.txt"
    if aviso.exists():
        aviso.unlink()
    print("Series guardadas en datos/procesados/. Siguiente paso: python ejecutar_todo.py")


if __name__ == "__main__":
    main()
