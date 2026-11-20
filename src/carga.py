"""
Carga y depuración de las series mensuales del Banco Central de Honduras.

Corresponde a la fase de preparación de los datos de CRISP-DM (apartado 4.6,
etapa 3). El módulo recibe las hojas de cálculo tal como las publica el BCH,
con encabezados múltiples y celdas combinadas, y devuelve una serie mensual
limpia con índice temporal de frecuencia fija.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

ABREVIATURAS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}


def _sin_tildes(texto: str) -> str:
    """Elimina tildes y pasa a minúsculas para comparar nombres de meses."""
    normalizado = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in normalizado if not unicodedata.combining(c)).strip().lower()


def _mes_a_numero(valor) -> int | None:
    """Convierte 'Enero', 'ene', 'ENE.' o 1 en el número de mes."""
    if isinstance(valor, (int, np.integer)) and 1 <= int(valor) <= 12:
        return int(valor)
    limpio = re.sub(r"[^a-z]", "", _sin_tildes(valor))
    if limpio in MESES:
        return MESES[limpio]
    if limpio[:3] in ABREVIATURAS:
        return ABREVIATURAS[limpio[:3]]
    return None


def _a_numero(valor) -> float:
    """
    Convierte a número un valor que puede venir como texto con separadores
    de miles, espacios no separables o guiones que representan ausencia.
    """
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float, np.number)):
        return float(valor)
    texto = str(valor).strip()
    texto = texto.replace("\xa0", "").replace(" ", "")
    if texto in {"", "-", "--", "n.d.", "n/d", "..."}:
        return np.nan
    texto = texto.replace(",", "")
    try:
        return float(texto)
    except ValueError:
        return np.nan


def desde_columnas(
    ruta: str,
    columna_fecha: str,
    columna_valor: str,
    nombre: str,
    hoja: str | int = 0,
    filas_encabezado: int = 0,
) -> pd.Series:
    """
    Lee una serie que viene en formato largo, con una columna de fechas y una
    columna de valores.

    Parámetros
    ----------
    ruta : ruta del archivo .xlsx, .xls o .csv
    columna_fecha : nombre de la columna que contiene la fecha
    columna_valor : nombre de la columna que contiene el valor de la serie
    nombre : nombre que llevará la serie resultante
    hoja : nombre o posición de la hoja de cálculo
    filas_encabezado : número de filas a saltar antes del encabezado real
    """
    if str(ruta).lower().endswith(".csv"):
        tabla = pd.read_csv(ruta, skiprows=filas_encabezado)
    else:
        tabla = pd.read_excel(ruta, sheet_name=hoja, skiprows=filas_encabezado)

    fechas = pd.to_datetime(tabla[columna_fecha], errors="coerce")
    valores = tabla[columna_valor].map(_a_numero)

    serie = pd.Series(valores.values, index=fechas, name=nombre)
    return _consolidar(serie)


def desde_matriz_anio_mes(
    ruta: str,
    nombre: str,
    hoja: str | int = 0,
    columna_anio: str | int = 0,
    filas_encabezado: int = 0,
) -> pd.Series:
    """
    Lee una serie que viene en formato matricial: una fila por año y doce
    columnas de meses. Es el formato habitual de los cuadros estadísticos
    del BCH.

    Parámetros
    ----------
    ruta : ruta del archivo
    nombre : nombre que llevará la serie resultante
    hoja : nombre o posición de la hoja de cálculo
    columna_anio : nombre o posición de la columna que contiene el año
    filas_encabezado : número de filas a saltar antes del encabezado real
    """
    if str(ruta).lower().endswith(".csv"):
        tabla = pd.read_csv(ruta, skiprows=filas_encabezado)
    else:
        tabla = pd.read_excel(ruta, sheet_name=hoja, skiprows=filas_encabezado)

    if isinstance(columna_anio, int):
        etiqueta_anio = tabla.columns[columna_anio]
    else:
        etiqueta_anio = columna_anio

    columnas_mes = {}
    for columna in tabla.columns:
        if columna == etiqueta_anio:
            continue
        numero = _mes_a_numero(columna)
        if numero is not None:
            columnas_mes[columna] = numero

    if not columnas_mes:
        raise ValueError(
            "No se identificó ninguna columna de mes. Revise el encabezado "
            "del archivo o ajuste el parámetro filas_encabezado."
        )

    registros = []
    for _, fila in tabla.iterrows():
        anio = _a_numero(fila[etiqueta_anio])
        if pd.isna(anio) or not (1900 < anio < 2100):
            continue
        for columna, numero_mes in columnas_mes.items():
            valor = _a_numero(fila[columna])
            registros.append((pd.Timestamp(int(anio), numero_mes, 1), valor))

    if not registros:
        raise ValueError("No se extrajo ninguna observación válida del archivo.")

    fechas, valores = zip(*registros)
    serie = pd.Series(valores, index=pd.DatetimeIndex(fechas), name=nombre)
    return _consolidar(serie)


def desde_matriz_mes_anio(
    ruta: str,
    nombre: str,
    hoja: str | int = 0,
    columna_mes: str | int | None = None,
    filas_encabezado: int = 0,
) -> pd.Series:
    """
    Lee una serie en formato matricial transpuesto: una fila por mes y una
    columna por año. Algunos cuadros del BCH, como el del IPC base 1999,
    vienen así.

    Parámetros
    ----------
    ruta : ruta del archivo
    nombre : nombre que llevará la serie resultante
    hoja : nombre o posición de la hoja de cálculo
    columna_mes : nombre o posición de la columna con el nombre del mes; si se
        omite, se detecta la que contiene los nombres de los meses
    filas_encabezado : número de filas a saltar antes del encabezado real
    """
    if str(ruta).lower().endswith(".csv"):
        tabla = pd.read_csv(ruta, skiprows=filas_encabezado)
    else:
        tabla = pd.read_excel(ruta, sheet_name=hoja, skiprows=filas_encabezado)

    if columna_mes is None:
        # La columna de meses es la que más nombres de mes contiene.
        conteo = {c: tabla[c].map(lambda v: isinstance(v, str) and _mes_a_numero(v) is not None).sum()
                  for c in tabla.columns}
        etiqueta_mes = max(conteo, key=conteo.get)
    elif isinstance(columna_mes, int):
        etiqueta_mes = tabla.columns[columna_mes]
    else:
        etiqueta_mes = columna_mes

    columnas_anio = {}
    for columna in tabla.columns:
        if columna == etiqueta_mes:
            continue
        anio = _a_numero(columna)
        if not pd.isna(anio) and 1900 < anio < 2100:
            columnas_anio[columna] = int(anio)

    if not columnas_anio:
        raise ValueError(
            "No se identificó ninguna columna de año. Revise el encabezado "
            "del archivo o ajuste el parámetro filas_encabezado."
        )

    registros = []
    for _, fila in tabla.iterrows():
        numero_mes = _mes_a_numero(fila[etiqueta_mes])
        if numero_mes is None:
            continue
        for columna, anio in columnas_anio.items():
            registros.append((pd.Timestamp(anio, numero_mes, 1), _a_numero(fila[columna])))

    if not registros:
        raise ValueError("No se extrajo ninguna observación válida del archivo.")

    fechas, valores = zip(*registros)
    serie = pd.Series(valores, index=pd.DatetimeIndex(fechas), name=nombre)
    return _consolidar(serie)


def desde_filas_anio_mes(
    ruta: str,
    nombre: str,
    hoja: str | int = 0,
    columna_valor: int | None = None,
) -> pd.Series:
    """
    Lee una serie en formato largo: una fila por mes, con el año en una
    columna (a veces solo en la primera fila de cada año), el nombre del mes
    en otra y el valor en una tercera. Es el formato de la hoja de la serie
    mensual del IPC del BCH.

    Si no se indica columna_valor, se toma la primera columna numérica a la
    derecha de la columna del mes.
    """
    if str(ruta).lower().endswith(".csv"):
        tabla = pd.read_csv(ruta, header=None)
    else:
        tabla = pd.read_excel(ruta, sheet_name=hoja, header=None)

    # La columna del mes es la que más celdas con nombre de mes contiene.
    conteo = {c: tabla[c].map(lambda v: isinstance(v, str) and _mes_a_numero(v) is not None).sum()
              for c in tabla.columns}
    col_mes = max(conteo, key=conteo.get)
    if conteo[col_mes] < 12:
        raise ValueError("No se encontró una columna con nombres de meses.")
    posicion = list(tabla.columns).index(col_mes)

    # La columna del año es la de la izquierda con más valores entre 1900 y 2100.
    def es_anio(v):
        n = _a_numero(v)
        return not pd.isna(n) and 1900 < n < 2100 and float(n).is_integer()
    candidatas = list(tabla.columns)[:posicion]
    col_anio = max(candidatas, key=lambda c: tabla[c].map(es_anio).sum())

    if columna_valor is None:
        for c in list(tabla.columns)[posicion + 1:]:
            if tabla[c].map(lambda v: not pd.isna(_a_numero(v))).sum() >= 12:
                columna_valor = c
                break
    if columna_valor is None:
        raise ValueError("No se encontró la columna de valores.")

    anios = tabla[col_anio].map(lambda v: int(_a_numero(v)) if es_anio(v) else np.nan).ffill()
    registros = []
    for anio, mes, valor in zip(anios, tabla[col_mes], tabla[columna_valor]):
        numero = _mes_a_numero(mes) if isinstance(mes, str) else None
        if numero is None or pd.isna(anio):
            continue
        registros.append((pd.Timestamp(int(anio), numero, 1), _a_numero(valor)))

    fechas, valores = zip(*registros)
    serie = pd.Series(valores, index=pd.DatetimeIndex(fechas), name=nombre)
    return _consolidar(serie)


def desde_balanza_cambiaria(
    ruta: str,
    nombre: str,
    rubro: str = "Transferencias Corrientes",
    tipo: str = "Ingresos",
    hoja: str | int = 0,
    a_millones: bool = True,
) -> pd.Series:
    """
    Lee el reporte RPT-OC-065 de la balanza cambiaria del BCH, que viene en
    formato largo: una fila por mes y rubro, con las columnas Fecha,
    Variable Español, Descripción Español y Monto (miles de US$).

    Por omisión extrae los ingresos por transferencias corrientes, la serie
    que la tesis usa como remesas familiares (apartado 2.5), y la convierte de
    miles a millones de dólares.
    """
    crudo = pd.read_excel(ruta, sheet_name=hoja, header=None)
    fila = crudo.index[crudo.eq("Fecha").any(axis=1)][0]
    tabla = pd.read_excel(ruta, sheet_name=hoja, header=fila)
    filtro = (tabla["Variable Español"].astype(str).str.strip() == tipo) & (
        tabla["Descripción Español"].astype(str).str.strip() == rubro
    )
    datos = tabla.loc[filtro, ["Fecha", "Monto"]]
    if datos.empty:
        raise ValueError(f"No hay filas de {tipo} / {rubro} en {ruta}.")
    valores = datos["Monto"].map(_a_numero) / (1000 if a_millones else 1)
    serie = pd.Series(valores.values, index=pd.to_datetime(datos["Fecha"]), name=nombre)
    return _consolidar(serie)


def desde_remesas_departamento(ruta: str, nombre: str, columna: str = "Total Ingresos de Divisas") -> pd.Series:
    """
    Lee el cuadro de remesas familiares por departamento (desde 2018, miles
    de US$) y devuelve el total en millones. Es la serie exacta de remesas
    que la tesis usa como prueba de robustez.
    """
    crudo = pd.read_excel(ruta, header=None)
    fila = crudo.index[crudo.eq("Fecha").any(axis=1)][0]
    tabla = pd.read_excel(ruta, header=fila)
    serie = pd.Series(
        tabla[columna].map(_a_numero).values / 1000,
        index=pd.to_datetime(tabla["Fecha"], errors="coerce"),
        name=nombre,
    )
    return _consolidar(serie)


def detectar(ruta: str, nombre: str, hoja: str | int = 0) -> pd.Series:
    """
    Prueba los tres formatos habituales del BCH y devuelve el que produzca la
    serie más larga sin meses ausentes. Útil cuando no se sabe cómo viene el
    cuadro descargado.
    """
    # El reporte de la balanza cambiaria tiene un formato propio.
    try:
        crudo = pd.read_excel(ruta, sheet_name=hoja, header=None, nrows=10)
        if crudo.eq("Descripción Español").any(axis=None):
            serie = desde_balanza_cambiaria(ruta, nombre, hoja=hoja)
            print(f"{nombre}: balanza cambiaria, ingresos por transferencias corrientes en "
                  f"millones de US$, {serie.notna().sum()} observaciones, "
                  f"{serie.index[0]:%Y-%m} a {serie.index[-1]:%Y-%m}")
            return serie
    except ValueError:
        pass

    intentos = []
    for funcion in (desde_filas_anio_mes, desde_matriz_anio_mes, desde_matriz_mes_anio):
        for encabezado in ((None,) if funcion is desde_filas_anio_mes else range(0, 40)):
            try:
                if encabezado is None:
                    serie = funcion(ruta, nombre, hoja=hoja)
                else:
                    serie = funcion(ruta, nombre, hoja=hoja, filas_encabezado=encabezado)
            except Exception:
                continue
            intentos.append((int(serie.notna().sum()) - 5 * int(serie.isna().sum()), funcion.__name__, serie))
    if not intentos:
        raise ValueError(f"No se reconoció el formato de {ruta}. Revise la hoja o use una función específica.")
    intentos.sort(key=lambda t: t[0], reverse=True)
    puntaje, metodo, serie = intentos[0]
    print(f"{nombre}: formato {metodo}, {serie.notna().sum()} observaciones, "
          f"{serie.index[0]:%Y-%m} a {serie.index[-1]:%Y-%m}")
    return serie


def _consolidar(serie: pd.Series) -> pd.Series:
    """
    Deja la serie con índice mensual al inicio de mes, ordenada, sin fechas
    inválidas ni duplicadas, y con el calendario completo entre la primera y
    la última observación.
    """
    serie = serie[~serie.index.isna()]
    serie = serie.sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]
    serie.index = serie.index.to_period("M").to_timestamp()
    serie = serie.asfreq("MS")

    # Se recortan los meses sin valor al inicio y al final, que corresponden
    # a celdas vacías del cuadro original y no a datos ausentes.
    validos = serie.dropna()
    if validos.empty:
        raise ValueError("La serie quedó vacía después de la depuración.")
    serie = serie.loc[validos.index[0]: validos.index[-1]]
    return serie


def diagnostico(serie: pd.Series) -> pd.DataFrame:
    """
    Devuelve un resumen de la serie para documentar el apartado 5.1 y para
    verificar la continuidad de los registros exigida en el criterio de
    inclusión del apartado 4.4.1.
    """
    faltantes = serie[serie.isna()]
    resumen = {
        "Serie": serie.name,
        "Primera observación": serie.index.min().strftime("%Y-%m"),
        "Última observación": serie.index.max().strftime("%Y-%m"),
        "Observaciones": len(serie),
        "Valores ausentes": int(serie.isna().sum()),
        "Meses ausentes": ", ".join(faltantes.index.strftime("%Y-%m")) or "ninguno",
        "Mínimo": round(float(serie.min()), 4),
        "Máximo": round(float(serie.max()), 4),
        "Media": round(float(serie.mean()), 4),
        "Desviación estándar": round(float(serie.std()), 4),
    }
    return pd.DataFrame(resumen, index=[0]).T.rename(columns={0: "Valor"})


def guardar(serie: pd.Series, ruta: str) -> None:
    """Guarda la serie depurada en formato CSV con dos columnas."""
    serie.rename("valor").rename_axis("fecha").to_frame().to_csv(ruta)


def leer(ruta: str, nombre: str | None = None) -> pd.Series:
    """Vuelve a leer una serie previamente depurada y guardada con guardar()."""
    from pathlib import Path

    tabla = pd.read_csv(ruta, parse_dates=["fecha"], index_col="fecha")
    serie = tabla["valor"].asfreq("MS")
    # Path.stem funciona igual con rutas de Windows y de Colab.
    serie.name = nombre or Path(ruta).stem
    return serie
