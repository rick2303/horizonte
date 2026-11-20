"""
Aplicación del prototipo publicada con Streamlit.

    streamlit run app/app.py

Horizonte: muestra la aplicación completa (app/plantilla.html: inicio, proyección,
recomendaciones, calculadora de inflación e información) con los datos que deja
ejecutar_todo.py en app/datos.json. Todos los números salen del motor en
Python: series del BCH depuradas, modelos reestimados con la serie completa,
valores SHAP y resultados de la validación de origen móvil.

Para actualizar los datos, depure las series nuevas y ejecute
`python ejecutar_todo.py`; si solo cambió la interfaz,
`python ejecutar_todo.py --solo-interfaz`.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

CARPETA = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Horizonte · Proyecciones de precios y remesas de Honduras",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# La pantalla ocupa todo el ancho: se quitan los márgenes de Streamlit.
st.markdown(
    """
    <style>
      .block-container {padding: 0 !important; max-width: 100% !important;}
      header[data-testid="stHeader"] {display: none;}
      /* Sin franjas blancas arriba de la barra ni abajo del pie: el fondo de
         Streamlit toma el azul del pie y no queda espacio entre elementos. */
      [data-testid="stAppViewContainer"] {background: #0B2545;}
      .block-container [data-testid="stVerticalBlock"] {gap: 0 !important;}
      [data-testid="stElementContainer"]:has(style) {display: none;}
      [data-testid="stElementContainer"]:has(iframe) {margin: 0 !important;}
      iframe {display: block;}
      /* Botones de descarga de respaldo, dentro del mismo azul del pie */
      .st-key-respaldo {background: #0B2545; margin-top: -8px; padding: 8px 24px 28px;
        position: relative; z-index: 1;}
      .st-key-respaldo p {color: #8FB8EA; font-size: 13px; margin: 0;}
      .st-key-respaldo [data-testid="stHorizontalBlock"] {max-width: 1360px; margin: 0 auto;
        justify-content: flex-start; align-items: center; flex-wrap: wrap; gap: 8px 12px;}
      .st-key-respaldo [data-testid="stColumn"] {width: auto !important; flex: 0 0 auto !important; min-width: 0;}
      .st-key-respaldo button {background: transparent; color: #BFD3EE; border: 1px solid #2C4767;
        border-radius: 8px; font-size: 14px; min-height: 40px;}
      .st-key-respaldo button:hover {border-color: #8FB8EA; color: #FFFFFF;}
    </style>
    """,
    unsafe_allow_html=True,
)

ruta_datos = CARPETA / "datos.json"
if not ruta_datos.exists():
    st.error(
        "No se encontró app/datos.json. Ejecute `python ejecutar_todo.py` desde la "
        "carpeta sistema/ para generar los resultados y la interfaz."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def pagina(marca_tiempo: float) -> tuple[str, dict]:
    datos = json.loads(ruta_datos.read_text(encoding="utf-8"))
    html = (CARPETA / "plantilla.html").read_text(encoding="utf-8")
    contenido = json.dumps(datos, ensure_ascii=False).replace("</", "<\\/")
    return html.replace("/*__DATOS__*/null", contenido), datos


# La caché se renueva si cambian los datos o la plantilla.
html, datos = pagina(ruta_datos.stat().st_mtime + (CARPETA / "plantilla.html").stat().st_mtime)
# Streamlit reciente: st.iframe mide solo la altura del contenido. Las versiones
# anteriores usan components.html (que Streamlit retirará); la página también
# ajusta la altura del marco por su cuenta.
if hasattr(st, "iframe"):
    st.iframe(html, height="content")
else:
    components.html(html, height=1100, scrolling=True)

# Respaldo de descarga: algunos navegadores bloquean descargas desde el marco.
with st.container(key="respaldo"):
    columnas = st.columns(3)
    columnas[0].markdown("¿No se descargó el CSV? Use estos enlaces:")
    for columna, (clave, nombre) in zip(columnas[1:], (("ipc", "IPC"), ("rem", "remesas"))):
        serie = datos["series"][clave]
        filas = ["mes,modelo_estadistico,estadistico_inferior_95,estadistico_superior_95,inteligencia_artificial,ia_inferior_95,ia_superior_95"]
        for i, mes in enumerate(serie["fechas_proyeccion"]):
            a, x = serie["proyecciones"]["sarima"][i], serie["proyecciones"]["xgb"][i]
            valores = [a["valor"], a["inferior"], a["superior"], x["valor"], x["inferior"], x["superior"]]
            filas.append(mes + "," + ",".join("" if v is None else f"{v:.4f}" for v in valores))
        columna.download_button(
            f"Descargar proyección {nombre} (CSV)",
            ("﻿" + "\n".join(filas)).encode("utf-8"),
            file_name=f"proyeccion_{nombre.lower()}.csv",
            mime="text/csv",
        )
