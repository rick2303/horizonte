# Prototipo de sistema inteligente para la proyección explicable del IPC y de las remesas familiares

Código del componente técnico de la tesis. Cada módulo corresponde a una etapa
del procedimiento declarado en el apartado 4.6 y a un apartado teórico del
Capítulo 3, de modo que lo programado y lo escrito se puedan defender juntos.

## Estructura

```
sistema/
├── src/
│   ├── carga.py            Depuración de las hojas del BCH → serie mensual limpia
│   ├── caracteristicas.py  Ingeniería de variables (apartado 3.2.1)
│   ├── modelos.py          Ingenuo, ARIMA, bosque aleatorio, XGBoost (3.1.3 y 3.2.2)
│   ├── evaluacion.py       Origen móvil, métricas, Diebold-Mariano (3.3)
│   ├── experimento.py      Orquestador: produce las Tablas 3 y 4 del Capítulo 5
│   ├── explicabilidad.py   Valores SHAP: importancia global y explicación local (3.4.2)
│   ├── sus.py              Puntaje de la escala SUS y escala de adjetivos (3.5)
│   └── tablero.py          Datos de la pantalla principal y explicación del SARIMA
├── app/
│   ├── plantilla.html      Aplicación Horizonte (inicio, proyección, recomendaciones, calculadora, información)
│   ├── app.py              Publicación con Streamlit
│   ├── datos.json          Generado por ejecutar_todo.py
│   └── tablero.html        Generado: la misma pantalla, se abre con doble clic
├── preparar_datos.py       Depura los Excel del BCH y guarda los CSV
├── ejecutar_todo.py        Corre experimento, SHAP y genera la pantalla
├── datos/
│   ├── crudos/             Archivos tal como se descargan del BCH (no se suben)
│   └── procesados/         Series depuradas en CSV (sí se suben)
├── notebooks/              Ejecución en Google Colab
├── resultados/             Tablas, JSON y figuras (sí se suben)
├── pruebas/                Verificación del motor
└── requirements.txt
```

## Instalación (Windows, VS Code)

```bash
cd C:\tesis\sistema
.venv\Scripts\activate
pip install -r requirements.txt
```

## Paso 1. Depurar las series del BCH

Copie los Excel tal como se descargan a `datos/crudos/` y ejecute:

```bash
python preparar_datos.py --ipc datos/crudos/ipc.xlsx --hoja-ipc AR1 --remesas datos/crudos/remesas.xlsx
```

Reconoce los tres formatos habituales del BCH, recorta al período 2000-01 a
2025-12 (cámbielo con `--desde` y `--hasta`) e imprime el diagnóstico de cada
serie. Si algún cuadro no se reconoce, use directamente las funciones de
`src/carga.py`.

## Paso 2. Correr el experimento completo

```bash
python ejecutar_todo.py
```

Tarda varios minutos por serie. Deja en `resultados/` las tablas de desempeño,
Diebold-Mariano, sobreajuste, la importancia SHAP y las figuras PNG para el
Capítulo 5. Para probar sin datos reales: `python ejecutar_todo.py --sintetico`
(crea series falsas y un aviso `SINTETICO.txt`; bórrelo al cargar las reales).

## Paso 3. Abrir la aplicación

`ejecutar_todo.py` termina generando la pantalla principal con los datos
reales. Hay dos formas de verla:

- Doble clic en `app/tablero.html`: funciona sin internet salvo la fuente
  tipográfica, ideal para la defensa.
- `streamlit run app/app.py`: la misma pantalla servida por Streamlit, que es
  lo que se publica.

Si solo cambió el diseño, `python ejecutar_todo.py --solo-interfaz` regenera la
pantalla en segundos sin repetir el experimento.

## Paso 4. Publicar en Streamlit Community Cloud

1. Suba la carpeta a GitHub, incluidos `datos/procesados/`, `resultados/` y `app/datos.json`.
2. En share.streamlit.io elija el repositorio y como archivo principal `app/app.py`.

**Versión estática (GitHub Pages).** `docs/index.html` es una copia de `app/tablero.html` con los datos incluidos; GitHub Pages la publica sin los íconos de Streamlit. En GitHub: Settings → Pages → Deploy from a branch → `main` y carpeta `/docs`. Cada vez que cambie la plantilla o los datos, regenere `app/tablero.html` y cópielo a `docs/index.html`.

## Paso 5. Calcular la escala SUS

Registre las respuestas en Excel, una fila por participante y columnas P1 a P10:

```python
import pandas as pd
from src import sus

respuestas = pd.read_excel("resultados/sus.xlsx")
r = sus.resumen(respuestas)
print(r["media"], r["ic95"], r["adjetivo"])
```

## Verificación del motor

```bash
python -m pruebas.prueba_alineacion   # alineación de horizontes e ingenuo
python -m pruebas.prueba_ipc
python -m pruebas.prueba_remesas
```

## Decisiones metodológicas que el código implementa

- **Alineación de horizontes.** La fila del mes t usa información hasta t−1 y
  se empareja con el valor de t+h−1, de modo que el modelo aprende exactamente
  h meses de distancia, que es la misma distancia contra la que se le evalúa.
- **Remesas en logaritmos para ambos modelos.** El ARIMA y el XGBoost se
  estiman sobre el logaritmo de las remesas y sus proyecciones se devuelven a
  millones de dólares, para que la comparación de Diebold y Mariano sea entre
  modelos que ven la misma serie.
- **Convención de signos de Diebold y Mariano.** El diferencial de pérdida es
  `L(error ARIMA) − L(error ML)`. Un estadístico negativo indica mayor precisión
  del ARIMA y uno positivo mayor precisión del aprendizaje automático
  (apartado 4.3 y nota de la Tabla 4).
