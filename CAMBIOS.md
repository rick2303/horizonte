# Cambios de esta versión (26 de octubre de 2026)

## Quinta entrega: Horizonte, la aplicación completa

La pantalla pasó de ser un solo tablero a una aplicación de cinco secciones con
nombre propio, **Horizonte** (el nombre está en `APP` y en los textos de
`app/plantilla.html`; se cambia con buscar y reemplazar).

- **Pantalla de carga** con el logotipo animado; se salta tocándola.
- **Barra de navegación** fija: Inicio, Proyección, Recomendaciones,
  Calculadora de inflación e Información.
- **Inicio** con gráfico interactivo (cambiar entre precios y remesas, deslizar
  la fecha de 1 a 12 meses), cuatro cifras del día con números animados,
  accesos a cada sección, explicación de cómo funciona y comparación de
  errores por plazo.
- **Color en las proyecciones:** el gráfico dibuja una cinta entre el último
  dato y la proyección (verde si es favorable, naranja si no: más precios es
  desfavorable, más remesas es favorable), franja de rango con degradado,
  puntos mensuales y etiqueta con el valor final. Nuevo interruptor para
  **comparar los dos modelos** en el mismo gráfico.
- **Ayuda al pasar el cursor** (o tocar, en celular) sobre cada cifra con
  subrayado punteado: indicadores, barras de la explicación, errores,
  prueba de Diebold y Mariano, tablas de verificación, calculadora y
  recomendaciones. Los símbolos ? siguen explicando las palabras técnicas.
- **Recomendaciones** según el perfil (hogar, familia que recibe remesas,
  negocio, analista o institución). Los números salen de las proyecciones;
  incluye el panorama de 12 meses, tres escenarios (bajo, central, alto) y un
  bloque «Qué tan confiable es esta guía».
- **Calculadora de inflación** con el IPC real y, para meses futuros, la
  proyección del modelo estadístico (con rango). Tres usos: ahorros, salario y
  precio o alquiler. Incluye la inflación de cada año desde 2000.
- **Información**: qué son el IPC y las remesas, fuentes, cómo se hicieron las
  proyecciones, resultados del examen, preguntas frecuentes, glosario con
  buscador y acerca del prototipo.
- **Nombres:** en pantalla, XGBoost aparece como **Inteligencia artificial**
  (machine learning) y SARIMA como **Modelo estadístico**; el nombre técnico
  queda como aclaración.
- **Streamlit:** `app/app.py` usa `st.iframe`, que reemplaza a
  `components.html` (Streamlit lo retira). Si la versión instalada es
  anterior, usa el método viejo. La altura se ajusta sola a cada sección.

Dos datos externos citados en la pantalla, con su fuente (revisar antes de
presentar): remesas de 2025 por US$ 12,212 millones y más de 25 % del PIB
(Revista EyN) y la inflación de junio de 2026 de 5.83 % (Infobae, 21 de
julio de 2026). Este último dato queda por encima del rango probable de la
proyección y se muestra como referencia externa, fuera del experimento. Para
quitarlo, ponga `EXTERNO = null` en `app/plantilla.html`.


## Cuarta entrega: pantalla para personas sin formación técnica

- Guía de bienvenida, pestañas numeradas y ayuda con el símbolo ? en cada
  término técnico (la misma definición aparece en el glosario de la pestaña 4).
- Pestaña 2: la explicación se lee como una suma (punto de partida + barras =
  proyección) y los términos del SARIMA tienen nombres llanos, con el nombre
  técnico al lado para la defensa.
- Pestaña 3: veredicto en una frase, barras de error promedio de cada modelo
  y la prueba de Diebold y Mariano traducida a "es real" o "no se puede
  asegurar". Las cuatro métricas quedan en un desplegable de detalle técnico.
- Comprobación con datos reales: los meses publicados después de diciembre de
  2025 (remesas de enero a agosto de 2026) se comparan con lo que dijo cada
  modelo. `preparar_datos.py` los guarda en `datos/procesados/*_posterior.csv`.


## Tercera entrega: datos oficiales del BCH y corrección del SARIMA

- **Selección del SARIMA corregida** (`modelos.py`). Antes el AIC elegía también
  d y D, lo cual no es válido porque al diferenciar cambian los datos y los AIC
  dejan de ser comparables. Con los datos reales eso producía para el IPC un
  modelo sin diferencias, peor que el ingenuo. Ahora d se fija con la prueba
  KPSS, D con la fuerza estacional de Wang, Smith y Hyndman (umbral 0.64), y
  el AIC elige p, q, P y Q (Hyndman y Khandakar, 2008). Con d = 1 y D = 0 se
  incluye una deriva, que en SARIMAX es la constante 'c' sobre la serie
  diferenciada.
- Especificaciones resultantes: IPC SARIMA(1,1,0)(1,0,1)[12] con deriva;
  remesas, sobre logaritmos, SARIMA(2,1,0)(1,0,1)[12] con deriva.
- `carga.desde_balanza_cambiaria` lee el reporte RPT-OC-065 y extrae los
  ingresos por transferencias corrientes en millones de US$.
  `carga.desde_remesas_departamento` lee la serie exacta por departamento.
- Período de estudio 2000-01 a 2025-12, 312 observaciones por serie.


## Segunda entrega: pantalla principal funcional

- `app/plantilla.html` reproduce el diseño elegido y ahora toma todos sus
  números del motor: historia, proyecciones a 1, 3, 6 y 12 meses, intervalo
  del 95 %, explicación, desempeño y Diebold-Mariano.
- La explicación del SARIMA descompone su propia ecuación de forma exacta
  (punto de partida, cambio estacional del año anterior, términos AR y MA); la
  del XGBoost usa valores SHAP. En ambos casos las barras suman la proyección.
- El intervalo del XGBoost sale de sus errores en la validación de origen
  móvil (percentiles 2.5 y 97.5), porque el modelo no produce intervalos.
- La etiqueta Datos ilustrativos solo aparece con series sintéticas; con las
  del BCH dice Datos oficiales.
- `preparar_datos.py` reconoce el formato de los Excel del BCH.

## Primera entrega

## Correcciones que obligan a volver a correr el experimento del IPC

1. **Desfase de un mes en el XGBoost** (`caracteristicas.py`). La fila de
   entrenamiento usaba información hasta t−1 y se emparejaba con t+h, así que el
   modelo aprendía a proyectar h+1 meses. Al proyectar, la fila arrancaba en el
   último dato y se comparaba contra t+h. Ahora ambas partes miden exactamente h
   meses. La prueba `python -m pruebas.prueba_alineacion` lo verifica.
   En el IPC sintético la conclusión no cambió: SARIMA sigue siendo mejor.

2. **Ingenuo estacional** (`modelos.py`). Tomaba el valor de 11 meses atrás en
   lugar de 12, y con h = 12 devolvía el primer dato de la serie. Esto afectaba
   su MAE y su MASE en las tablas.

3. **El mes codificado** ahora es el del mes que se proyecta y no el del mes de
   la fila, que es la estacionalidad que de verdad importa.

## Cambio metodológico en remesas

- El ARIMA también se estima sobre el logaritmo de las remesas, igual que el
  XGBoost. Antes solo el XGBoost usaba logaritmos, y la prueba de Diebold y
  Mariano comparaba modelos que veían series distintas. Conviene mencionarlo en
  el apartado 4.6, etapa 3.

## Opción nueva, apagada por defecto: `--variacion`

Los árboles de decisión no pueden proyectar valores mayores que los que vieron
al entrenar. En una serie con tendencia, como el IPC, todo valor futuro supera
el máximo histórico, y el XGBoost se queda plano. Esto también explica buena
parte de la razón de sobreajuste tan baja: no es solo memorización, es
incapacidad de extrapolar.

Con `python ejecutar_todo.py --variacion` el XGBoost aprende el **cambio**
respecto del último dato y luego se lo suma. Resultado en las series
sintéticas, contra el mismo ARIMA:

| Serie | Horizonte | MASE ARIMA | MASE XGBoost nivel | MASE XGBoost variación | DM con variación |
|---|---|---|---|---|---|
| IPC | 1 | 0.061 | 0.259 | 0.070 | −2.57, p = 0.011, gana ARIMA |
| IPC | 12 | 0.145 | 2.04 | 0.154 | +0.18, p = 0.858, sin diferencia |
| Remesas | 1 | 1.935 | 3.80 | 2.442 | −2.18, p = 0.031, gana ARIMA |
| Remesas | 12 | 2.072 | 6.69 | 2.758 | −1.37, p = 0.173, sin diferencia |

Son datos sintéticos; con los del BCH las cifras serán otras. Es una decisión
metodológica que debe consultar con su asesor antes de adoptarla.

## Módulos nuevos

- `src/explicabilidad.py`: valores SHAP, importancia global, explicación de la
  proyección del próximo mes en lenguaje llano y figuras para el Capítulo 5.
- `src/sus.py`: puntaje SUS, intervalo de confianza y escala de adjetivos.
- `app/app.py`: aplicación Streamlit con cuatro pestañas.
- `ejecutar_todo.py`: corre experimento, SHAP y figuras de las dos series.
- `carga.desde_matriz_mes_anio`: lector para cuadros con una fila por mes.
- `.gitignore`: ahora sube `datos/procesados/` y `resultados/`, que la
  aplicación publicada necesita.

## Sexta entrega (26 oct 2026)
- **Modo oscuro**: botón sol/luna en la barra superior. Se recuerda la elección y, la primera vez, sigue la preferencia del dispositivo. Los gráficos, tablas y colores de las proyecciones cambian con el tema.
- **Barra de navegación siempre visible**, tanto en `tablero.html` como dentro de Streamlit (se desplaza con la página para quedar pegada arriba).
