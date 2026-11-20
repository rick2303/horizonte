"""
Modelos de pronóstico.

Implementa los tres niveles previstos en el apartado 4.6, etapa 4:

  1. Modelo ingenuo estacional, que fija el desempeño mínimo exigible.
  2. Modelo econométrico de referencia (SARIMA), del apartado 3.1.3.
  3. Modelos de aprendizaje automático basados en árboles, del apartado 3.2.2.

Todos exponen la misma interfaz —ajustar(serie, horizonte) y proyectar()—
para que el módulo de evaluación pueda tratarlos de manera indistinta dentro
de la validación de origen móvil.
"""

from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from . import caracteristicas

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Modelo:
    """Interfaz común. Toda subclase debe implementar ajustar y proyectar."""

    nombre = "modelo"

    def ajustar(self, serie: pd.Series, horizonte: int) -> "Modelo":
        raise NotImplementedError

    def proyectar(self) -> float:
        raise NotImplementedError


class IngenuoEstacional(Modelo):
    """
    Repite el valor observado en el mismo mes del año anterior.

    No tiene parámetros que estimar. Su función es servir de referencia
    mínima: un modelo que no lo supere no aporta nada, y su error es además
    el denominador de la métrica MASE.
    """

    nombre = "Ingenuo estacional"

    def __init__(self, periodo: int = 12):
        self.periodo = periodo
        self.serie = None
        self.horizonte = 1

    def ajustar(self, serie: pd.Series, horizonte: int) -> "IngenuoEstacional":
        self.serie = serie
        self.horizonte = horizonte
        return self

    def proyectar(self) -> float:
        # Con datos hasta T (posición -1), el mes T+h se proyecta con el valor
        # de T+h-12, que ocupa la posición h-13. Para h=12 es el propio T.
        # El horizonte no puede exceder el período estacional, porque ese
        # valor todavía no estaría observado.
        if self.horizonte > self.periodo:
            raise ValueError("El ingenuo estacional admite horizontes de hasta 12 meses.")
        posicion = self.horizonte - self.periodo - 1
        return float(self.serie.iloc[posicion])


class Arima(Modelo):
    """
    Modelo autorregresivo integrado de media móvil con componente estacional.

    La especificación se selecciona una sola vez, sobre la ventana inicial de
    entrenamiento, mediante búsqueda en malla con el criterio de información
    de Akaike. En cada origen posterior de la validación el modelo se vuelve a
    estimar con la misma especificación, pero no se reselecciona. Esto responde
    a dos razones: reproduce lo que haría un analista en la práctica, que no
    reidentifica el modelo cada mes, y evita que la selección se contamine con
    información de toda la muestra.
    """

    nombre = "ARIMA de referencia"

    def __init__(
        self,
        orden=None,
        orden_estacional=None,
        periodo: int = 12,
        logaritmo: bool = False,
    ):
        self.orden = orden
        self.orden_estacional = orden_estacional
        self.periodo = periodo
        self.logaritmo = logaritmo
        self.horizonte = 1
        self.resultado = None

    def _transformar(self, serie: pd.Series) -> pd.Series:
        # Para las remesas el ARIMA se estima sobre el logaritmo, igual que el
        # modelo de aprendizaje automático. Así ambos modelos enfrentan la
        # misma serie y la comparación de Diebold y Mariano es justa; además
        # es la práctica habitual de Box y Jenkins ante crecimiento
        # multiplicativo, porque estabiliza la varianza.
        if not self.logaritmo:
            return serie
        if (serie <= 0).any():
            raise ValueError("El logaritmo exige valores positivos.")
        return np.log(serie)

    @staticmethod
    def _tendencia(d: int, D: int):
        """
        Con una sola diferencia regular y ninguna estacional, el modelo
        incluye una deriva: el crecimiento promedio por mes. Sin ella, la
        proyección de una serie con tendencia se aplana. Con d = 1 y D = 1 la
        deriva queda absorbida por las diferencias y no se incluye.
        En SARIMAX la tendencia se aplica a la serie ya diferenciada, así que
        la deriva se declara como constante 'c' (con 't' sería una tendencia
        cuadrática en niveles).
        """
        return "c" if (d, D) == (1, 0) else None

    @staticmethod
    def diferencias_regulares(serie: pd.Series, max_d: int = 2, alfa: float = 0.05) -> int:
        """
        Número de diferencias regulares según la prueba KPSS: se diferencia
        mientras se rechace la hipótesis nula de estacionariedad.
        """
        from statsmodels.tsa.stattools import kpss

        x = serie.dropna()
        for d in range(max_d + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                valor_p = kpss(x, regression="c", nlags="auto")[1]
            if valor_p >= alfa:
                return d
            x = x.diff().dropna()
        return max_d

    @staticmethod
    def fuerza_estacional(serie: pd.Series, periodo: int = 12) -> float:
        """
        Fuerza de la estacionalidad (Wang, Smith y Hyndman, 2006):
        max(0, 1 - Var(resto) / Var(estacional + resto)) en una
        descomposición STL. Va de 0 (sin estacionalidad) a 1.
        """
        from statsmodels.tsa.seasonal import STL

        partes = STL(serie.dropna(), period=periodo, robust=True).fit()
        return float(max(0.0, 1 - np.var(partes.resid) / np.var(partes.seasonal + partes.resid)))

    def seleccionar(self, serie: pd.Series, max_p=2, max_q=2, umbral_estacional=0.64) -> "Arima":
        """
        Selecciona la especificación en dos pasos, como el procedimiento de
        Hyndman y Khandakar (2008):

        1. Los órdenes de integración se fijan con pruebas, no con el AIC:
           d con la prueba KPSS y D = 1 solo si la fuerza estacional de la
           serie diferenciada supera 0.64. El AIC no sirve para elegir d o D,
           porque al diferenciar cambian los datos sobre los que se calcula
           la verosimilitud y los valores dejan de ser comparables. Elegir d
           con el AIC puede llevar a un modelo sin diferencias sobre una serie
           con tendencia, cuyas proyecciones revierten hacia la media.
        2. Con d y D fijos, p, q, P y Q se eligen por menor AIC en una malla
           acotada. Se mantiene pequeña de forma deliberada: con poco más de
           trescientas observaciones, ampliarla produce modelos
           sobreparametrizados y multiplica el tiempo de cómputo.
        """
        serie = self._transformar(serie)
        d = self.diferencias_regulares(serie, max_d=1)
        base = serie.diff(d).dropna() if d else serie
        self.fuerza = self.fuerza_estacional(base, self.periodo)
        D = 1 if self.fuerza > umbral_estacional else 0
        self.d_D = (d, D)

        mejor_aic = np.inf
        mejor = ((1, d, 1), (0, D, 1, self.periodo))

        combinaciones = product(range(max_p + 1), range(max_q + 1), range(2), range(2))
        for p, q, P, Q in combinaciones:
            if p == q == P == Q == 0 and d == D == 0:
                continue
            try:
                ajuste = SARIMAX(
                    serie,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, self.periodo),
                    trend=self._tendencia(d, D),
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False)
                if np.isfinite(ajuste.aic) and ajuste.aic < mejor_aic:
                    mejor_aic = ajuste.aic
                    mejor = ((p, d, q), (P, D, Q, self.periodo))
            except Exception:
                continue

        self.orden, self.orden_estacional = mejor
        self.aic = mejor_aic
        return self

    def ajustar(self, serie: pd.Series, horizonte: int) -> "Arima":
        if self.orden is None:
            self.seleccionar(serie)
        self.horizonte = horizonte
        # Se imponen estacionariedad e invertibilidad. Sin estas restricciones
        # la optimización puede converger a raíces autorregresivas fuera del
        # círculo unitario en algunos orígenes de la validación, lo que produce
        # pronósticos que divergen y contaminan las métricas del modelo de
        # referencia. La restricción es además la condición teórica que el
        # apartado 3.1.2 exige para que las estimaciones sean válidas.
        self.resultado = SARIMAX(
            self._transformar(serie),
            order=self.orden,
            seasonal_order=self.orden_estacional,
            trend=self._tendencia(self.orden[1], self.orden_estacional[1]),
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
        return self

    def proyectar(self) -> float:
        valor = float(self.resultado.forecast(steps=self.horizonte).iloc[-1])
        return float(np.exp(valor)) if self.logaritmo else valor

    def trayectoria(self, pasos: int, nivel: float = 0.95) -> pd.DataFrame:
        """
        Proyección de 1 a `pasos` meses con su intervalo de confianza, en la
        escala original de la serie. La usa la aplicación para dibujar el
        abanico de incertidumbre.
        """
        pronostico = self.resultado.get_forecast(steps=pasos)
        tabla = pd.DataFrame({"proyectado": pronostico.predicted_mean})
        intervalo = pronostico.conf_int(alpha=1 - nivel)
        tabla["inferior"] = intervalo.iloc[:, 0].values
        tabla["superior"] = intervalo.iloc[:, 1].values
        return np.exp(tabla) if self.logaritmo else tabla

    def especificacion(self) -> str:
        p, d, q = self.orden
        P, D, Q, s = self.orden_estacional
        deriva = " con deriva" if self._tendencia(d, D) else ""
        return f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]{deriva}"


class ModeloArbol(Modelo):
    """
    Base de los modelos de aprendizaje automático.

    Construye internamente las variables predictoras a partir de la serie, de
    modo que desde fuera se comporta igual que el ARIMA: recibe una serie y
    devuelve una proyección.
    """

    def __init__(
        self,
        logaritmo: bool = False,
        variacion: bool = False,
        semilla: int = 42,
        **parametros,
    ):
        self.logaritmo = logaritmo
        # Ver caracteristicas.construir: con variacion=True el árbol aprende
        # el cambio respecto del último dato en lugar del nivel.
        self.variacion = variacion
        self.semilla = semilla
        self.parametros = parametros
        self.estimador = None
        self.serie = None
        self.columnas = None
        self.horizonte = 1

    def _crear_estimador(self):
        raise NotImplementedError

    def ajustar(self, serie: pd.Series, horizonte: int) -> "ModeloArbol":
        self.serie = serie
        self.horizonte = horizonte
        X, y = self.matriz_entrenamiento()
        self.columnas = list(X.columns)
        self.estimador = self._crear_estimador()
        self.estimador.fit(X, y)
        return self

    def nivel_de_partida(self, X: pd.DataFrame) -> np.ndarray:
        """
        Lo que hay que sumar a la salida del árbol para obtener el nivel de la
        serie, en la escala del modelo: el último dato conocido si se modela
        la variación, cero si se modela el nivel directamente.
        """
        if self.variacion:
            return X["rezago_1"].to_numpy(dtype=float)
        return np.zeros(len(X))

    def fila_prediccion(self) -> pd.DataFrame:
        """Variables del último mes, con las que se genera la proyección."""
        fila = caracteristicas.variables_para_prediccion(
            self.serie, horizonte=self.horizonte, logaritmo=self.logaritmo
        )
        return fila[self.columnas]

    def proyectar(self) -> float:
        fila = self.fila_prediccion()
        prediccion = float(self.estimador.predict(fila)[0]) + float(self.nivel_de_partida(fila)[0])
        # Si se entrenó en logaritmos, se devuelve el valor a la escala
        # original de la serie para que las métricas sean comparables.
        return float(np.exp(prediccion)) if self.logaritmo else prediccion

    def matriz_entrenamiento(self, horizonte: int | None = None):
        """Devuelve X e y del último ajuste, para el cálculo de valores SHAP."""
        return caracteristicas.construir(
            self.serie,
            horizonte=horizonte or self.horizonte,
            logaritmo=self.logaritmo,
            variacion=self.variacion,
        )


class BosqueAleatorio(ModeloArbol):
    """
    Bosques aleatorios (Breiman, 2001), apartado 3.2.2.1.

    La profundidad se limita y se exige un mínimo de observaciones por hoja
    porque la muestra es de pocos cientos de registros. Sin esas restricciones
    el bosque memoriza el ruido de la serie.
    """

    nombre = "Bosque aleatorio"

    def _crear_estimador(self):
        parametros = dict(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=3,
            max_features=0.7,
            random_state=self.semilla,
            n_jobs=-1,
        )
        parametros.update(self.parametros)
        return RandomForestRegressor(**parametros)


class Xgboost(ModeloArbol):
    """
    Potenciación del gradiente (Chen y Guestrin, 2016), apartado 3.2.2.2.

    Se emplean árboles poco profundos, tasa de aprendizaje baja y términos de
    regularización activos, que es la configuración indicada cuando el número
    de observaciones es reducido frente al número de predictores.
    """

    nombre = "XGBoost"

    def _crear_estimador(self):
        parametros = dict(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=self.semilla,
            n_jobs=-1,
            verbosity=0,
        )
        parametros.update(self.parametros)
        return XGBRegressor(**parametros)


CATALOGO = {
    "ingenuo": IngenuoEstacional,
    "arima": Arima,
    "bosque": BosqueAleatorio,
    "xgboost": Xgboost,
}
