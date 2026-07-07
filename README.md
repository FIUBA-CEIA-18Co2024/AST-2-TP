# Análisis de Series de Tiempo II — Entrega de Tareas

Repositorio de entrega de las tareas de la materia **Análisis de Series de Tiempo II** (Maestría en Inteligencia Artificial, FIUBA).

### Integrantes

- Joaquín González
- Pablo Gómez Verdini

## Tarea 1 — Feature Engineering y Validación Temporal con GBM

📓 [`Tareas/Tarea_1,_Feature_Engineering_y_Validación_Temporal_con_GBM.ipynb`](Tareas/Tarea_1,_Feature_Engineering_y_Validación_Temporal_con_GBM.ipynb)

Resolución sobre la serie mensual **AirPassengers** (1949–1960):

1. **Feature engineering:** features de calendario (mes, trimestre, año, día del año) con encoding cíclico sin/cos, lags [1, 2, 3, 12], ventanas móviles (rolling 3 y 12: media, desvío, máximo, mínimo) y ventanas expansivas, todas con `.shift(1)` para evitar leakage.
2. **Dos conjuntos de features:** solo calendario (`features_fecha`) vs. calendario + lags + rolling + expanding (`features_all`).
3. **Dos particiones train/test:** holdout temporal (últimos 24 meses como test) vs. holdout con shuffle (incorrecto a propósito, para evidenciar el leakage temporal).
4. **Modelado:** 4 modelos `GradientBoostingRegressor` (2 conjuntos de features × 2 holdouts), con curva de aprendizaje del boosting vía `staged_predict`.
5. **Métricas:** MAE, RMSE, MAPE y MASE (referencia: naive estacional de lag 12).
6. **Interpretabilidad:** importancia nativa del modelo vs. permutation importance sobre test, y comparación de ambos rankings.
7. **Análisis:** respuestas a las preguntas de la consigna con los resultados de la corrida.

## Reproducir la ejecución

```bash
python3 -m venv venv
venv/bin/pip install numpy pandas matplotlib scikit-learn statsmodels nbconvert ipykernel
venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  "Tareas/Tarea_1,_Feature_Engineering_y_Validación_Temporal_con_GBM.ipynb"
```

## Estructura del repositorio

```
├── Tarea1/             # Notebooks de entrega y planes de resolución
└── README.md
```
