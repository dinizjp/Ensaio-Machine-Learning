# Ensaio-Machine-Learning

## Descrição
Ensaio de 16 algoritmos de Machine Learning com o intuido de adquirir conhecimento sobres o treinamento e ajuste fino de cada algoritmo, buscando uma melhor performance nas métricas de desempenho.

## Objetivo
O objetivo desse projeto foi realizar ensaios com algoritmos de Classificação, Regressão e Clusterização, para estudar a mudança do comportamento da performance, a medida que os valores dos principais parâmetros de controle de overfitting e underfitting mudam.

## Produto final
O produto final será 7 tabelas mostrando a performance dos algoritmos, avaliados usando múltiplas métricas, para 3 conjuntos de dados diferentes: Treinamento, validação e teste.

# Algoritmos usados no ensaio:

## **Algoritmos de Classificação**

- **Algoritmos:** K Nearest Neighbors (KNN), Decision Tree Classifier, Random Forest Classifier, Logistic Regression Classifier

- **Métricas de Desempenho:** Accuracy, Precision, Recall e F1-Score

## **Algoritmos de Regressão**

- **Algoritmos:** Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polinomial Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net, Polinomial Regression Lasso, Polinomial Regression Ridge e Polinomial Regression Elastic Net

- **Métricas de Desempenho:** MSE, RAMSE, MAE e MAPE

## **Algoritmos de Clusterização**

- **Algoritmos:** K-Means e Affinity Propagation

- **Métricas de Desempenho:** Silhouette Score

## Ferramentas usadas 

- Jupyter Lab 
- Python 3.10.9
- scikit-learn (sklearn): 1.2.1
- Plotly: 5.9.0
- Seaborn: 0.12.2
- NumPy: 1.23.5
- Matplotlib: 3.7.0
- Yellowbrick: 1.5
- Pandas: 1.5.3

# Resultados obtidos dos algoritmos de Classificação

## Resultados de Treino para Classificação 

|    | Metric    |   KNN |   Decision Tree |   Random Forest |   Logistic Regression |
|---:|:----------|------:|----------------:|----------------:|----------------------:|
|  0 | Accuracy  | 0.958 |           0.978 |           0.923 |                 0.875 |
|  1 | Precision | 0.974 |           0.984 |           0.924 |                 0.87  |
|  2 | Recall    | 0.928 |           0.964 |           0.895 |                 0.836 |
|  3 | F1-Score  | 0.951 |           0.974 |           0.909 |                 0.852 |

## Resultados de Validação para Classificação 

|    | Metric    |   KNN |   Decision Tree |   Random Forest |   Logistic Regression |
|---:|:----------|------:|----------------:|----------------:|----------------------:|
|  0 | Accuracy  | 0.925 |           0.951 |           0.923 |                 0.874 |
|  1 | Precision | 0.942 |           0.951 |           0.924 |                 0.869 |
|  2 | Recall    | 0.881 |           0.935 |           0.896 |                 0.835 |
|  3 | F1-Score  | 0.911 |           0.943 |           0.91  |                 0.852 |

## Resultados de Teste para Classificação 

|    | Metric    |   KNN |   Decision Tree |   Random Forest |   Logistic Regression |
|---:|:----------|------:|----------------:|----------------:|----------------------:|
|  0 | Accuracy  | 0.925 |           0.95  |           0.922 |                 0.872 |
|  1 | Precision | 0.942 |           0.952 |           0.925 |                 0.869 |
|  2 | Recall    | 0.883 |           0.933 |           0.895 |                 0.834 |
|  3 | F1-Score  | 0.911 |           0.942 |           0.91  |                 0.851 |

# Resultados obtidos dos algoritmos de Regressão 

## Resultados de Treino para Regressão 

|                                  |    R2 |     MSE |   RMSE |    MAE |   MAPE |
|:---------------------------------|------:|--------:|-------:|-------:|-------:|
| Linear Model                     | 0.045 | 456.561 | 21.367 | 17.015 |  8.628 |
| Linear Model Lasso               | 0.007 | 474.475 | 21.782 | 17.305 |  8.737 |
| Linear Model Ridge               | 0.045 | 456.561 | 21.367 | 17.015 |  8.628 |
| Linear Model ElasticNet          | 0.012 | 472.045 | 21.727 | 17.266 |  8.716 |
| Decision Tree Regressor          | 0.992 |   3.94  |  1.985 |  0.214 |  0.083 |
| Random Forest Regressor          | 0.901 |  47.417 |  6.886 |  4.931 |  2.621 |
| Polinomial Regression            | 0.087 | 436.635 | 20.896 | 16.568 |  8.361 |
| Polinomial Regression Lasso      | 0.014 | 471.28  | 21.709 | 17.23  |  8.649 |
| Polinomial Regression Ridge      | 0.086 | 437.12  | 20.907 | 16.58  |  8.379 |
| Polinomial Regression ElasticNet | 0.024 | 466.323 | 21.595 | 17.136 |  8.641 |

## Resultados de Validação para Regressão .

|                                  |     R2 |     MSE |   RMSE |    MAE |   MAPE |
|:---------------------------------|-------:|--------:|-------:|-------:|-------:|
| Linear Model                     |  0.04  | 458.197 | 21.406 | 17.041 |  8.663 |
| Linear Model Lasso               |  0.008 | 473.747 | 21.766 | 17.265 |  8.696 |
| Linear Model Ridge               |  0.04  | 458.196 | 21.406 | 17.041 |  8.663 |
| Linear Model ElasticNet          |  0.013 | 471.528 | 21.715 | 17.222 |  8.692 |
| Decision Tree Regressor          | -0.339 | 639.158 | 25.282 | 17.278 |  7.123 |
| Random Forest Regressor          |  0.331 | 319.346 | 17.87  | 13.021 |  7.062 |
| Polinomial Regression            |  0.065 | 446.644 | 21.134 | 16.791 |  8.54  |
| Polinomial Regression Lasso      |  0.014 | 470.756 | 21.697 | 17.181 |  8.656 |
| Polinomial Regression Ridge      |  0.067 | 445.744 | 21.113 | 16.78  |  8.555 |
| Polinomial Regression ElasticNet |  0.022 | 466.777 | 21.605 | 17.107 |  8.665 |

## Resultados de Teste para Regressão 

|                                  |     R2 |     MSE |   RMSE |    MAE |   MAPE |
|:---------------------------------|-------:|--------:|-------:|-------:|-------:|
| Linear Model                     |  0.048 | 463.691 | 21.533 | 17.178 |  8.528 |
| Linear Model Lasso               |  0.008 | 483.178 | 21.981 | 17.473 |  8.753 |
| Linear Model Ridge               |  0.048 | 463.689 | 21.533 | 17.178 |  8.529 |
| Linear Model ElasticNet          |  0.013 | 480.687 | 21.925 | 17.426 |  8.736 |
| Decision Tree Regressor          | -0.29  | 627.937 | 25.059 | 17.362 |  6.477 |
| Random Forest Regressor          |  0.344 | 319.335 | 17.87  | 13.133 |  6.615 |
| Polinomial Regression            |  0.08  | 448.027 | 21.167 | 16.847 |  8.333 |
| Polinomial Regression Lasso      | -0.004 | 488.715 | 22.107 | 17.445 |  8.756 |
| Polinomial Regression Ridge      |  0.079 | 448.471 | 21.177 | 16.852 |  8.339 |
| Polinomial Regression ElasticNet |  0.021 | 476.917 | 21.838 | 17.319 |  8.716 |




