# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
##
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
import os
import gzip
import json
import pickle
from pathlib import Path
from glob import glob

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (precision_score, balanced_accuracy_score, 
                             recall_score, f1_score, confusion_matrix)

# Paso 1: Cargar y limpiar datos
def cargar_y_preparar(ruta_zip):
    df = pd.read_csv(ruta_zip, compression="zip").copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)].copy()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v >= 4 else v)
    df = df.dropna()
    return df

df_train = cargar_y_preparar("files/input/train_data.csv.zip")
df_test = cargar_y_preparar("files/input/test_data.csv.zip")

# Paso 2: Dividir datasets
x_train = df_train.drop(columns=["default"])
y_train = df_train["default"]
x_test = df_test.drop(columns=["default"])
y_test = df_test["default"]

# Paso 3: Pipeline
cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
num_features = [col for col in x_train.columns if col not in cat_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), cat_features),
        ("num", StandardScaler(), num_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("selector", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("mlp", MLPClassifier(max_iter=15000)),
    ]
)

# Paso 4: Optimización de hiperparámetros
# Búsqueda muy fina alrededor de alpha=0.27 y learning_rate=0.00095
param_grid = {
    "selector__k": [21],
    "pca__n_components": [None],
    "mlp__hidden_layer_sizes": [
        (50, 30, 40, 60),
        (52, 32, 42, 62),
        (48, 28, 38, 58),
    ],
    "mlp__alpha": [0.25, 0.26, 0.265, 0.27, 0.275, 0.28, 0.29],
    "mlp__learning_rate_init": [0.0009, 0.00092, 0.00095, 0.00098, 0.001, 0.00102],
    "mlp__random_state": [21],
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)

model.fit(x_train, y_train)

# Paso 5: Guardar modelo
modelos_dir = Path("files/models")
if modelos_dir.exists():
    for fichero in glob(str(modelos_dir / "*")):
        os.remove(fichero)
    try:
        os.rmdir(modelos_dir)
    except OSError:
        pass
modelos_dir.mkdir(parents=True, exist_ok=True)

with gzip.open(modelos_dir / "model.pkl.gz", "wb") as f:
    pickle.dump(model, f)

# Paso 6 y 7: Calcular y guardar métricas
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Métricas
metrics = [
    {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred, zero_division=0),
        "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
    },
    {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
    },
]

# Matrices de confusión
for dataset_name, y_true, y_pred in [("train", y_train, y_train_pred), 
                                      ("test", y_test, y_test_pred)]:
    cm = confusion_matrix(y_true, y_pred)
    metrics.append({
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    })

# Guardar métricas
out_dir = Path("files/output")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
    for metric in metrics:
        f.write(json.dumps(metric) + "\n")

print(f"Best score: {model.best_score_:.4f}")
print(f"Best params: {model.best_params_}")
print(f"Train score: {model.score(x_train, y_train):.4f}")
print(f"Test score: {model.score(x_test, y_test):.4f}")