import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import os

# Подключение к MLflow
mlflow.set_tracking_uri("")    # адрес учебного трекинг-сервера
os.environ["MLFLOW_TRACKING_USERNAME"] = ""    # логин для доступа к серверу
os.environ["MLFLOW_TRACKING_PASSWORD"] = ""   # пароль для доступа к серверу

os.environ["AWS_ACCESS_KEY_ID"] = ""   # пароль для доступа к серверу
os.environ["AWS_SECRET_ACCESS_KEY"] = ""   # пароль для доступа к серверу
os.environ["MLFLOW_S3_ENDPOINT_URL"] = ""
mlflow.set_experiment("Logistic_Regression_Experiment")

# 1. Подготовка данных
data = load_iris()
# Так как в параметрах указано "iris_binary", отфильтруем только 2 класса
X = data.data[data.target < 2]
y = data.target[data.target < 2]

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_params=[{"C": 1.0, "max_iter": 100, "solver": "lbfgs"},
            {"C": 0.1, "max_iter": 100, "solver": "lbfgs"},
            {"C": 10.0, "max_iter": 100, "solver": "lbfgs"}]

for params in lr_params:
    with mlflow.start_run(run_name=f"LogisticRegression_C_{params['C']}") as run:
        # Обучение модели
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Метрики
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)

        # Логирование параметров
        mlflow.log_params(params)

        # Логирование метрик
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)

        # input_example (пример входных данных для сигнатуры модели)
        input_example = X_train[:3]

        # Логирование самой модели
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            registered_model_name=f"LogisticRegression"
        )

    run_id = run.info.run_id

    print(f"Run completed! ID: {run_id}")    

print(f"Run completed! ID: {run_id}")