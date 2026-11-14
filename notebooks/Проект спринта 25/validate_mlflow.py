import mlflow
from mlflow.tracking import MlflowClient
import sys
import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
]

ALLOWED_METRIC_NAMES = {"accuracy", "precision", "recall", "f1", "roc_auc", "f1_score", "roc-auc"}
MIN_METRICS_COUNT = 3


def check_env_vars() -> list:
    """Проверяет наличие всех необходимых переменных окружения."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        return [f"Отсутствуют переменные окружения: {', '.join(missing)}"]
    return []


def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = "") -> list:
    """Рекурсивно получает все артефакты запуска."""
    artifacts = []
    try:
        items = client.list_artifacts(run_id, path)
    except Exception as e:
        print(f"  Предупреждение: не удалось получить артефакты по пути '{path}': {e}")
        return artifacts

    for item in items:
        if item.is_dir:
            artifacts.extend(list_artifacts_recursive(client, run_id, item.path))
        else:
            artifacts.append(item)

    return artifacts


def check_metrics(metrics: dict, run_name: str, run_id: str) -> list:
    """
    Проверяет метрики запуска:
    - минимум MIN_METRICS_COUNT метрик
    - все метрики должны быть из разрешённого набора
    """
    errors = []

    if len(metrics) < MIN_METRICS_COUNT:
        errors.append(
            f"Запуск '{run_name}' (ID: {run_id}): найдено только {len(metrics)} метрик. "
            f"Требуется ≥{MIN_METRICS_COUNT}."
        )

    unknown_metrics = [
        name for name in metrics
        if name.lower() not in ALLOWED_METRIC_NAMES
    ]
    if unknown_metrics:
        errors.append(
            f"Запуск '{run_name}' (ID: {run_id}): найдены неизвестные метрики: "
            f"{', '.join(unknown_metrics)}. "
            f"Допустимые: {', '.join(sorted(ALLOWED_METRIC_NAMES))}."
        )

    return errors


def check_hyperparams_unique(runs: list) -> list:
    """
    Проверяет, что гиперпараметры отличаются хотя бы в одном параметре
    между разными запусками.
    """
    errors = []
    if len(runs) < 2:
        return errors

    param_sets = []
    for run in runs:
        param_sets.append(frozenset(run.data.params.items()))

    # Ищем дубликаты
    duplicates = []
    for i in range(len(param_sets)):
        for j in range(i + 1, len(param_sets)):
            if param_sets[i] == param_sets[j]:
                name_i = runs[i].info.run_name or f"run_{i + 1}"
                name_j = runs[j].info.run_name or f"run_{j + 1}"
                duplicates.append(f"'{name_i}' и '{name_j}'")

    if duplicates:
        errors.append(
            f"Следующие запуски имеют идентичные гиперпараметры: "
            f"{'; '.join(duplicates)}. Гиперпараметры должны отличаться между запусками."
        )

    return errors


def check_mlflow_requirements(experiment_name: str = "fraud_detection_classification") -> list:
    """
    Проверяет наличие в MLflow:
    - Минимум 3 запусков.
    - В каждом запуске: ≥3 параметров, ≥MIN_METRICS_COUNT метрик из допустимого набора,
      модель, графики (confusion_matrix, learning_curve, SHAP).
    - Гиперпараметры различаются между запусками.
    - Хотя бы одна зарегистрированная модель.
    Выводит ошибки для невыполненных требований.
    """
    errors = []

    # Проверяем переменные окружения
    env_errors = check_env_vars()
    if env_errors:
        errors.extend(env_errors)
        return errors

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])

    # Проверяем наличие эксперимента
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            errors.append(f"Эксперимент '{experiment_name}' не найден.")
            return errors
        experiment_id = experiment.experiment_id
    except Exception as e:
        errors.append(f"Ошибка при получении эксперимента: {str(e)}")
        return errors

    # Проверяем количество запусков
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            max_results=1000,
        )
    except Exception as e:
        errors.append(f"Ошибка при получении запусков: {str(e)}")
        return errors

    if len(runs) < 3:
        errors.append(f"Найдено только {len(runs)} запусков. Требуется минимум 3.")
        return errors

    # Проверяем уникальность гиперпараметров между запусками
    errors.extend(check_hyperparams_unique(runs))

    # Проверяем каждый запуск
    for i, run in enumerate(runs, 1):
        run_id = run.info.run_id
        run_name = run.info.run_name or f"run_{i}"

        # Проверяем параметры
        params = run.data.params
        if len(params) < 3:
            errors.append(
                f"Запуск '{run_name}' (ID: {run_id}): найдено только {len(params)} параметров. "
                f"Требуется ≥3."
            )

        # Проверяем метрики (количество + допустимые названия)
        errors.extend(check_metrics(run.data.metrics, run_name, run_id))

        # Рекурсивно получаем все артефакты
        all_artifacts = list_artifacts_recursive(client, run_id)
        artifact_paths = [a.path.lower() for a in all_artifacts]

        # Проверяем наличие модели
        model_found = any(
            p.endswith(".pkl") or "model" in p
            for p in artifact_paths
        )
        if not model_found:
            errors.append(
                f"Запуск '{run_name}' (ID: {run_id}): модель не найдена "
                f"(ожидался артефакт .pkl или логированная модель)."
            )

        # Проверяем наличие графиков
        required_plots = ["confusion_matrix", "learning_curve", "shap"]
        missing_plots = [
            plot for plot in required_plots
            if not any(plot in p for p in artifact_paths)
        ]
        if missing_plots:
            errors.append(
                f"Запуск '{run_name}' (ID: {run_id}): отсутствуют графики: "
                f"{', '.join(missing_plots)}."
            )

    # Проверяем наличие зарегистрированной модели
    try:
        models = client.search_model_versions("name='FraudDetectionModel'")
        if not models:
            errors.append(
                "Не найдено ни одной версии модели 'FraudDetectionModel' в Model Registry."
            )
    except Exception as e:
        errors.append(f"Ошибка при проверке Model Registry: {str(e)}")

    # Выводим результаты
    if not errors:
        print("Все требования выполнены!")
    else:
        print("Найдены следующие ошибки:")
        for error in errors:
            print(f"  - {error}")

    return errors


if __name__ == "__main__":
    # Получаем имя эксперимента из аргументов (по умолчанию: "fraud_detection_classification")
    experiment_name = "fraud_detection_classification"
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    errors = check_mlflow_requirements(experiment_name)

    sys.exit(1 if errors else 0)
