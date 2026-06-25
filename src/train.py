import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.
    """

    # 1. Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tách features và target
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 3. Bắt đầu MLflow run
    with mlflow.start_run():
        # 4. Log params
        mlflow.log_params(params)

        # 5. Train model
        model = RandomForestClassifier(
            **params,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 6. Evaluate
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # 7. Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 8. Log model vào MLflow
        mlflow.sklearn.log_model(model, "model")

        # 9. In kết quả
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 10. Lưu metrics cho CI/CD
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1
                },
                f,
                indent=2
            )

        # 11. Lưu model local
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # 12. Trả về accuracy
    return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)