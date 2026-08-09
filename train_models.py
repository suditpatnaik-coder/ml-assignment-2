from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from model.logistic_regression_model import build_model as build_logistic
from model.decision_tree_model import build_model as build_tree
from model.knn_model import build_model as build_knn
from model.naive_bayes_model import build_model as build_nb
from model.random_forest_model import build_model as build_rf
from utils import calculate_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
SEED = 314
TEST_SIZE = 0.25


def prepare_dataset():
    digits = load_digits()
    feature_names = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]
    X = pd.DataFrame(digits.data, columns=feature_names)
    y = pd.Series(digits.target.astype(int), name="target")

    full = X.copy()
    full["target"] = y
    full.to_csv(DATA_DIR / "digits_full_dataset.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    test_data = X_test.copy()
    test_data["target"] = y_test.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    return X_train, X_test, y_train, y_test, feature_names


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, feature_names = prepare_dataset()

    model_builders = {
        "Logistic Regression": ("logistic_regression.joblib", build_logistic),
        "Decision Tree": ("decision_tree.joblib", build_tree),
        "kNN": ("knn.joblib", build_knn),
        "Naive Bayes": ("naive_bayes.joblib", build_nb),
        "Random Forest": ("random_forest.joblib", build_rf),
    }

    all_metrics = {}
    for display_name, (filename, builder) in model_builders.items():
        model = builder()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)
        metrics = calculate_metrics(y_test, prediction, probability)
        all_metrics[display_name] = metrics
        joblib.dump(model, MODEL_DIR / filename)
        print(display_name, {k: round(v, 4) for k, v in metrics.items()})

    pd.DataFrame(all_metrics).T.to_csv(MODEL_DIR / "metrics.csv", index_label="Model")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    metadata = {
        "dataset_name": "Optical Recognition of Handwritten Digits (UCI; scikit-learn bundled subset)",
        "uci_dataset_id": 80,
        "source_doi": "10.24432/C50P49",
        "implementation_instances": int(len(X_train) + len(X_test)),
        "features": len(feature_names),
        "classes": list(range(10)),
        "feature_names": feature_names,
        "target_column": "target",
        "split_seed": SEED,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
