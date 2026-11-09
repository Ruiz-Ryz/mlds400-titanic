"""Train a Titanic logistic regression model and write test-set predictions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL


def read_inputs(directory):
    frames = []
    for filename in ["train.csv", "test.csv"]:
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"Missing {path}. Download the Kaggle CSVs as described in README.md.")
        frame = pd.read_csv(path)
        required = ["PassengerId"] + FEATURES
        if filename == "train.csv":
            required.append("Survived")
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")
        if frame.empty or frame["PassengerId"].isna().any() or frame["PassengerId"].duplicated().any():
            raise ValueError(f"{filename} must contain rows with unique, nonmissing PassengerId values.")
        frames.append(frame)
    train, test = frames
    if train["Survived"].isna().any() or set(train["Survived"].unique()) != {0, 1}:
        raise ValueError("train.csv must contain both Survived classes, coded 0 and 1.")
    return train, test


def features(frame):
    result = frame[FEATURES].copy()
    for name in NUMERIC:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    for name in CATEGORICAL:
        result[name] = result[name].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        result[name] = result[name].replace("", np.nan)
    return result


def new_model():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    return Pipeline([
        ("prepare", ColumnTransformer([
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ])),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def show_preprocessing(model, label):
    prepare = model.named_steps["prepare"]
    medians = prepare.named_transformers_["numeric"].named_steps["imputer"].statistics_
    modes = prepare.named_transformers_["categorical"].named_steps["imputer"].statistics_
    print(f"[Q14] {label} numeric fill values: {dict(zip(NUMERIC, np.round(medians, 4).tolist()))}")
    print(f"[Q14] {label} categorical fill values: {dict(zip(CATEGORICAL, modes))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("src/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/python"))
    args = parser.parse_args()

    print(f"[Q13] Python {sys.version.split()[0]}; pandas {pd.__version__}; scikit-learn {sklearn.__version__}")
    train, test = read_inputs(args.data_dir)
    print(f"[Q13, Q17] Loaded {len(train)} training passengers and {len(test)} test passengers.")
    print(f"[Q14] Overall training survival rate: {train['Survived'].mean():.2%}")
    print("[Q14] Training survival rates by sex:")
    print(train.groupby("Sex")["Survived"].agg(["count", "mean"]).to_string())
    print("[Q14] Training survival rates by passenger class:")
    print(train.groupby("Pclass")["Survived"].agg(["count", "mean"]).to_string())

    x, x_test = features(train), features(test)
    y = train["Survived"].astype(int)
    print(f"[Q14] Features retained: {FEATURES}")
    print("[Q14] PassengerId, Name, Ticket, and Cabin are excluded from the model.")
    print(f"[Q14] Training missing counts: {x.isna().sum().to_dict()}")
    print(f"[Q14] Test missing counts: {x_test.isna().sum().to_dict()}")
    print("[Q14] Numeric values: median imputation and standardization. Categories: mode imputation and one-hot encoding.")
    print("[Q14] Rows are retained. Imputation/scaling are fitted only on each model's training portion.")

    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42,
    )
    validation_model = new_model()
    validation_model.fit(x_fit, y_fit)
    show_preprocessing(validation_model, "Validation-fit")
    validation_accuracy = accuracy_score(y_valid, validation_model.predict(x_valid))
    print(f"[Q15] Logistic regression; stratified split: {len(x_fit)} fit rows / {len(x_valid)} validation rows; seed 42.")
    print(f"[Q18] Holdout VALIDATION accuracy: {validation_accuracy:.4f} ({validation_accuracy:.2%}).")

    final_model = new_model()
    final_model.fit(x, y)
    show_preprocessing(final_model, "Full-training-fit")
    train_accuracy = accuracy_score(y, final_model.predict(x))
    print(f"[Q16] Full-training accuracy (resubstitution): {train_accuracy:.4f} ({train_accuracy:.2%}).")
    predictions = final_model.predict(x_test).astype(int)
    result = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
    print(f"[Q17] Generated {len(result)} test predictions; threshold 0.5.")
    print(result.head(5).to_string(index=False))

    test_accuracy = None
    if "Survived" in test.columns:
        if test["Survived"].isna().any() or not test["Survived"].isin([0, 1]).all():
            raise ValueError("Provided test Survived labels must be complete and coded 0 or 1.")
        test_accuracy = float(accuracy_score(test["Survived"], predictions))
        print(f"[Q18] Test accuracy using supplied labels: {test_accuracy:.4f}.")
    else:
        print("[Q18] Kaggle test accuracy: N/A. test.csv has no true Survived labels.")
        print("[Q18] The holdout score above is validation accuracy, not Kaggle test accuracy.")
        print("[Q18] gender_submission.csv is an example prediction baseline, not ground truth.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics = {
        "language": "Python", "model": "logistic_regression", "seed": 42,
        "training_rows": len(train), "validation_fit_rows": len(x_fit),
        "validation_rows": len(x_valid), "test_rows": len(test),
        "training_accuracy": float(train_accuracy),
        "validation_accuracy": float(validation_accuracy),
        "test_accuracy": test_accuracy,
        "test_accuracy_note": (
            "Unavailable for the original Kaggle test.csv because it has no true labels."
            if test_accuracy is None else "Evaluated using the supplied test Survived labels."
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[Q19] Saved {args.output_dir / 'predictions.csv'} and {args.output_dir / 'metrics.json'}.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
