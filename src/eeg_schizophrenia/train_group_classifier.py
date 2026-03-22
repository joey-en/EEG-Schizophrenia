# pip install pandas scikit-learn matplotlib joblib

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Force a non-interactive backend so the script can save plots on machines
# without a GUI session.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FEATURES_PATH = Path("data/processed/all_csv_agg")
DEFAULT_DEMOGRAPHICS_PATH = Path("data/raw/demographic.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/group_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model to predict demographic.csv[group] from aggregated EEG features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Feature CSV file or directory containing the aggregated part-*.csv files.",
    )
    parser.add_argument(
        "--demographics",
        type=Path,
        default=DEFAULT_DEMOGRAPHICS_PATH,
        help="Path to demographic.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where the model and evaluation files will be saved.",
    )
    parser.add_argument(
        "--target-column",
        default="group",
        help="Column in demographic.csv to predict.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of subjects kept for the final holdout set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible subject splits.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Maximum grouped cross-validation folds on the training subjects.",
    )
    parser.add_argument(
        "--exclude-sax",
        action="store_true",
        help="Ignore *_sax columns and train only on numeric/statistical EEG features.",
    )
    return parser.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files were found in {path}.")
        return pd.concat((pd.read_csv(csv_path) for csv_path in csv_paths), ignore_index=True)
    if not path.exists():
        raise FileNotFoundError(f"Feature path does not exist: {path}")
    return pd.read_csv(path)


def load_demographics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Demographics path does not exist: {path}")

    demographics = pd.read_csv(path)

    # The source file often contains extra spaces in the headers, e.g. " group ".
    # Strip them once here so the rest of the pipeline can always use plain names.
    demographics.columns = [str(column).strip() for column in demographics.columns]
    return demographics


def prepare_dataset(
    features_path: Path,
    demographics_path: Path,
    target_column: str,
    include_sax: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = load_features(features_path).copy()
    demographics = load_demographics(demographics_path).copy()

    required_feature_columns = {"subject", "condition"}
    missing_columns = required_feature_columns - set(features.columns)
    if missing_columns:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing_columns)}")
    if "subject" not in demographics.columns:
        raise ValueError("demographic.csv must contain a subject column.")
    if target_column not in demographics.columns:
        raise ValueError(
            f"demographic.csv does not contain '{target_column}'. "
            f"Available columns: {sorted(demographics.columns)}"
        )

    features["subject"] = pd.to_numeric(features["subject"], errors="coerce")
    features["condition"] = pd.to_numeric(features["condition"], errors="coerce")
    features = features.dropna(subset=["subject", "condition"])
    features["subject"] = features["subject"].astype(int)
    features["condition"] = features["condition"].astype(int)

    demographics["subject"] = pd.to_numeric(demographics["subject"], errors="coerce")
    demographics[target_column] = pd.to_numeric(demographics[target_column], errors="coerce")
    demographics = demographics.dropna(subset=["subject", target_column])
    demographics["subject"] = demographics["subject"].astype(int)
    demographics[target_column] = demographics[target_column].astype(int)

    merged = features.merge(
        demographics[["subject", target_column]].drop_duplicates(),
        on="subject",
        how="left",
        validate="many_to_one",
    )

    missing_subjects = sorted(merged.loc[merged[target_column].isna(), "subject"].unique().tolist())
    if missing_subjects:
        raise ValueError(
            "Some subjects in the EEG feature file do not have a label in demographic.csv: "
            f"{missing_subjects[:10]}"
        )

    sax_columns = [column for column in merged.columns if column.endswith("_sax")]
    if not include_sax:
        sax_columns = []

    excluded_columns = {target_column, "subject", "trial"}
    numeric_columns = [
        column
        for column in merged.columns
        if column not in excluded_columns and not column.endswith("_sax")
    ]

    # Decision: do not use subject as a feature. The label is defined per person,
    # so including subject would let the model memorize identities instead of EEG patterns.
    # Decision: drop trial as well because it is mainly an index, not a physiological signal.
    # Decision: keep condition because it describes the stimulus/task context for each row.
    merged[numeric_columns] = merged[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if sax_columns:
        merged[sax_columns] = merged[sax_columns].fillna("").astype(str)

    return merged, numeric_columns, sax_columns


def combine_sax_columns(frame: pd.DataFrame) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return frame.fillna("").astype(str).agg(" ".join, axis=1)


def build_pipeline(numeric_columns: list[str], sax_columns: list[str]) -> Pipeline:
    transformers = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_columns,
        )
    ]

    if sax_columns:
        transformers.append(
            (
                "sax",
                Pipeline(
                    steps=[
                        # Decision: combine all SAX strings into one text field per row and
                        # model them with character n-grams. That is a compact baseline for
                        # symbolic sequences without hand-crafting sequence features.
                        ("combine", FunctionTransformer(combine_sax_columns, validate=False)),
                        ("vectorizer", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3)),
                    ]
                ),
                sax_columns,
            )
        )

    # Decision: start with logistic regression. It is a strong baseline for
    # wide, sparse feature spaces and gives stable training on mixed numeric/text features.
    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=4000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_grouped_cv(train_frame: pd.DataFrame, target_column: str, requested_splits: int, random_state: int):
    subject_labels = train_frame[["subject", target_column]].drop_duplicates()
    min_class_subjects = int(subject_labels[target_column].value_counts().min())
    n_splits = min(requested_splits, min_class_subjects)
    if n_splits < 2:
        raise ValueError("Need at least two training subjects per class for grouped cross-validation.")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )
    axis.set_title("Group Prediction Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    dataset, numeric_columns, sax_columns = prepare_dataset(
        features_path=args.features,
        demographics_path=args.demographics,
        target_column=args.target_column,
        include_sax=not args.exclude_sax,
    )

    subject_labels = dataset[["subject", args.target_column]].drop_duplicates().sort_values("subject")

    # Decision: split by subject, not by row. Each subject contributes many rows,
    # and a normal random row split would leak the same person's signal into both train and test.
    train_subjects, test_subjects = train_test_split(
        subject_labels["subject"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=subject_labels[args.target_column],
    )

    train_subjects = set(train_subjects.tolist())
    test_subjects = set(test_subjects.tolist())
    train_frame = dataset[dataset["subject"].isin(train_subjects)].copy()
    test_frame = dataset[dataset["subject"].isin(test_subjects)].copy()

    feature_columns = [*numeric_columns, *sax_columns]
    pipeline = build_pipeline(numeric_columns=numeric_columns, sax_columns=sax_columns)
    grouped_cv = build_grouped_cv(train_frame, args.target_column, args.cv_splits, args.random_state)

    cv_scores = cross_validate(
        pipeline,
        train_frame[feature_columns],
        train_frame[args.target_column],
        groups=train_frame["subject"],
        cv=grouped_cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
    )

    pipeline.fit(train_frame[feature_columns], train_frame[args.target_column])
    predictions = pipeline.predict(test_frame[feature_columns])
    probabilities = pipeline.predict_proba(test_frame[feature_columns])[:, 1]

    metrics = {
        "accuracy": accuracy_score(test_frame[args.target_column], predictions),
        "balanced_accuracy": balanced_accuracy_score(test_frame[args.target_column], predictions),
        "f1": f1_score(test_frame[args.target_column], predictions),
        "roc_auc": roc_auc_score(test_frame[args.target_column], probabilities),
    }
    cv_metrics = {
        metric_name.replace("test_", ""): float(values.mean())
        for metric_name, values in cv_scores.items()
        if metric_name.startswith("test_")
    }
    report = classification_report(test_frame[args.target_column], predictions, digits=4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "group_classifier.joblib"
    metrics_path = args.output_dir / "metrics.json"
    confusion_matrix_path = args.output_dir / "confusion_matrix.png"

    joblib.dump(pipeline, model_path)
    save_confusion_matrix(test_frame[args.target_column], predictions, confusion_matrix_path)

    metrics_payload = {
        "config": {
            "features": str(args.features),
            "demographics": str(args.demographics),
            "target_column": args.target_column,
            "include_sax": not args.exclude_sax,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "cv_splits": grouped_cv.n_splits,
        },
        "data_summary": {
            "rows": int(dataset.shape[0]),
            "subjects": int(dataset["subject"].nunique()),
            "numeric_feature_count": len(numeric_columns),
            "sax_feature_count": len(sax_columns),
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
        },
        "holdout_metrics": metrics,
        "cross_validation_metrics": cv_metrics,
        "classification_report": report,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Rows used: {dataset.shape[0]}")
    print(f"Subjects used: {dataset['subject'].nunique()}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"SAX features: {len(sax_columns)}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print("Holdout metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Grouped CV metrics:")
    for name, value in cv_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Classification report:")
    print(report)
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
