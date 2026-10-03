"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> split features/target
    -> leak-safe train/test split -> preprocess + train (week 3's encoder/scaler pair)
    -> evaluate (accuracy, fairness) -> save results
"""
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target, build_preprocessor, split_dev_test, train_test_split
from src.model import make_pipeline
from src.evaluate import evaluate, fairness_report, cross_validate_pipeline, cv_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    df_raw=load_data(config["data"]["path"])
    cv_config=config["cv"]
    shuffle = cv_config.get("shuffle", True)
    pipeline = make_pipeline(config["model"])
    scoring = cv_config.get("scoring", "accuracy")
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                     random_state=cv_config.get("random_state") if shuffle else None)
    X, y, extras = split_features_target(df_raw, config["data"], config["preprocessing"]["mnar_indicator_sources"])

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"], random_state=config["test_set"]["random_state"]
    )

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_config.get("n_jobs", 1)
    )

    final_model = make_pipeline(config["model"]).fit(X_dev, y_dev)
    print(f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows.")
    y_train_pred = final_model.predict(X_dev)
    y_test_pred = final_model.predict(X_test)

    report = cv_report(fold_scores, scoring)
    report += "\n" + fairness_report(
    y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")

'''

def main():
    config = load_config()

    # load + diagnose-and-clean (week 3): domain-rule/placeholder -> NaN, category
    # cleanup, de-duplication, redundant-column removal -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # leak-safe split: everything above this line is target/split-independent and may
    # see the whole dataset; everything below (imputation, encoding, scaling) is fit
    # only on the training fold, inside the Pipeline below
    X_train, X_test, y_train, y_test, extras_train, extras_test = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    preprocessor = build_preprocessor(config["preprocessing"])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", build_model(config["model"])),
    ])
    pipeline.fit(X_train, y_train)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
    '''