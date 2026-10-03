"""Model construction."""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor
import yaml 
from pathlib import Path

_MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "dummy": DummyClassifier
}

PIPELINE_DIR = Path("C:/Users/joaoa/Desktop/JP/Universidade/Mestrado/Ano 1/S1/ETAI/ETAI-Pipeline")

with open(PIPELINE_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)



def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def make_pipeline(model_config, preprocessing_config=None):
    # Same as main.py: Pipeline([("prep", build_preprocessor(...)), ("model", build_model(...))])
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config or config["preprocessing"])),
        ("model", build_model(model_config)),
    ])