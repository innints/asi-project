"""Model preprocessing i modele bazowe (Sprint 1).

Model preprocessing (imputacja, skalowanie, kodowanie — wszystko, co wymaga ``fit``) żyje
wewnątrz sklearn Pipeline. Dzięki temu przy predykcji wykonują się dokładnie te same kroki,
które zostały dopasowane na X_train, a cały model można zapisać jako jeden artefakt.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from asi_project.config import load_pipeline_parameters


def make_dummy() -> DummyClassifier:
    """Baseline bez wiedzy o cechach: zawsze przewiduje najczęstszą klasę z X_train."""
    return DummyClassifier(strategy="most_frequent")


def build_baseline_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    model_parameters: Mapping[str, int] | None = None,
) -> Pipeline:
    """Buduje preprocessing i klasyfikator z parametrami przekazanymi przez wywołującego."""
    if model_parameters is None:
        model_parameters = load_pipeline_parameters(Path(__file__).parent)["logistic_regression"]

    transformModules = []

    if numeric_features:
        numeric_transformModel = Pipeline(
            steps=[("fillerMean", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
        )
        transformModules.append(("numeric", numeric_transformModel, list(numeric_features)))

    if categorical_features:
        categorical_transformModel = Pipeline(
            steps=[
                ("fillerMostFreq", SimpleImputer(strategy="most_frequent")),
                (("ignore", OneHotEncoder(handle_unknown="ignore"))),
            ]
        )
        transformModules.append(
            ("categorical", categorical_transformModel, list(categorical_features))
        )

    preprocessor = ColumnTransformer(
        transformers=transformModules,
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**model_parameters)),
        ]
    )

    return pipeline
