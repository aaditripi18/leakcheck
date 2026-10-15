"""The four E2 baselines and their input pipelines.

Fixed before any E2 result was seen. Standard, untuned settings: the question
is how much a *typical* baseline's score depends on leaked rows, not how high a
tuned model can go.

  lr   signed-log + median impute + standardise numeric; one-hot categorical
       (top 50 levels); logistic regression
  rf   random forest, 100 trees; ordinal categorical; NaN handled natively
  hgb  histogram gradient boosting, native categorical support
  mlp  same inputs as lr; MLP (128, 64), Adam, early stopping

A free-text column (SMS, URLs) becomes TF-IDF word 1-2 grams (top 20,000) for
every model; HistGradientBoosting needs dense input, so it gets the top 2,000.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

MODELS = ("lr", "rf", "hgb", "mlp")
MAX_CATEGORIES = 50
TFIDF_FEATURES = 20_000
TFIDF_FEATURES_DENSE = 2_000
MISSING_SENTINEL = -1e9  # rf + text only, see preprocessor()


def _slog(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    return np.sign(x) * np.log1p(np.abs(x))


def _finite(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, np.nan)


def _as_text(x):
    return pd.Series(np.asarray(x).ravel()).fillna("").astype(str).to_numpy()


def column_kinds(X: pd.DataFrame, text: str | None) -> tuple[list[str], list[str]]:
    num = [c for c in X.columns if c != text and pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c != text and c not in num]
    return num, cat


def preprocessor(model: str, num: list[str], cat: list[str], text: str | None) -> ColumnTransformer:
    parts = []
    if model in ("lr", "mlp"):
        if num:
            parts.append(("num", Pipeline([
                ("slog", FunctionTransformer(_slog)),
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), num))
        if cat:
            parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=MAX_CATEGORIES,
                                               sparse_output=True), cat))
    else:
        # A text column makes the matrix sparse, and RandomForest takes NaN only in
        # dense input. In that case missing values become a sentinel far below any
        # real value, which a tree can still split off. (Found in week 6 on
        # PhiUSIIL; SMS Spam, the only earlier text dataset, has no other columns.)
        sentinel = model == "rf" and text is not None
        if num:
            num_steps = [("finite", FunctionTransformer(_finite))]
            if sentinel:
                num_steps.append(("missing", SimpleImputer(strategy="constant", fill_value=MISSING_SENTINEL,
                                                           keep_empty_features=True)))
            parts.append(("num", Pipeline(num_steps), num))
        if cat:
            fill = -1 if sentinel else np.nan  # category codes start at 0; must be an int here
            parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=fill,
                                                encoded_missing_value=fill, max_categories=250), cat))
    if text:
        k = TFIDF_FEATURES_DENSE if model == "hgb" else TFIDF_FEATURES
        parts.append(("text", Pipeline([("str", FunctionTransformer(_as_text)),
                                        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=k, sublinear_tf=True))]),
                      [text]))
    sparse = 0.0 if model == "hgb" else 0.3
    return ColumnTransformer(parts, sparse_threshold=sparse)


def estimator(model: str, seed: int, n_num: int, n_cat: int):
    if model == "lr":
        return LogisticRegression(max_iter=2000)
    if model == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if model == "hgb":
        # the preprocessor puts numeric columns first, then categorical, then text
        cats = list(range(n_num, n_num + n_cat)) or None
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed, categorical_features=cats)
    if model == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=50, batch_size=512,
                             random_state=seed)
    raise ValueError(model)


def build(model: str, X: pd.DataFrame, text: str | None, seed: int) -> Pipeline:
    num, cat = column_kinds(X, text)
    return Pipeline([("pre", preprocessor(model, num, cat, text)), ("model", estimator(model, seed, len(num), len(cat)))])
