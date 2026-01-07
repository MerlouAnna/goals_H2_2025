from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ALLOWED_MODELS = {
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "svr_rbf",
    "knn",
}


def _make_model(name: str, random_state: int):
    """Factory to create models by name."""
    if name == "linear_regression":
        return (
            LinearRegression()
        )  # fits a straight-line relationship between features and target

    if name == "ridge":
        return Ridge(
            alpha=1.0, random_state=random_state
        )  # linear regression with L2 regularization to prevent overfitting -> alpha: Strength of regularization → higher alpha: smaller weights → less overfitting, but can underfit -> penalizes large weights -- helps when features are correlated

    if name == "lasso":
        return Lasso(
            alpha=0.001, max_iter=10_000
        )  # linear regression with L1 regularization tends to push some weights exactly to 0, so it can act like feature selection -> alpha: Strength of regularization → higher alpha: more coefficients set to zero →  feature selection

    if name == "elastic_net":
        return ElasticNet(
            alpha=0.001, l1_ratio=0.5, max_iter=10_000
        )  # combines L1 and L2 regularization -> l1_ratio: 0 = Ridge, 1 = Lasso, between 0 and 1: mix of both -> l1_ratio=0.5 (balanced mix of L1 and L2)

    if (
        name == "random_forest"
    ):  # builds many decision trees on bootstrapped samples and averages them -> reduces overfitting compared to single decision tree
        return RandomForestRegressor(
            n_estimators=300,  # number of trees in the forest
            random_state=random_state,
            n_jobs=1,  # number of parallel jobs to run -> None or 1 means no parallelism/ -1 means use all processors
        )

    if name == "extra_trees":
        return ExtraTreesRegressor(  # extremely randomized trees - similar to random forest but splits are chosen more randomly -> can reduce variance further
            n_estimators=500,  # number of trees in the forest
            random_state=random_state,
            n_jobs=1,
        )

    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            learning_rate=0.05,  # step size at each iteration while moving toward a minimum of the loss function -> smaller values require more trees but can improve performance
            n_estimators=200,  # number of boosting stages to perform
            random_state=random_state,
        )  # builds trees sequentially, each trying to correct errors of the previous one -> more prone to overfitting

    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(  # faster version of gradient boosting that uses histogram-based binning of continuous features
            learning_rate=0.05,  # step size at each iteration while moving toward a minimum of the loss function -> smaller values require more trees but can improve performance
            max_iter=200,  # maximum number of iterations
            random_state=random_state,
        )
    if name == "svr_rbf":
        # SVR is sensitive to feature scaling -> use a pipeline
        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),  # transforms each feature to roughly mean 0 and std 1 -distance- and margin-based models (KNN, SVR) break if one feature has bigger numeric scale than others
                (
                    "svr",
                    SVR(kernel="rbf", C=10.0, gamma="scale"),
                ),  # tries to fit a function with a margin of tolerance (epsilon-insensitive loss) -> C: regularization parameter -> high C: low bias, high variance (overfitting) - low C: high bias, low variance (underfitting) // gamma: kernel coefficient -> 'scale' is 1 / (n_features * X.var()) - higher gamma: closer points have more influence - can lead to overfitting
            ]
        )

    if name == "knn":
        # KNN is distance-based -> scaling matters
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsRegressor(n_neighbors=7),
                ),  # predicts by averaging the target of the k nearest points in feature space - “similarity” model
            ]
        )

    raise ValueError(f"Unknown model: {name}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true (np.ndarray): the true target values
        y_pred (np.ndarray): the predicted target values

    Returns:
        dict[str, float]: a dictionary with regression metrics

    The RMSE (Root Mean Squared Error - punishes big errors more) is computed as the square root of MSE, when MSE is Mean Squared Error as the difference between predicted and true values squared and averaged.
    The MAE (Mean Absolute Error) is the average of the absolute differences between predicted and true values.
    The R2 (R-squared score - variance score) is the proportion of the variance in the dependent variable that is predictable from the independent variables.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _train_one_model(
    job_id: str,
    model_name: str,
    model_dir: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """Train one model and return the results.
    Args:
        job_id (str): the job identifier
        model_name (str): the model name
        model_dir (str): the directory to save the model
        test_size (float): the test size fraction
        random_state (int): the random state for reproducibility
    Returns:
        dict[str, Any]: the training results

    The traibning results include:
        - model_name: the name of the trained model
        - model_path: the path to the saved model file
        - metrics: the regression metrics (dictionary with 'rmse', 'mae', 'r2')
        - y_test: the true target values for the test set
        - y_pred: the predicted target values for the test set
    """
    # load the diabetes dataset
    X, y = load_diabetes(return_X_y=True)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # create the model
    model: (
        LinearRegression
        | Ridge
        | Lasso
        | ElasticNet
        | RandomForestRegressor
        | ExtraTreesRegressor
        | GradientBoostingRegressor
        | HistGradientBoostingRegressor
        | Pipeline
    ) = _make_model(model_name, random_state=random_state)

    # fit the model - train on the training set
    model.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = model.predict(X_test)

    # compute metrics
    metrics = _compute_metrics(y_test, y_pred)

    # save the model
    model_path = Path(model_dir) / f"{job_id}__{model_name}.joblib"
    joblib.dump(model, model_path)

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "metrics": metrics,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }


class TrainConfig(BaseModel):
    """Configuration for training models in parallel.

    Args:
        model_names (list[str]): list of model names to train
        test_size (float, optional): fraction of data to use as test set. Defaults to 0.2.
        random_state (int, optional): random state for reproducibility. Defaults to 42.
    """

    model_names: list[str]
    test_size: float = 0.2
    random_state: int = 42


def train_models_in_parallel(job_id: str, cfg: TrainConfig) -> dict[str, Any]:
    """
    This is the main orchestrator function. The API calls the orchestrator to train multiple models, using multiprocessing.
    Train multiple models in parallel and return the best model and all results.
    Args:
        job_id (str): the job identifier
        cfg (TrainConfig): the training configuration
    Returns:
        dict[str, Any]: the training results including the best model and all results
    """
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    unknown = [m for m in cfg.model_names if m not in ALLOWED_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown models requested: {unknown}. Allowed: {sorted(ALLOWED_MODELS)}"
        )

    # determine the number of workers for parallel training - at most one per CPU core - but not more than the number of models to train
    # for example, if we have 4 CPU cores and 3 models to train -> min(3, 4) use 3 workers
    # if we have 4 CPU cores and 6 models to train -> min(6, 4) use 4 workers

    max_workers = min(len(cfg.model_names), os.cpu_count() or 1)

    results: list[dict[str, Any]] = []
    # use ProcessPoolExecutor to train models in parallel - each in its own process
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            # submit one process per model
            ex.submit(
                _train_one_model,
                job_id,
                model_name,
                str(model_dir),
                cfg.test_size,
                cfg.random_state,
            )
            for model_name in cfg.model_names
        ]
        # collect results as they complete - as_completed yields futures as they complete
        for fut in as_completed(futures):
            results.append(fut.result())

    # determine the best model based on R2 score (the higher the better)
    best = max(results, key=lambda r: r["metrics"]["r2"])

    return {
        "job_id": job_id,
        "best_model": best["model_name"],
        "best_metrics": best["metrics"],
        "all_results": [
            {
                "model_name": r["model_name"],
                "metrics": r["metrics"],
                "model_path": r["model_path"],
            }
            for r in results
        ],
        "best_y_test": best["y_test"],
        "best_y_pred": best["y_pred"],
        "best_model_path": best["model_path"],
    }
