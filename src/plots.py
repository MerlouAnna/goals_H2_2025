from __future__ import annotations

from io import BytesIO
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # important for Docker/headless

import matplotlib.pyplot as plt
import numpy as np


def _to_np(x: Iterable[float]) -> np.ndarray:
    return np.array(list(x), dtype=float)


def plot_pred_vs_true_png(y_true, y_pred) -> bytes:
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(y_true, y_pred)
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_v, max_v], [min_v, max_v])

    ax.set_title("Predicted vs True (Diabetes)")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def plot_residuals_png(y_true, y_pred) -> bytes:
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)
    residuals = y_true - y_pred

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(residuals, bins=30)
    ax.set_title("Residuals Histogram (True - Predicted)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
