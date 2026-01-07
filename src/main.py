from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.plots import plot_pred_vs_true_png, plot_residuals_png
from src.training import ALLOWED_MODELS, TrainConfig, train_models_in_parallel

# instantiate the API server
app = FastAPI(title="Diabetes ML Service (Docker + sklearn + matplotlib + concurrency)")

# pool of 2 threads - can run 2 training jobs concurrently - others queued until a thread is free
# actual CPU parallelism happens inside training via ProcessPoolExecutor - this is just to manage multiple requests to keep FastAPI responsive
TRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=2)

JOBS_LOCK = Lock()  # to protect access to JOBS dict
JOBS: dict[
    str, dict[str, Any]
] = {}  # in memory dict to hold job info - in real app use persistent storage


class TrainRequest(BaseModel):
    """Training request payload.

    Args:
        model_names: List of model names to train.
        test_size: Fraction of data to use as test set.
        random_state: Random seed for reproducibility.
    """

    model_names: list[
        Literal[
            "elastic_net",
            "extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "knn",
            "lasso",
            "linear_regression",
            "random_forest",
            "ridge",
            "svr_rbf",
        ]
    ] = Field(default_factory=lambda: ["linear_regression", "ridge", "random_forest"])
    test_size: float = 0.2
    random_state: int = 42


@app.get("/")
def root():
    """Quick sanity check endpoint."""
    return {
        "message": "OK. Use POST /train to start a job.",
        "allowed_models": sorted(list(ALLOWED_MODELS)),
    }


def _set_job(job_id: str, patch: dict[str, Any]) -> None:
    """Helper function to update job info atomically.

    Args:
        job_id: ID of the job to update.
        patch: Dict of fields to update.
    """
    with JOBS_LOCK:
        JOBS[job_id].update(patch)


def _run_training_job(job_id: str, req: TrainRequest) -> None:
    """Helper function to run the training job and update job status.

    Args:
        job_id (str): ID of the job to run.
        req (TrainRequest): Training request payload.
    """
    _set_job(
        job_id,
        {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()},
    )
    try:
        cfg = TrainConfig(
            model_names=list(req.model_names),
            test_size=req.test_size,
            random_state=req.random_state,
        )
        results = train_models_in_parallel(job_id=job_id, cfg=cfg)
        _set_job(
            job_id,
            {
                "status": "done",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
        )
    except Exception as e:
        _set_job(
            job_id,
            {
                "status": "error",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            },
        )


@app.post("/train")
def train(req: TrainRequest):
    """Start a new training job.
    Args:
        req (TrainRequest): Training request payload.
    Returns:
        dict: Job ID and initial status.
    """
    job_id = str(uuid4())

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "results": None,
            "error": None,
        }

    TRAIN_EXECUTOR.submit(_run_training_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    """Get the status of a training job.

    Args:
        job_id (str): ID of the job to check.

    Raises:
        HTTPException: If job not found.

    Returns:
        dict: Job status information.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        k: job.get(k)
        for k in [
            "job_id",
            "status",
            "created_at",
            "started_at",
            "finished_at",
            "error",
        ]
    }


@app.get("/results/{job_id}")
def results(job_id: str):
    """Get the results of a completed training job.
    Args:
        job_id (str): ID of the job to get results for.
    Raises:
        HTTPException: If job not found or not done.
    Returns:
        dict: Training results.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(
            status_code=400, detail=f"Job not done. Current status: {job['status']}"
        )
    return job["results"]


@app.get("/plot/{job_id}/pred-vs-true")
def plot_pred_vs_true(job_id: str):
    """Get the plot of predicted vs true values for a completed training job.
    Args:
        job_id (str): ID of the job to get plot for.
    Raises:
        HTTPException: If job not found or not done.
    Returns:
        Response: PNG image response.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not done or not found")

    r = job["results"]
    png = plot_pred_vs_true_png(r["best_y_test"], r["best_y_pred"])
    return Response(content=png, media_type="image/png")


@app.get("/plot/{job_id}/residuals")
def plot_residuals(job_id: str):
    """Get the plot of residuals for a completed training job.
    Args:
        job_id (str): ID of the job to get plot for.
    Raises:
        HTTPException: If job not found or not done.
    Returns:
        Response: PNG image response.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not done or not found")

    r = job["results"]
    png = plot_residuals_png(r["best_y_test"], r["best_y_pred"])
    return Response(content=png, media_type="image/png")


@app.get("/jobs")
def list_jobs():
    """List all training jobs with lightweight info.
    Returns:
        dict: List of jobs with basic info.
    """
    with JOBS_LOCK:
        # return a lightweight view (no big arrays)
        items = []
        for job_id, job in JOBS.items():
            items.append(
                {
                    "job_id": job_id,
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                    "error": job.get("error"),
                    "best_model": (job.get("results") or {}).get("best_model"),
                    "best_metrics": (job.get("results") or {}).get("best_metrics"),
                }
            )
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"count": len(items), "jobs": items}
