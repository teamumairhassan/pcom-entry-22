"""Web service for the PCOM prediction competition (based on the course template).

GET  /health  -> {"health": "up"}
POST /predict -> body {"testdata_download_endpoint": ..., "prediction_upload_endpoint": ...}
                 downloads the input CSV, runs the model, uploads a one-row prediction CSV.
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from predictor import get_bundle, run_model, start_prediction  # noqa: F401  (run_model: used by tests)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_bundle()                      # load the model once at startup, so /predict is fast
    yield


app = FastAPI(title="ML Prediction API", lifespan=lifespan)


class EvaluationBody(BaseModel):
    testdata_download_endpoint: str
    prediction_upload_endpoint: str


@app.post("/predict", status_code=200)
async def prediction(payload: EvaluationBody, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_prediction, payload.testdata_download_endpoint,
                              payload.prediction_upload_endpoint)
    return {"status": "prediction started"}


@app.get("/health", status_code=200)
async def health():
    return {"health": "up"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
