import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .rainfall_api import router as rainfall_router
    from .prediction_api import router as prediction_router
    from .temperature_api import router as temperature_router
    from .live_api import router as live_router
    from .scenario_api import router as scenario_router
except ImportError:
    from rainfall_api import router as rainfall_router
    from prediction_api import router as prediction_router
    from temperature_api import router as temperature_router
    from live_api import router as live_router
    from scenario_api import router as scenario_router

app = FastAPI(
    title="India Climate Twin API",
    description="Climate data, AI predictions, live telemetry, and What-If simulation services",
    version="0.2.0",
)

app.include_router(rainfall_router)
app.include_router(temperature_router)
app.include_router(prediction_router)
app.include_router(live_router)
app.include_router(scenario_router)

configured_origins = os.environ.get("FRONTEND_ORIGINS")
allowed_origins = (
    [
        origin.strip().rstrip("/")
        for origin in configured_origins.split(",")
        if origin.strip()
    ]
    if configured_origins is not None
    else [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "India Climate Twin",
        "status": "running",
        "message": "Climate intelligence and What-If simulation API is ready",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
