
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rainfall_api import router as rainfall_router
from prediction_api import router as prediction_router
from temperature_api import router as temperature_router

app = FastAPI(
    title="India Climate Twin API",
    description="Climate data, AI predictions and simulation services",
    version="0.1.0",
)

app.include_router(rainfall_router)
app.include_router(temperature_router)
app.include_router(prediction_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "India Climate Twin",
        "status": "running",
        "message": "Climate intelligence API is ready",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
