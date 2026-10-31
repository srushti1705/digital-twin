
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="India Climate Twin API",
    description="Climate data, AI predictions and simulation services",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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

