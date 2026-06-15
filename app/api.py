from fastapi import FastAPI

app = FastAPI(
    title="SciTime-Agent API",
    description="Scientific time-series modeling agent backend",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SciTime-Agent backend is running."
    }