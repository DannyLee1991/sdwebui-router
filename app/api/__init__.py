from fastapi import APIRouter
from fastapi import FastAPI
from .api_manager import router as manager_router
from .api_sd import router as sd_router

app = FastAPI()


@app.get("/health_check")
def health_check_view():
    return {"status": "ok"}


api_router = APIRouter()
# router注入
api_router.include_router(sd_router, prefix="/sd", tags=["sd"])
api_router.include_router(manager_router, prefix="/manager", tags=["manager"])

app.include_router(api_router, prefix="/api")
