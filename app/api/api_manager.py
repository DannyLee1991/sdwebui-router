from fastapi import APIRouter
from .model import ResponseModel

router = APIRouter()


@router.post("")
def api_test(name: str) -> ResponseModel:
    return ResponseModel(data={"name": name})
