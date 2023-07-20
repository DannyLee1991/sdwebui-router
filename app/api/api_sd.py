from fastapi import APIRouter
from app.core.pool import BusyException
from .model import ResponseModel, RWModel

router = APIRouter()


class GenImg(RWModel):
    mode: str = "txt2img"
    prompt: str = ""
    setup_params: dict = {}
    sd_params: dict


@router.post("/gen")
def gen_img(item: GenImg):
    from app import pool
    try:
        res = pool.pick()
        with res:
            data = res.process(item)
            return ResponseModel(data=data)
    except BusyException as e:
        return ResponseModel(data={}, status=203, message=f"{e}")
    except Exception as e:
        return ResponseModel(data={}, status=500, message=f"{e}")
