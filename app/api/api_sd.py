from fastapi import APIRouter
from app.core.pool import BusyException
from loguru import logger
from .model import ResponseModel, RWModel

router = APIRouter()


class GenImg(RWModel):
    mode: str = "txt2img"
    prompt: str = ""
    setup_params: dict = {}
    sd_params: dict


@router.post("/gen")
def gen_img(item: GenImg):
    logger.info(f"start gen_img: {item}")
    from app import pool
    try:
        ckpt_model_name = item.setup_params.get("base_model", {}).get("name", "")
        controlnet_model_list = [unit['model'] for unit in item.sd_params["controlnet_units"]]
        res = pool.pick(ckpt_model_name, controlnet_model_list)
        with res:
            data = res.process(item)
            return ResponseModel(data=data)
    except BusyException as e:
        return ResponseModel(data={}, status=203, message=f"{e}")
    except Exception as e:
        logger.exception(e)
        return ResponseModel(data={}, status=500, message=f"{e}")
