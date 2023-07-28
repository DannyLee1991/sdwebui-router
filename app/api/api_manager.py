from fastapi import APIRouter
from .model import ResponseModel

router = APIRouter()


@router.post("/register")
def api_register(origin: str) -> ResponseModel:
    """
    注册服务资源
    :param origin:
    :return:
    """
    from app import pool
    try:
        pool.register(origin)
        return ResponseModel(data={})
    except Exception as e:
        return ResponseModel(data={}, status=500, message=f"{e}")


@router.delete("/register")
def api_unregister(host: str) -> ResponseModel:
    """
    删除服务资源
    :param host:
    :return:
    """
    from app import pool
    try:
        pool.unregister(host)
        return ResponseModel(data={})
    except Exception as e:
        return ResponseModel(data={}, status=500, message=f"{e}")


@router.get("/list")
def api_res_list() -> ResponseModel:
    """
    查看res列表
    :return:
    """
    from app import pool
    data = pool.list_res()
    return ResponseModel(data=data)
