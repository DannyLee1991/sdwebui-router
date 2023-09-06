import json
import time
import random
from typing import List
from loguru import logger
from app.module.file_downloader import FileDownloader
from webuiapi import WebUIApi
from webuiapi import ControlNetUnit
import requests
import base64
import pickle
import zipfile
from PIL import Image
from io import BytesIO
from app.core.model_load_history import History

S_RUNNING = "running"
S_IDLE = "idle"


class Res:
    origin: str
    status: str = S_IDLE

    def __init__(self, origin, dl_server_origin, status=S_IDLE, status_time=time.time(), ckpt_history_size=5,
                 controlnet_history_size=5):
        self.origin = origin
        self.status = status
        self.status_time = status_time
        self.file_downloader = FileDownloader(origin=dl_server_origin)
        self.webuiapi = WebUIApi(baseurl=f"{self.origin}/sdapi/v1")
        self.cpkt_history = History(size=ckpt_history_size)
        self.controlnet_history = History(size=controlnet_history_size)

    def __enter__(self):
        self._tic = time.time()
        self._occupy()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release()

    def get_state_duration(self):
        return time.time() - self.status_time

    def _occupy(self):
        logger.info(f"{self.origin} start occupy")
        self._update_status(S_RUNNING)

    def _release(self):
        logger.info(f"{self.origin} release, cost time: {time.time() - self._tic:.2f}s")
        self._update_status(S_IDLE)

    def _update_status(self, status):
        self.status = status
        self.status_time = time.time()

    def _setup(self, item):
        # 前置准备工作
        base_model = item.setup_params.get("base_model", {})
        base_model_name = base_model['name']

        vae_model = item.setup_params.get("vae_model", {})
        controlnet_list = item.setup_params.get("controlnet_list", [])
        lora_info_list = item.setup_params.get("lora_info_list", [])
        # ---lora---
        _prompts_lora_suffix = self._prepare_for_lora(lora_info_list)
        item.prompt = f"{item.prompt}{_prompts_lora_suffix}"

        # ---vae---
        target_vae_filepath = f"models/VAE/{vae_model.get('name', '')}"
        target_vae_copy_filepath = f"models/VAE/{base_model_name.split('.')[0]}.vae.pt"
        if vae_model:
            vae_copy_is_exist = self.file_downloader.check(target_vae_copy_filepath)
            if not vae_copy_is_exist:
                vae_is_exist = self.file_downloader.check(target_vae_filepath)
                if not vae_is_exist:
                    self.file_downloader.fetch(url=vae_model['url'], save_to=target_vae_filepath)
                self.file_downloader.make_copy(src=target_vae_filepath, target=target_vae_copy_filepath)
        else:
            if self.file_downloader.check(filepath=target_vae_copy_filepath):
                self.file_downloader.remove(filepath=target_vae_copy_filepath)

        # ---controlnet---
        for controlnet in controlnet_list:
            target_control_filepath = f"models/ControlNet/{controlnet['name']}"
            controlnet_is_exist = self.file_downloader.check(target_control_filepath)
            logger.info(f"controlnet是否存在 => {controlnet_is_exist} : {controlnet_is_exist}")
            if not controlnet_is_exist:
                self.file_downloader.fetch(
                    url=controlnet['url'],
                    save_to=target_control_filepath)
                logger.info(f"controlnet模型下载完成 {controlnet}")

        # ---基础模型---
        target_model_filepath = f"models/Stable-diffusion/{base_model_name}"
        model_is_exist = self.file_downloader.check(target_model_filepath)
        logger.info(f"基础模型是否已存在 => {target_model_filepath} : {model_is_exist}")
        if not model_is_exist:
            logger.info(f"开始下载基础模型: {base_model_name}")
            self.file_downloader.fetch(url=base_model['url'],
                                       save_to=target_model_filepath)
            logger.info(f"基础模型下载完成: {base_model_name}")
        # 刷新ckpt(即使模型存在 也需要refresh一下)
        self.webuiapi.refresh_checkpoints()

        # 参数预处理
        self._sd_params_preprocessing(item)
        # 切换模型
        self._switch_model(model=base_model_name)

    def _prepare_for_lora(self, lora_info_list=[]):
        """
        lora相关的预处理操作
        """
        logger.info("prepare_for_lora start")
        _prompts_lora_suffix = ""
        for lora_info in lora_info_list:
            lora_hash = lora_info["hash"]
            lora_alpha = lora_info["alpha"]
            lora_url = lora_info["url"]

            target_filepath = f"models/Lora/{lora_hash}.safetensors"
            is_exist = self.file_downloader.check(target_filepath)
            if not is_exist:
                self.file_downloader.fetch(
                    url=lora_url,
                    save_to=target_filepath)
                self.webuiapi.session.post(url=f"{self.webuiapi.baseurl}/refresh-loras")
            _prompts_lora_suffix += f",<lora:{lora_hash}:{lora_alpha}>"
        return _prompts_lora_suffix

    def _sd_params_preprocessing(self, item):
        logger.info(f"_sd_params_preprocessing start")
        # img2img参数预处理
        mode = item.mode
        if mode == "img2img":
            logger.info(f"_sd_params_preprocessing img2img start")
            # 将可读图片链接转化成PIL类型
            item.sd_params["images"] = [
                Image.open(BytesIO(requests.get(img, timeout=10).content)) for img in item.sd_params["images"]
            ]
        # 将sd_params 中的 mask_image 转化成PIL类型
        if "mask_image" in item.sd_params:
            logger.info(f"_sd_params_preprocessing mask_image start")
            item.sd_params["mask_image"] = Image.open(
                BytesIO(requests.get(item.sd_params["mask_image"], timeout=10).content)
            )
        #  将sd_params中的controlnet_units列表中的图片链接转化成PIL类型并替换
        if "controlnet_units" in item.sd_params:
            logger.info(f"_sd_params_preprocessing controlnet_units start")
            item.sd_params["controlnet_units"] = [
                ControlNetUnit(
                    input_image=Image.open(BytesIO(requests.get(unit.pop("input_image"), timeout=10).content)),
                    **unit
                )
                for unit in item.sd_params["controlnet_units"]
            ]
        # prompt参数设置
        item.sd_params["prompt"] = item.prompt
        # 修复sdwebui会错误携带controlnet模型的问题
        item.sd_params["alwayson_scripts"] = {}

    def _switch_model(self, model):
        logger.info(f"切换模型: {model}")
        if model is not None:
            self.webuiapi.util_set_model(model)
            self.cpkt_history.add(model)

    def process(self, item):
        # 初始化
        self._setup(item)
        # 开始触发生成
        result_data = getattr(self.webuiapi, item.mode)(**item.sd_params)
        data = {"info": result_data.info, "parameters": result_data.parameters, "images": []}
        for image in result_data.images:
            data["images"].append(base64.b64encode(pickle.dumps(image)))
        return data


class Pool:
    res_list: List[Res] = []

    def __init__(self, res_origin_list=[], dl_server_origin="", max_running_timeout=600):
        # 处于running态的最大超时时间
        self.max_running_timeout = max_running_timeout
        # file-downloader服务地址
        self.dl_server_origin = dl_server_origin
        # 注册资源
        for origin in res_origin_list:
            self.register(origin)

    def list_res(self):
        return [
            {"host": item.origin, "status": item.status,
             "state_duration": item.get_state_duration(),
             "ckpt_history": item.cpkt_history.data,
             "controlnet_history": item.controlnet_history.data
             } for item in
            self.res_list]

    def register(self, origin: str, ckpt_history_size=5, controlnet_history_size=5):
        if origin not in [res.origin for res in self.res_list]:
            self.res_list.append(Res(origin=origin,
                                     dl_server_origin=self.dl_server_origin,
                                     status_time=time.time(),
                                     ckpt_history_size=ckpt_history_size,
                                     controlnet_history_size=controlnet_history_size))
        else:
            raise Exception("host already in register res list")

    def unregister(self, host: str):
        for res in self.res_list:
            if res.origin == host:
                self.res_list.remove(res)
                break
        else:
            raise Exception("host not in register res list")

    def refresh(self):
        for res in self.res_list:
            if res.status == S_RUNNING:
                if res.get_state_duration() > self.max_running_timeout:
                    res._release()

    def idle_res_list(self, block=False, shuffle=True):
        while block:
            self.refresh()
            idle_res = []
            for res in self.res_list:
                if res.status == S_IDLE:
                    idle_res.append(res)
            if idle_res:
                if shuffle:
                    random.shuffle(idle_res)
                return idle_res
            if block:
                time.sleep(1)

    def pick(self, ckpt_model_name, controlnet_list=[]) -> Res:
        res_list = self.idle_res_list(block=True, shuffle=True)
        if res_list:
            # 智能路由逻辑，优先从有过记录的 闲置节点中触发生成
            score_list = []
            for i, res in enumerate(res_list):
                score = 0
                if res.cpkt_history.is_exist(ckpt_model_name) >= 0:
                    # checkpoint模型权重为1分
                    score += 2
                for cn in controlnet_list:
                    if res.controlnet_history.is_exist(cn) >= 0:
                        # controlnet模型权重为1分
                        score += 1
                score_list.append(score)
            idx = score_list.index(max(score_list))
            res = res_list[idx]
            logger.info(f"pick res from {len(res_list)} idle res => {res.origin}")
            return res
        else:
            raise BusyException("all res is busy")

    def _find_res_by_host(self, host: str) -> Res:
        for res in self.res_list:
            if res.origin == host:
                return res
        raise Exception(f"res {host} not found")


class BusyException(Exception):
    pass


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
