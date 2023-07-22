import time
import random
import re
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

S_RUNNING = "running"
S_IDLE = "idle"


class Res:
    host: str
    status: str = S_IDLE

    def __init__(self, host, status=S_IDLE, status_time=time.time()):
        self.host = host
        self.status = status
        self.status_time = status_time
        self.file_downloader = FileDownloader(host=self.host, port=8000)
        self.webuiapi = WebUIApi(host=self.host)

    def __enter__(self):
        self._tic = time.time()
        self._occupy()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release()

    def get_state_duration(self):
        return time.time() - self.status_time

    def _occupy(self):
        logger.info(f"{self.host} start occupy")
        self._update_status(S_RUNNING)

    def _release(self):
        logger.info(f"{self.host} release, cost time: {time.time() - self._tic:.2f}s")
        self._update_status(S_IDLE)

    def _update_status(self, status):
        self.status = status
        self.status_time = time.time()

    def _setup(self, item):
        # 前置准备工作
        base_model = item.setup_params.get("base_model", {})
        vae_model = item.setup_params.get("vae_model", {})
        controlnet_list = item.setup_params.get("controlnet_list", [])
        lora_info_list = item.setup_params.get("lora_info_list", [])
        # ---lora---
        _prompts_lora_suffix = self._prepare_for_lora(lora_info_list)
        item.prompt = f"{item.prompt}{_prompts_lora_suffix}"

        # ---vae---
        target_vae_filepath = f"data/VAE/{vae_model.get('name', '')}"
        target_vae_copy_filepath = f"data/VAE/{base_model['name'].split('.')[0]}.vae.pt"
        if vae_model:
            vae_copy_is_exist = self.file_downloader.check(target_vae_copy_filepath)
            if not vae_copy_is_exist:
                vae_is_exist = self.file_downloader.check(target_vae_filepath)
                if not vae_is_exist:
                    self.file_downloader.fetch(url=vae_model['url'], save_to=target_vae_filepath)
                self.file_downloader.make_copy(src=target_vae_filepath, target=target_vae_copy_filepath)
        else:
            self.file_downloader.remove(filepath=target_vae_copy_filepath)

        # ---controlnet---
        for controlnet in controlnet_list:
            target_control_filepath = f"data/ControlNet/{controlnet['name']}"
            controlnet_is_exist = self.file_downloader.check(target_control_filepath)
            logger.info(f"controlnet是否存在 => {controlnet_is_exist} : {controlnet_is_exist}")
            if not controlnet_is_exist:
                self.file_downloader.fetch(
                    url=controlnet['url'],
                    save_to=target_control_filepath)
                logger.info(f"controlnet模型下载完成 {controlnet}")

        # ---基础模型---
        target_model_filepath = f"data/StableDiffusion/{base_model['name']}"
        model_is_exist = self.file_downloader.check(target_model_filepath)
        logger.info(f"基础模型是否已存在 => {target_model_filepath} : {model_is_exist}")
        if not model_is_exist:
            logger.info(f"开始下载基础模型: {base_model['name']}")
            self.file_downloader.fetch(url=base_model['url'],
                                       save_to=target_model_filepath)
            logger.info(f"基础模型下载完成: {base_model['name']}")
            # 刷新ckpt
            self.webuiapi.refresh_checkpoints()

        # 参数预处理
        self._sd_params_preprocessing(item)
        # 切换模型
        self._switch_model(model=item.setup_params.get("base_model")['name'])

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

            target_filepath = f"data/Lora/{lora_hash}.safetensors"
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
                Image.open(BytesIO(requests.get(img, timeout=10).content)) for img in cp_kwargs["sd_params"]["images"]
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

    def _switch_model(self, model):
        logger.info(f"切换模型: {model}")
        if model is not None:
            self.webuiapi.util_set_model(model)

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

    def __init__(self, res_host_list=[], max_running_timeout=600):
        # 处于running态的最大超时时间
        self.max_running_timeout = max_running_timeout
        # 注册资源
        for host in res_host_list:
            self.register(host)

    def list_res(self):
        return [{"host": item.host, "status": item.status,
                 "state_duration": item.get_state_duration()} for item in
                self.res_list]

    def register(self, host: str):
        if not is_valid_ipv4(host):
            raise Exception(f"{host} is not valid ipv4 address")
        elif host not in [res.host for res in self.res_list]:
            self.res_list.append(Res(host=host, status_time=time.time()))
        else:
            raise Exception("host already in register res list")

    def unregister(self, host: str):
        for res in self.res_list:
            if res.host == host:
                self.res_list.remove(res)
                break
        else:
            raise Exception("host not in register res list")

    def refresh(self):
        for res in self.res_list:
            if res.status == S_RUNNING:
                if res.get_state_duration() > self.max_running_timeout:
                    res._release()

    def idle_res_list(self, block=False):
        while block:
            self.refresh()
            idle_res = []
            for res in self.res_list:
                if res.status == S_IDLE:
                    idle_res.append(res)
            if idle_res:
                return idle_res
            if block:
                time.sleep(1)

    def pick(self) -> Res:
        res_list = self.idle_res_list(block=True)
        if res_list:
            res = random.choice(res_list)
            logger.info(f"pick res from {len(res_list)} idle res => {res.host}")
            return res
        else:
            raise BusyException("all res is busy")

    def _find_res_by_host(self, host: str) -> Res:
        for res in self.res_list:
            if res.host == host:
                return res
        raise Exception(f"res {host} not found")


class BusyException(Exception):
    pass


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def is_valid_ipv4(ip_str):
    # Regular expression pattern for IPv4 address
    ipv4_pattern = r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$'

    # Check if the input string matches the IPv4 pattern
    match = re.match(ipv4_pattern, ip_str)

    if match:
        # Check each octet to ensure it is in the valid range (0 to 255)
        octets = [int(octet) for octet in match.groups()]
        if all(0 <= octet <= 255 for octet in octets):
            return True
    return False
