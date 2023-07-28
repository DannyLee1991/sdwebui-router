import requests
from retry import retry
from loguru import logger


class FileDownloader:
    def __init__(self, origin):
        self.origin = origin

    @retry(tries=3, delay=1, backoff=2, max_delay=4)
    def check(self, filepath):
        try:
            r = requests.post(f"{self.origin}/check", json={"filepath": filepath})
            if r.ok:
                return r.json()["data"]["exist"]
        except Exception as e:
            logger.error(e)
        # 为了尽量能够保证生成，默认都会放行
        raise True

    @retry(tries=3, delay=1, backoff=2, max_delay=4)
    def fetch(self, url, save_to):
        r = requests.post(f"{self.origin}/fetch", json={"url": url, "filepath": save_to})
        if r.ok:
            return True
        else:
            raise Exception(f"文件获取异常: {r.content}")

    @retry(tries=3, delay=1, backoff=2, max_delay=4)
    def make_copy(self, src, target):
        r = requests.post(f"{self.origin}/copy", json={"source": src, "target": target})
        if r.ok:
            return True
        else:
            raise Exception(f"文件获取异常: {r.content}")

    @retry(tries=3, delay=1, backoff=2, max_delay=4)
    def remove(self, filepath):
        r = requests.delete(f"{self.origin}/remove", json={"filepath": filepath})
        if r.ok:
            return True
        else:
            raise Exception(f"文件删除异常: {r.content}")
