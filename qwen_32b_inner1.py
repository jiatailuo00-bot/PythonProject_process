"""LLM helper that talks to the internal Qwen 32B endpoint."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import requests


DEFAULT_QWEN_API_URL = "http://10.72.1.166:33032/v1/chat/completions"
# DEFAULT_QWEN_API_URL = "http://10.72.1.166:33034/v1/chat/completions"
# 公司内网 Qwen3-32B 默认密钥，若环境变量提供同名 KEY 会覆盖此值
DEFAULT_QWEN_API_KEY = (
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiYWQ0OTUwMzIxMmQ0OGQ3OTAzNmE0NTgyZDM2ZmE1MiIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc0NzEyNjI1OCIsImV4cCI6IjE3NDcxMjk4NTgifQ.gT0U8JXXk66iRjbfF7qZs21quXu1WbkEgUEZvmbyub5DEdx9Bbu353tsJl7sGaJ_6-QJtBfkHHSG2sA7_zfsZ6Syg6bkm97CWN8fFoK_oqPBzZLh6eu06Pzh5PaVpdfZYRoIMNyOm5xw5zeFkRGlqdW4jhXt8v-IeAZ-x2mgs3HLD7yhPqowKDhlEcmIQu_ZBvdJEbWXe_qOpemOmSQgZVo-e87fG04y1R3R90Ql0Bl4OZAXaiFXVW-DUiaRr-Zsxa_OXjHqxMPy5v9utnjhxpZ7ttTetLFEoorVrHImIcfHma-xRMFEPryLLNr5p9XXoPLJBzmi9-uD8M4Ov7MRxQ"

    # "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiYWQ0OTUwMzIxMmQ0OGQ3OTAzNmE0NTgyZDM2ZmE1MiIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc1MzQ5NDc0NCIsImV4cCI6IjE3NTM0OTgzNDQifQ.rKJ3xnQvcNoDafDsBCE87KaZvmJby_JfnFk8yI0c3_7HIkyp_DHKHBNCL3MQCJyz-Uch5dBVTOirFTKOD7MvofFIX6k8O9tWO2-3HHqTZLdqPEyfJVLooj-hDpeerecaCfk3iHOCrbdFmA4gzK2b4B53mvmBCff4LOP4LrG0hZ-219QHnbRX5E2mKMkQGlOds9hQM8Q5JRTVWwbumBwpsNqSRrIxEJwzd-300Nk6a1FLlpkE-FQs-7Kp8VxiMHaKhLSABL6oqAxALhtWnFOy2FkSMVbVxF9IS4gyRCU-uawvpeC8p7qD9-gxu4cH6BsD-V1o2moGyd46OXTL4PAa0g"

)
DEFAULT_QWEN_MODEL = "Qwen3-32B"
DEFAULT_QWEN_TIMEOUT = 300.0


@dataclass
class QwenConfig:
    api_url: str = os.environ.get("QWEN_API_URL", DEFAULT_QWEN_API_URL)
    api_key: str = os.environ.get("QWEN_API_KEY", DEFAULT_QWEN_API_KEY)
    model: str = os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    timeout: float = float(os.environ.get("QWEN_TIMEOUT", DEFAULT_QWEN_TIMEOUT))


class Qwen32BClient:
    def __init__(self, config: QwenConfig | None = None) -> None:
        self.config = config or QwenConfig()
        if not self.config.api_url:
            raise ValueError("Missing Qwen API URL")

    def generate(self, prompt: str, system: str = "", temperature: float = 0) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        resp = requests.post(
            self.config.api_url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unexpected Qwen response: {data}") from exc


__all__ = ["QwenConfig", "Qwen32BClient"]


def main(prompt: str = "你好，请简单介绍一下珍酒") -> None:
    client = Qwen32BClient()
    response = client.generate(prompt, system="你是一名珍酒产品专家")
    print(f"Prompt: {prompt}\nResponse: {response}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"调用 Qwen 接口失败：{exc}")