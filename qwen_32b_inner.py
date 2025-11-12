import argparse
import io
import json
import logging
import os
import re
import threading
import time
import traceback
from datetime import datetime

import requests
import urllib3
from jinja2 import DebugUndefined, Template

import pandas as pd


logger = logging.getLogger(__name__)



DEFAULT_AUTH_TOKEN = (
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiYWQ0OTUwMzIxMmQ0OGQ3OTAzNmE0NTgyZDM2ZmE1MiIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc0NzEyNjI1OCIsImV4cCI6IjE3NDcxMjk4NTgifQ.gT0U8JXXk66iRjbfF7qZs21quXu1WbkEgUEZvmbyub5DEdx9Bbu353tsJl7sGaJ_6-QJtBfkHHSG2sA7_zfsZ6Syg6bkm97CWN8fFoK_oqPBzZLh6eu06Pzh5PaVpdfZYRoIMNyOm5xw5zeFkRGlqdW4jhXt8v-IeAZ-x2mgs3HLD7yhPqowKDhlEcmIQu_ZBvdJEbWXe_qOpemOmSQgZVo-e87fG04y1R3R90Ql0Bl4OZAXaiFXVW-DUiaRr-Zsxa_OXjHqxMPy5v9utnjhxpZ7ttTetLFEoorVrHImIcfHma-xRMFEPryLLNr5p9XXoPLJBzmi9-uD8M4Ov7MRxQ"
)


def _build_authorization_header() -> str:
    token = os.getenv("QWEN32B_AUTH_TOKEN", DEFAULT_AUTH_TOKEN).strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def llm_qwen3_32b(prompt, temperature=0.1, timeout=360):
    """内网Qwen32B API调用函数。"""
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": _build_authorization_header(),
    }
    url = "http://10.72.1.166:33032/v1/chat/completions"
    body = {
        "model": "Qwen3-32B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "user": "zhaowu42129",
    }

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        res = requests.post(
            url,
            headers=headers,
            data=json.dumps(body, ensure_ascii=False),
            verify=False,
            timeout=timeout,
        )
        res.raise_for_status()
        res_dict = res.json()
        answer = res_dict["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout as timeout_err:
        logger.error("请求超时: %s\nPROMPT: %s", timeout_err, prompt)
        answer = ""
    except Exception as err:
        logger.error("API 异常: %s", err, exc_info=True)
        try:
            print(res.content)  # type: ignore[has-type]
        except Exception:
            pass
        answer = ""
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-32B 连通性测试工具")
    parser.add_argument(
        "--prompt",
        default="请用一句话介绍你自己。",
        help="发送给模型的提示词",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="采样温度设置",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="请求超时时间（秒）",
    )
    args = parser.parse_args()

    print(">>> 正在调用Qwen32B接口...")
    response = llm_qwen3_32b(
        args.prompt,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    if response:
        print(">>> 接口返回：")
        print(response)
    else:
        print(">>> 调用失败或未获取到有效响应")


if __name__ == "__main__":
    main()
