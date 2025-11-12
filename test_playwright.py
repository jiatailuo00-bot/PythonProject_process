from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

BASE_URLS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
]


def _goto_app(page: Page) -> None:
    for url in BASE_URLS:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=8000)
            page.wait_for_timeout(1500)
            return
        except Exception:
            continue
    raise RuntimeError("无法打开脚本控制台，请确认前端 dev server 已启动在 5173/5174 端口")


def run_sop_flow(excel_path: Path) -> dict:
    if not excel_path.exists():
        raise FileNotFoundError(f"未找到测试文件: {excel_path}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _goto_app(page)

        # 选中 SOP 脚本
        page.get_by_text("SOP流程标注", exact=True).first.click()

        # 打开文件助手并上传 Excel
        page.get_by_role("button", name="文件助手", exact=True).click()
        upload_input = page.locator("input[type='file']").first
        upload_input.set_input_files(str(excel_path))
        page.get_by_text("文件上传成功").wait_for(timeout=15000)

        # 运行脚本
        run_button = page.get_by_role("button", name="运行脚本")
        run_button.click()

        # 等待执行完成
        page.get_by_text("执行完成").wait_for(timeout=240000)

        # 读取结果面板内的 JSON 数据
        result_pre = page.locator(".result-pre").first
        expect(result_pre).to_be_visible(timeout=1000)
        result_text = result_pre.text_content() or "{}"
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json = {"raw": result_text}

        browser.close()
        return result_json


if __name__ == "__main__":
    output = run_sop_flow(Path("bad12.xlsx").resolve())
    print("UI 自动化运行完成，返回数据:")
    print(json.dumps(output, ensure_ascii=False, indent=2))
