#!/usr/bin/env python3
"""
快速测试脚本 - 验证SOP流程标注工具的核心功能
"""

import asyncio
from playwright.async_api import async_playwright

async def quick_test():
    """快速测试核心功能"""
    print("🚀 开始快速测试...")

    # 检查服务器状态
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:5174", timeout=5)
        print("✅ 服务器运行正常")
    except:
        print("❌ 服务器未运行，请先启动前端服务")
        return

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()

    results = []

    try:
        # 1. 页面加载
        await page.goto("http://localhost:5174")
        await page.wait_for_load_state("networkidle")
        title = await page.title()
        results.append(("页面加载", "frontend" in title.lower()))

        # 2. 脚本选择
        await page.wait_for_selector("select.script-dropdown", timeout=5000)
        scripts = await page.query_selector_all("select.script-dropdown option")
        results.append(("脚本选择", len(scripts) >= 3))

        # 3. 文件上传区域
        upload_area = await page.query_selector(".upload-area")
        results.append(("上传区域", upload_area is not None))

        # 4. 文件历史
        file_history = await page.query_selector(".file-history-section")
        results.append(("文件历史", file_history is not None))

        # 5. 运行按钮
        run_button = await page.query_selector(".run-script-btn")
        results.append(("运行按钮", run_button is not None))

        # 截图
        await page.screenshot(path="quick_test_screenshot.png")

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        results.append(("测试执行", False))
    finally:
        await browser.close()

    # 打印结果
    print("\n📊 快速测试结果:")
    passed = 0
    for test, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test}")
        if success:
            passed += 1

    success_rate = (passed / len(results)) * 100 if results else 0
    print(f"\n🎯 通过率: {success_rate:.1f}% ({passed}/{len(results)})")

    if success_rate >= 80:
        print("🎉 应用状态良好！")
    elif success_rate >= 60:
        print("⚠️ 应用基本可用，但有一些问题")
    else:
        print("❌ 应用存在严重问题，需要修复")

if __name__ == "__main__":
    asyncio.run(quick_test())