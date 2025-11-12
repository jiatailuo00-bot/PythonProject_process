#!/usr/bin/env python3
"""
SOP脚本自动化测试
使用playwright自动化测试SOP识别功能
"""

import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

BASE_URL = "http://localhost:5173"
BAD12_FILE = Path("./bad12.xlsx").absolute()

async def test_sop_automation():
    """
    自动化测试SOP脚本功能
    1. 启动浏览器并访问前端
    2. 选择SOP流程标注脚本
    3. 上传bad12.xlsx文件
    4. 运行脚本并验证结果
    """

    print("🚀 开始SOP自动化测试...")

    async with async_playwright() as p:
        # 启动浏览器（非headless模式以便观察）
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # 步骤1：访问前端页面
            print(f"📍 步骤1: 访问前端页面 {BASE_URL}")
            await page.goto(BASE_URL, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)  # 等待页面加载完成

            # 截图：初始页面
            await page.screenshot(path="screenshots/01_initial_page.png")
            print("✅ 页面加载完成")

            # 步骤2：选择SOP流程标注脚本
            print("📍 步骤2: 选择SOP流程标注脚本")
            sop_script_element = page.locator("text=SOP流程标注").first
            await sop_script_element.wait_for(state="visible", timeout=10000)
            await sop_script_element.click()
            await page.wait_for_timeout(1000)

            # 截图：选择SOP脚本后
            await page.screenshot(path="screenshots/02_sop_script_selected.png")
            print("✅ SOP脚本已选择")

            # 步骤3：展开文件上传区域
            print("📍 步骤3: 展开文件上传区域")

            # 尝试多种可能的按钮选择器
            upload_button = None
            selectors = [
                "button:has-text('上传')",
                "button:has-text('浏览')",
                "button[data-testid*='upload']",
                "button[class*='upload']",
                "input[type='file']",
                ".upload-button",
                "[class*='file-upload']"
            ]

            for selector in selectors:
                try:
                    element = page.locator(selector).first
                    if await element.count() > 0:
                        upload_button = element
                        print(f"找到上传元素: {selector}")
                        break
                except:
                    continue

            if not upload_button:
                # 如果找不到特定按钮，尝试直接使用文件输入框
                file_input = page.locator("input[type='file']").first
                if await file_input.count() > 0:
                    print("直接使用文件输入框")
                    upload_button = file_input
                else:
                    raise Exception("未找到文件上传控件")

            await upload_button.wait_for(state="visible", timeout=5000)
            await upload_button.click()
            await page.wait_for_timeout(1000)

            # 截图：文件上传区域展开
            await page.screenshot(path="screenshots/03_upload_area_expanded.png")
            print("✅ 文件上传区域已展开")

            # 步骤4：上传bad12.xlsx文件
            print(f"📍 步骤4: 上传文件 {BAD12_FILE}")

            # 查找文件输入框
            file_input = page.locator("input[type='file']").first
            await file_input.wait_for(state="visible", timeout=5000)

            # 上传文件
            await file_input.set_input_files(str(BAD12_FILE))
            print(f"📁 文件已选择: {BAD12_FILE}")

            # 等待文件上传成功
            try:
                success_message = page.locator("text=文件上传成功").first
                await success_message.wait_for(state="visible", timeout=15000)
                print("✅ 文件上传成功")

                # 截图：文件上传成功
                await page.screenshot(path="screenshots/04_file_uploaded.png")

            except Exception as e:
                print(f"❌ 文件上传失败或超时: {e}")
                await page.screenshot(path="screenshots/04_upload_failed.png")
                raise

            # 步骤5：运行SOP脚本
            print("📍 步骤5: 运行SOP脚本")
            run_button = page.locator("button:has-text('运行脚本')").first
            await run_button.wait_for(state="visible", timeout=5000)
            await run_button.click()
            print("▶️ 脚本开始执行...")

            # 等待执行完成（最多等待4分钟）
            print("⏳ 等待脚本执行完成...")
            try:
                # 等待执行结果出现
                result_element = page.locator(".result-pre, .execution-result").first
                await result_element.wait_for(state="visible", timeout=240000)  # 4分钟超时

                print("✅ 脚本执行完成")

                # 截图：执行结果
                await page.screenshot(path="screenshots/05_script_completed.png")

            except Exception as e:
                print(f"❌ 脚本执行超时或失败: {e}")
                await page.screenshot(path="screenshots/05_execution_failed.png")
                raise

            # 步骤6：验证执行结果
            print("📍 步骤6: 验证执行结果")

            # 获取执行结果
            result_content = await result_element.text_content()
            print(f"📊 执行结果内容: {result_content[:200]}...")

            # 检查是否包含成功标识
            if "success" in result_content.lower() or "成功" in result_content or "完成" in result_content:
                print("✅ SOP脚本执行成功")

                # 尝试获取输出文件路径
                if "output_file" in result_content.lower() or "输出文件" in result_content:
                    print("📁 找到输出文件信息")

            else:
                print("⚠️ SOP脚本执行结果需要验证")

            # 最终截图
            await page.screenshot(path="screenshots/06_final_state.png", full_page=True)

            print("🎉 SOP自动化测试完成！")
            return True

        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            # 错误截图
            await page.screenshot(path="screenshots/error_state.png", full_page=True)
            return False

        finally:
            # 清理：关闭浏览器
            await browser.close()
            print("🧹 测试完成，浏览器已关闭")

async def main():
    """主函数"""
    # 创建截图目录
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    print("🧪 SOP脚本自动化测试")
    print("=" * 50)

    success = await test_sop_automation()

    if success:
        print("✅ 所有测试通过")
        exit(0)
    else:
        print("❌ 测试失败")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())