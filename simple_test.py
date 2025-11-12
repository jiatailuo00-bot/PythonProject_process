#!/usr/bin/env python3
"""
简单测试脚本 - 基础功能验证
"""

import asyncio
from playwright.async_api import async_playwright

async def simple_test():
    """简单功能测试"""
    print("🚀 开始简单功能测试...")

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False, slow_mo=500)  # 显示浏览器，慢速操作
    page = await browser.new_page()

    try:
        # 1. 打开网页
        print("1️⃣ 打开网页...")
        await page.goto("http://localhost:5174")
        await page.wait_for_load_state("networkidle")
        print("   ✅ 网页加载完成")

        # 检查标题
        title = await page.title()
        print(f"   页面标题: {title}")

        # 2. 检查脚本选择
        print("\n2️⃣ 检查脚本选择...")
        script_dropdown = await page.query_selector("select.script-dropdown")
        if script_dropdown:
            print("   ✅ 找到脚本选择下拉框")

            # 获取选项文本
            options = await script_dropdown.query_selector_all("option")
            print(f"   发现 {len(options)} 个脚本选项:")
            for option in options:
                text = await option.text_content()
                if text and text.strip():
                    print(f"   - {text.strip()}")
        else:
            print("   ❌ 未找到脚本选择下拉框")

        # 3. 检查上传区域
        print("\n3️⃣ 检查上传区域...")
        upload_area = await page.query_selector(".upload-area")
        if upload_area:
            print("   ✅ 找到上传区域")
            upload_text = await upload_area.text_content()
            print(f"   上传区域内容: {upload_text[:100]}...")
        else:
            print("   ❌ 未找到上传区域")

        # 4. 检查文件历史
        print("\n4️⃣ 检查文件历史...")
        file_history = await page.query_selector(".file-history-section")
        if file_history:
            print("   ✅ 找到文件历史区域")
            file_items = await file_history.query_selector_all(".file-item")
            print(f"   发现 {len(file_items)} 个历史文件")
        else:
            print("   ❌ 未找到文件历史区域")

        # 5. 检查参数配置
        print("\n5️⃣ 检查参数配置...")
        params_section = await page.query_selector(".params-section")
        if params_section:
            print("   ✅ 找到参数配置区域")
            # 检查是否有参数输入框
            inputs = await params_section.query_selector_all("input, select")
            print(f"   发现 {len(inputs)} 个参数输入框")
        else:
            print("   ❌ 未找到参数配置区域")

        # 6. 检查运行按钮
        print("\n6️⃣ 检查运行按钮...")
        run_button = await page.query_selector(".run-script-btn")
        if run_button:
            print("   ✅ 找到运行按钮")
            button_text = await run_button.text_content()
            print(f"   按钮文本: {button_text}")
            is_disabled = await run_button.is_disabled()
            print(f"   按钮状态: {'禁用' if is_disabled else '启用'}")
        else:
            print("   ❌ 未找到运行按钮")

        # 7. 检查最近执行
        print("\n7️⃣ 检查最近执行...")
        recent_executions = await page.query_selector(".recent-executions-section")
        if recent_executions:
            print("   ✅ 找到最近执行区域")
        else:
            print("   ❌ 未找到最近执行区域")

        # 等待一段时间，让用户查看
        print("\n⏳ 等待10秒，您可以手动测试功能...")
        await asyncio.sleep(10)

        # 最终截图
        await page.screenshot(path="simple_test_screenshot.png")
        print("\n📸 已保存截图: simple_test_screenshot.png")

        print("\n🎉 简单测试完成！")
        print("请查看浏览器窗口中的网页，手动验证功能是否正常。")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    finally:
        # 保持浏览器打开一段时间
        print("\n⏳ 浏览器将保持打开状态，您可以继续手动测试...")
        await asyncio.sleep(30)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(simple_test())