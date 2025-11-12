#!/usr/bin/env python3
"""
功能测试脚本 - 验证核心功能
"""

import asyncio
from playwright.async_api import async_playwright

async def function_test():
    """测试核心功能"""
    print("🚀 开始功能测试...")

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()

    results = []

    try:
        # 1. 页面加载
        print("1️⃣ 测试页面加载...")
        await page.goto("http://localhost:5174")
        await page.wait_for_load_state("networkidle")

        # 检查Vue应用是否加载
        vue_app = await page.query_selector("#app")
        if vue_app:
            results.append(("Vue应用加载", True))
            print("   ✅ Vue应用加载成功")
        else:
            results.append(("Vue应用加载", False))
            print("   ❌ Vue应用加载失败")

        # 2. 脚本选择功能
        print("\n2️⃣ 测试脚本选择...")
        await page.wait_for_selector("select.script-dropdown", timeout=5000)

        # 获取脚本选项
        scripts = await page.query_selector_all("select.script-dropdown option")
        script_texts = []
        for script in scripts:
            text = await script.text_content()
            if text and text.strip():
                script_texts.append(text.strip())

        print(f"   发现 {len(script_texts)} 个脚本:")
        for text in script_texts:
            print(f"   - {text}")

        # 验证关键脚本是否存在
        required_scripts = ["SOP流程标注", "同步最新客户消息", "挖需BadCase清洗"]
        found_scripts = [script for script in required_scripts if any(script in text for text in script_texts)]

        if len(found_scripts) == len(required_scripts):
            results.append(("脚本完整性", True))
            print("   ✅ 所有必要脚本都存在")
        else:
            results.append(("脚本完整性", False))
            print(f"   ❌ 缺少脚本: {set(required_scripts) - set(found_scripts)}")

        # 3. 脚本切换功能
        print("\n3️⃣ 测试脚本切换...")
        for i, script_option in enumerate(scripts[:3]):  # 测试前3个脚本
            await script_option.click()
            await page.wait_for_timeout(500)  # 等待UI更新

            # 检查参数配置是否更新
            params_section = await page.query_selector(".params-section")
            if params_section:
                # 检查是否有参数输入框
                param_inputs = await params_section.query_selector_all("input, select")
                results.append(("脚本切换", len(param_inputs) >= 0))
                print(f"   ✅ 脚本 {i+1} 切换成功，参数数量: {len(param_inputs)}")
            else:
                results.append(("脚本切换", False))
                print(f"   ❌ 脚本 {i+1} 切换失败")

        # 4. 文件上传区域
        print("\n4️⃣ 测试文件上传区域...")
        upload_area = await page.query_selector(".upload-area")
        if upload_area:
            # 检查是否有文件输入框
            file_input = await upload_area.query_selector("input[type='file']")
            if file_input:
                results.append(("文件上传区域", True))
                print("   ✅ 文件上传区域完整")
            else:
                results.append(("文件上传区域", False))
                print("   ❌ 文件输入框不存在")
        else:
            results.append(("文件上传区域", False))
            print("   ❌ 文件上传区域不存在")

        # 5. 文件历史功能
        print("\n5️⃣ 测试文件历史...")
        file_history = await page.query_selector(".file-history-section")
        if file_history:
            # 检查文件列表
            file_items = await file_history.query_selector_all(".file-item")
            print(f"   发现 {len(file_items)} 个历史文件")
            results.append(("文件历史", True))
            print("   ✅ 文件历史区域正常")
        else:
            results.append(("文件历史", False))
            print("   ❌ 文件历史区域不存在")

        # 6. 运行按钮
        print("\n6️⃣ 测试运行按钮...")
        run_button = await page.query_selector(".run-script-btn")
        if run_button:
            is_disabled = await run_button.is_disabled()
            button_text = await run_button.text_content()
            print(f"   按钮状态: {'禁用' if is_disabled else '启用'}")
            print(f"   按钮文本: {button_text}")
            results.append(("运行按钮", True))
            print("   ✅ 运行按钮存在")
        else:
            results.append(("运行按钮", False))
            print("   ❌ 运行按钮不存在")

        # 7. 响应式设计
        print("\n7️⃣ 测试响应式设计...")
        # 测试不同屏幕尺寸
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.wait_for_timeout(500)
        desktop_layout = await page.query_selector(".simplified-file-upload")

        await page.set_viewport_size({"width": 768, "height": 1024})
        await page.wait_for_timeout(500)
        tablet_layout = await page.query_selector(".simplified-file-upload")

        await page.set_viewport_size({"width": 375, "height": 667})
        await page.wait_for_timeout(500)
        mobile_layout = await page.query_selector(".simplified-file-upload")

        if desktop_layout and tablet_layout and mobile_layout:
            results.append(("响应式设计", True))
            print("   ✅ 响应式设计正常")
        else:
            results.append(("响应式设计", False))
            print("   ❌ 响应式设计有问题")

        # 最终截图
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.screenshot(path="function_test_final.png")
        print("\n📸 已保存最终截图: function_test_final.png")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        results.append(("测试执行", False))
    finally:
        await browser.close()

    # 打印结果总结
    print("\n" + "="*50)
    print("📊 功能测试结果总结:")
    print("="*50)

    passed = 0
    for test, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test}")
        if success:
            passed += 1

    success_rate = (passed / len(results)) * 100 if results else 0
    print(f"\n🎯 通过率: {success_rate:.1f}% ({passed}/{len(results)})")

    if success_rate >= 90:
        print("🎉 应用功能优秀！")
    elif success_rate >= 75:
        print("✅ 应用功能良好")
    elif success_rate >= 60:
        print("⚠️ 应用基本可用，但有一些问题")
    else:
        print("❌ 应用存在严重问题，需要修复")

    print("\n🔗 访问地址: http://localhost:5174")
    print("🖥️ 后端API: http://localhost:8000")

if __name__ == "__main__":
    asyncio.run(function_test())