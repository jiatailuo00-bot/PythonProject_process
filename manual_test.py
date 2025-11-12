#!/usr/bin/env python3
"""
手动测试脚本 - 逐步验证网页应用功能
"""

import asyncio
from playwright.async_api import async_playwright

async def manual_test():
    """手动测试步骤"""
    print("🚀 开始手动测试...")
    print("请按照以下步骤手动验证功能：")
    print()

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False, slow_mo=1000)
    page = await browser.new_page()

    try:
        # 1. 打开网页
        print("1️⃣ 打开网页...")
        await page.goto("http://localhost:5174")
        await page.wait_for_load_state("networkidle")
        print("   ✅ 网页已加载，请查看浏览器窗口")

        # 等待用户确认
        input("   按回车键继续...")
        print()

        # 2. 检查脚本选择
        print("2️⃣ 测试脚本选择...")
        await page.wait_for_selector("select.script-dropdown", timeout=10000)

        # 获取所有脚本选项
        scripts = await page.query_selector_all("select.script-dropdown option")
        print(f"   发现 {len(scripts)} 个脚本选项:")

        for i, script in enumerate(scripts):
            text = await script.text_content()
            value = await script.get_attribute("value")
            print(f"   - {text}")

        print("   请手动测试：")
        print("   a) 点击下拉选择框，选择不同脚本")
        print("   b) 观察参数配置区域是否变化")
        input("   完成测试后按回车键继续...")
        print()

        # 3. 测试文件上传
        print("3️⃣ 测试文件上传...")
        upload_area = await page.query_selector(".upload-area")
        if upload_area:
            print("   ✅ 找到上传区域")
            print("   请手动测试：")
            print("   a) 点击上传区域")
            print("   b) 或者拖拽文件到上传区域")
            input("   完成测试后按回车键继续...")
        print()

        # 4. 测试参数配置
        print("4️⃣ 测试参数配置...")
        params_section = await page.query_selector(".params-section")
        if params_section:
            print("   ✅ 找到参数配置区域")
            print("   请手动测试：")
            print("   a) 选择不同的脚本")
            print("   b) 观察参数配置的变化")
            print("   c) 尝试填写参数")
            input("   完成测试后按回车键继续...")
        print()

        # 5. 测试脚本执行
        print("5️⃣ 测试脚本执行...")
        run_button = await page.query_selector(".run-script-btn")
        if run_button:
            print("   ✅ 找到运行按钮")
            is_disabled = await run_button.is_disabled()
            if is_disabled:
                print("   ⚠️ 按钮当前不可用，请先选择脚本和填写参数")
            else:
                print("   ✅ 按钮可用，可以尝试运行脚本")

            print("   请手动测试：")
            print("   a) 选择脚本")
            print("   b) 上传文件")
            print("   c) 填写参数")
            print("   d) 点击运行脚本按钮")
            input("   完成测试后按回车键继续...")
        print()

        # 6. 测试文件历史
        print("6️⃣ 测试文件历史...")
        file_history = await page.query_selector(".file-history-section")
        if file_history:
            print("   ✅ 找到文件历史区域")
            print("   请手动测试：")
            print("   a) 查看文件列表")
            print("   b) 点击选择按钮")
            print("   c) 点击复制路径按钮")
            print("   d) 点击刷新按钮")
            input("   完成测试后按回车键继续...")
        print()

        # 7. 测试最近执行
        print("7️⃣ 测试最近执行...")
        recent_executions = await page.query_selector(".recent-executions-section")
        if recent_executions:
            print("   ✅ 找到最近执行区域")
            print("   查看是否有执行记录")
            input("   完成测试后按回车键继续...")
        print()

        print("🎉 手动测试完成！")
        print("请根据您的测试结果反馈功能是否正常。")

        # 截图
        await page.screenshot(path="manual_test_final.png")
        print("📸 已保存最终截图: manual_test_final.png")

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
    finally:
        await browser.close()

if __name__ == "__main__":
    asyncio.run(manual_test())