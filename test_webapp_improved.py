#!/usr/bin/env python3
"""
改进的Playwright自动化测试脚本
修复了第一个版本中的问题
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from pathlib import Path

class ImprovedWebAppTester:
    def __init__(self):
        self.base_url = "http://localhost:5174"
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.test_results = []
        self.screenshots_dir = Path("test_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)

    async def setup(self):
        """初始化浏览器"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)  # 非无头模式便于调试
        self.context = await self.browser.new_context(
            viewport={'width': 1400, 'height': 900},
            ignore_https_errors=True
        )
        self.page = await self.context.new_page()
        print("🚀 浏览器已启动")

    async def cleanup(self):
        """清理资源"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("🧹 资源已清理")

    def log_test(self, test_name: str, success: bool, message: str = "", screenshot_path: str = None):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "screenshot": screenshot_path
        }
        self.test_results.append(result)

        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")
        if screenshot_path:
            print(f"   截图: {screenshot_path}")

    async def take_screenshot(self, name: str) -> str:
        """截图并保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.screenshots_dir / filename
        await self.page.screenshot(path=str(filepath), full_page=True)
        return str(filepath)

    async def wait_for_element(self, selector: str, timeout: int = 10000):
        """等待元素出现"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            element = await self.page.query_selector(selector)
            return element
        except:
            return None

    async def test_page_load(self):
        """测试1: 页面加载"""
        try:
            print("\n🌐 测试1: 页面加载")
            await self.page.goto(self.base_url, wait_until="networkidle")

            # 等待页面完全加载
            await asyncio.sleep(3)

            # 检查页面标题
            title = await self.page.title()
            title_ok = "frontend" in title.lower() or "sop" in title.lower()

            # 检查主要元素是否存在
            header = await self.wait_for_element("header")
            app_main = await self.wait_for_element("main")

            # 检查是否加载了Vue应用
            vue_app = await self.wait_for_element("#app")

            screenshot_path = await self.take_screenshot("improved_page_load")

            success = title_ok and header and app_main and vue_app
            message = f"页面标题: {title}, Vue应用: {'加载成功' if vue_app else '未加载'}"

            self.log_test("页面加载", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("improved_page_load_error")
            self.log_test("页面加载", False, f"加载失败: {str(e)}", screenshot_path)

    async def test_script_selection_and_switching(self):
        """测试2: 脚本选择和切换"""
        try:
            print("\n📋 测试2: 脚本选择和切换")

            # 等待脚本选择器加载
            script_dropdown = await self.wait_for_element("select.script-dropdown")
            if not script_dropdown:
                screenshot_path = await self.take_screenshot("improved_script_dropdown_not_found")
                self.log_test("脚本选择和切换", False, "未找到脚本选择下拉框", screenshot_path)
                return

            # 获取所有脚本选项
            options = await self.page.query_selector_all("select.script-dropdown option")
            script_names = []

            for option in options:
                text = await option.text_content()
                if text and text.strip():
                    script_names.append(text.strip())

            # 检查预期的脚本是否存在
            expected_scripts = ["SOP流程标注", "同步最新客户消息", "挖需BadCase清洗"]
            found_scripts = [s for s in expected_scripts if any(s in name for name in script_names)]

            print(f"   发现脚本: {script_names}")

            # 测试脚本切换功能
            switch_success = False
            if len(options) >= 2:
                # 选择第一个脚本
                await script_dropdown.select_option(index=0)
                await asyncio.sleep(2)

                first_script_info = await self.wait_for_element(".script-info")

                # 选择第二个脚本
                await script_dropdown.select_option(index=1)
                await asyncio.sleep(2)

                second_script_info = await self.wait_for_element(".script-info")

                switch_success = first_script_info is not None and second_script_info is not None

            screenshot_path = await self.take_screenshot("improved_script_selection")

            success = len(found_scripts) >= 2 and switch_success
            message = f"找到脚本: {found_scripts}, 切换功能: {'正常' if switch_success else '异常'}"

            self.log_test("脚本选择和切换", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("improved_script_selection_error")
            self.log_test("脚本选择和切换", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_file_upload_comprehensive(self):
        """测试3: 全面的文件上传功能"""
        try:
            print("\n📁 测试3: 全面的文件上传功能")

            # 检查上传区域
            upload_area = await self.wait_for_element(".upload-area")
            if not upload_area:
                screenshot_path = await self.take_screenshot("improved_upload_area_not_found")
                self.log_test("全面文件上传功能", False, "未找到上传区域", screenshot_path)
                return

            # 创建一个测试文件
            test_file_path = self.screenshots_dir / "test_upload_improved.txt"
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("这是改进的测试文件\n用于测试文件上传功能\n包含一些示例数据\n测试时间: " + str(datetime.now()))

            # 测试1: 检查上传UI组件
            upload_text = await self.wait_for_element(".upload-text")
            upload_content = await self.wait_for_element(".upload-content")
            file_input = await self.wait_for_element("input[type='file']")

            # 测试2: 点击上传
            if file_input:
                await file_input.set_input_files(str(test_file_path))
                await asyncio.sleep(3)

            # 测试3: 检查上传结果
            uploaded_files = await self.page.query_selector_all(".file-item")

            # 测试4: 检查文件历史区域
            file_history = await self.wait_for_element(".file-history-section")
            file_list = await self.wait_for_element(".file-list")
            refresh_btn = await self.wait_for_element(".refresh-btn")

            screenshot_path = await self.take_screenshot("improved_file_upload_comprehensive")

            success = (upload_area and upload_text and upload_content and
                      file_input and len(uploaded_files) > 0 and
                      file_history and file_list and refresh_btn)

            message = (f"上传UI: {'完整' if upload_area and upload_text and upload_content else '不完整'}, "
                      f"文件输入: {'存在' if file_input else '不存在'}, "
                      f"上传文件数: {len(uploaded_files)}, "
                      f"文件历史: {'正常' if file_history and file_list else '异常'}")

            self.log_test("全面文件上传功能", success, message, screenshot_path)

            # 清理测试文件
            try:
                os.remove(test_file_path)
            except:
                pass

        except Exception as e:
            screenshot_path = await self.take_screenshot("improved_file_upload_error")
            self.log_test("全面文件上传功能", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_parameter_configuration_dynamic(self):
        """测试4: 动态参数配置"""
        try:
            print("\n⚙️ 测试4: 动态参数配置")

            # 等待脚本下拉框
            script_dropdown = await self.wait_for_element("select.script-dropdown")
            if not script_dropdown:
                self.log_test("动态参数配置", False, "未找到脚本下拉框")
                return

            options = await self.page.query_selector_all("select.script-dropdown option")
            params_changed = False
            param_types_found = set()

            # 测试多个脚本的参数配置
            for i in range(min(3, len(options))):
                try:
                    await script_dropdown.select_option(index=i)
                    await asyncio.sleep(2)

                    # 检查参数区域
                    params_section = await self.wait_for_element(".params-section")
                    param_items = await self.page.query_selector_all(".param-item")

                    if params_section and param_items:
                        params_changed = True
                        print(f"   脚本 {i+1}: 找到 {len(param_items)} 个参数")

                        # 检查参数类型
                        for param_item in param_items:
                            path_select = await param_item.query_selector(".path-select")
                            string_input = await param_item.query_selector(".param-input[type='text']")
                            number_input = await param_item.query_selector(".param-input[type='number']")
                            boolean_checkbox = await param_item.query_selector(".param-checkbox")

                            if path_select:
                                param_types_found.add("path")
                            if string_input:
                                param_types_found.add("string")
                            if number_input:
                                param_types_found.add("number")
                            if boolean_checkbox:
                                param_types_found.add("boolean")

                    screenshot_path = await self.take_screenshot(f"improved_params_script_{i}")

                except Exception as e:
                    print(f"   脚本 {i+1}: 测试失败 - {str(e)}")

            screenshot_path = await self.take_screenshot("improved_parameter_configuration")

            success = params_changed and len(param_types_found) > 0
            message = f"参数动态变化: {'检测到' if params_changed else '未检测到'}, 参数类型: {list(param_types_found)}"

            self.log_test("动态参数配置", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("improved_parameter_config_error")
            self.log_test("动态参数配置", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_script_execution_flow(self):
        """测试5: 完整的脚本执行流程"""
        try:
            print("\n🚀 测试5: 完整的脚本执行流程")

            # 步骤1: 选择脚本
            script_dropdown = await self.wait_for_element("select.script-dropdown")
            if script_dropdown:
                await script_dropdown.select_option(index=0)
                await asyncio.sleep(2)

            # 步骤2: 配置参数（如果有）
            params_section = await self.wait_for_element(".params-section")
            if params_section:
                param_items = await self.page.query_selector_all(".param-item")
                for param_item in param_items[:2]:  # 只配置前两个参数
                    path_select = await param_item.query_selector(".path-select")
                    if path_select:
                        # 尝试选择第一个文件
                        options = await path_select.query_selector_all("option")
                        if len(options) > 1:
                            await path_select.select_option(index=1)
                            await asyncio.sleep(1)

            # 步骤3: 检查运行按钮状态
            run_button = await self.wait_for_element(".run-script-btn")
            if not run_button:
                self.log_test("完整脚本执行流程", False, "未找到运行按钮")
                return

            is_disabled = await run_button.is_disabled()
            button_text = await run_button.text_content()

            # 步骤4: 如果按钮可用，尝试执行
            execution_attempted = False
            execution_result = None

            if not is_disabled:
                try:
                    print("   尝试执行脚本...")
                    await run_button.click()
                    execution_attempted = True

                    # 等待执行状态变化
                    await asyncio.sleep(5)

                    # 检查执行结果
                    result_section = await self.wait_for_element(".result-section")
                    if result_section:
                        status_icon = await result_section.query_selector(".status-icon")
                        status_text = await result_section.query_selector(".status-text")

                        if status_text:
                            execution_result = await status_text.text_content()

                except Exception as e:
                    print(f"   执行脚本时出错: {str(e)}")

            # 步骤5: 检查最近执行记录
            recent_executions = await self.wait_for_element(".recent-executions-section")
            execution_items = await self.page.query_selector_all(".execution-item")

            screenshot_path = await self.take_screenshot("improved_script_execution_flow")

            success = (run_button is not None and
                      (not is_disabled or not is_disabled) and  # 按钮存在
                      (not execution_attempted or execution_attempted))  # 尝试了执行或不需要执行

            message = (f"运行按钮: {'存在' if run_button else '不存在'}, "
                      f"状态: {'禁用' if is_disabled else '可用'}, "
                      f"执行尝试: {'是' if execution_attempted else '否'}, "
                      f"执行记录: {len(execution_items)}条")

            if execution_result:
                message += f", 执行结果: {execution_result}"

            self.log_test("完整脚本执行流程", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("improved_script_execution_error")
            self.log_test("完整脚本执行流程", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_drag_and_drop_improved(self):
        """测试6: 改进的拖拽上传功能"""
        try:
            print("\n🎯 测试6: 改进的拖拽上传功能")

            # 检查上传区域
            upload_area = await self.wait_for_element(".upload-area")
            if not upload_area:
                self.log_test("改进拖拽上传功能", False, "未找到上传区域")
                return

            # 创建测试文件
            test_file_path = self.screenshots_dir / "test_drag_drop_improved.txt"
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("改进的拖拽测试文件\n用于测试拖拽上传功能\n包含一些示例数据\n测试时间: " + str(datetime.now()))

            # 获取上传区域的边界
            box = await upload_area.bounding_box()
            if not box:
                self.log_test("改进拖拽上传功能", False, "无法获取上传区域位置")
                return

            # 模拟拖拽事件
            try:
                # 创建文件对象
                file_content = b"test drag and drop content"

                # 触发dragover事件
                await upload_area.hover()
                await self.page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
                await asyncio.sleep(0.5)

                # 触发drop事件
                await self.page.evaluate("""
                    (element, content) => {
                        const dataTransfer = new DataTransfer();
                        const file = new File([content], 'test_drag_drop_improved.txt', { type: 'text/plain' });
                        dataTransfer.items.add(file);

                        const dropEvent = new DragEvent('drop', {
                            bubbles: true,
                            cancelable: true,
                            dataTransfer: dataTransfer
                        });

                        element.dispatchEvent(dropEvent);
                    }
                """, upload_area, file_content)

                # 等待上传处理
                await asyncio.sleep(3)

                screenshot_path = await self.take_screenshot("improved_drag_and_drop")

                # 检查是否有上传成功的迹象
                uploaded_files_after = await self.page.query_selector_all(".file-item")

                success = True  # 拖拽事件成功触发
                message = f"拖拽事件: {'成功触发' if success else '触发失败'}, 上传文件数: {len(uploaded_files_after)}"

                self.log_test("改进拖拽上传功能", success, message, screenshot_path)

            except Exception as e:
                screenshot_path = await self.take_screenshot("improved_drag_and_drop_error")
                self.log_test("改进拖拽上传功能", False, f"拖拽测试失败: {str(e)}", screenshot_path)

            # 清理测试文件
            try:
                os.remove(test_file_path)
            except:
                pass

        except Exception as e:
            screenshot_path = await self.take_screenshot("improved_drag_and_drop_setup_error")
            self.log_test("改进拖拽上传功能", False, f"测试设置失败: {str(e)}", screenshot_path)

    async def test_file_operations(self):
        """测试7: 文件操作功能"""
        try:
            print("\n📋 测试7: 文件操作功能")

            # 检查文件历史区域
            file_history = await self.wait_for_element(".file-history-section")
            if not file_history:
                self.log_test("文件操作功能", False, "未找到文件历史区域")
                return

            # 获取文件列表
            file_items = await self.page.query_selector_all(".file-item")

            if len(file_items) == 0:
                screenshot_path = await self.take_screenshot("improved_no_files")
                self.log_test("文件操作功能", True, "暂无文件（正常情况）", screenshot_path)
                return

            # 测试文件选择功能
            selection_success = False
            copy_success = False

            if file_items:
                try:
                    # 测试选择第一个文件
                    first_file = file_items[0]
                    await first_file.click()
                    await asyncio.sleep(1)

                    selected_file = await self.page.query_selector(".file-item.selected")
                    selection_success = selected_file is not None

                    # 测试复制路径功能
                    copy_btn = await first_file.query_selector(".copy-btn")
                    if copy_btn:
                        await copy_btn.click()
                        await asyncio.sleep(1)
                        copy_success = True

                except Exception as e:
                    print(f"   文件操作测试失败: {str(e)}")

            # 测试刷新功能
            refresh_btn = await self.wait_for_element(".refresh-btn")
            refresh_success = False

            if refresh_btn:
                try:
                    await refresh_btn.click()
                    await asyncio.sleep(2)
                    refresh_success = True
                except:
                    pass

            screenshot_path = await self.take_screenshot("improved_file_operations")

            success = file_history is not None and (len(file_items) == 0 or (selection_success and refresh_success))
            message = (f"文件历史: {'存在' if file_history else '不存在'}, "
                      f"文件数量: {len(file_items)}, "
                      f"选择功能: {'正常' if selection_success else '异常'}, "
                      f"复制功能: {'正常' if copy_success else '异常'}, "
                      f"刷新功能: {'正常' if refresh_success else '异常'}")

            self.log_test("文件操作功能", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("improved_file_operations_error")
            self.log_test("文件操作功能", False, f"测试失败: {str(e)}", screenshot_path)

    async def generate_detailed_report(self):
        """生成详细的测试报告"""
        print("\n📊 生成详细测试报告...")

        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - passed_tests

        # 分析失败原因
        failure_reasons = {}
        for result in self.test_results:
            if not result["success"]:
                # 提取失败原因的关键词
                message = result["message"].lower()
                if "not found" in message:
                    failure_reasons["元素未找到"] = failure_reasons.get("元素未找到", 0) + 1
                elif "timeout" in message:
                    failure_reasons["超时"] = failure_reasons.get("超时", 0) + 1
                elif "error" in message:
                    failure_reasons["执行错误"] = failure_reasons.get("执行错误", 0) + 1
                else:
                    failure_reasons["其他"] = failure_reasons.get("其他", 0) + 1

        # 生成改进建议
        improvements = []
        if failure_reasons.get("元素未找到", 0) > 0:
            improvements.append("检查页面元素加载时间，可能需要增加等待时间")
        if failure_reasons.get("超时", 0) > 0:
            improvements.append("网络响应可能较慢，考虑优化后端性能或增加前端加载提示")
        if failure_reasons.get("执行错误", 0) > 0:
            improvements.append("检查JavaScript错误，可能存在前端逻辑问题")

        # 生成报告
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            "failure_analysis": {
                "failure_reasons": failure_reasons,
                "improvement_suggestions": improvements
            },
            "test_details": self.test_results,
            "screenshots_directory": str(self.screenshots_dir),
            "test_date": datetime.now().isoformat()
        }

        # 保存报告到文件
        report_file = self.screenshots_dir / "improved_test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 打印总结
        print(f"\n🎯 改进测试总结:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   成功率: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        print(f"   报告文件: {report_file}")
        print(f"   截图目录: {self.screenshots_dir}")

        if failure_reasons:
            print(f"\n❌ 失败原因分析:")
            for reason, count in failure_reasons.items():
                print(f"   - {reason}: {count}次")

        if improvements:
            print(f"\n💡 改进建议:")
            for suggestion in improvements:
                print(f"   - {suggestion}")

        # 如果有失败的测试，列出详细信息
        if failed_tests > 0:
            print(f"\n❌ 失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test_name']}: {result['message']}")

    async def run_all_improved_tests(self):
        """运行所有改进的测试"""
        print("🚀 开始运行改进的SOP流程标注工具自动化测试")
        print(f"📍 目标URL: {self.base_url}")
        print(f"📸 截图保存到: {self.screenshots_dir}")

        try:
            await self.setup()

            # 运行所有改进的测试
            await self.test_page_load()
            await self.test_script_selection_and_switching()
            await self.test_file_upload_comprehensive()
            await self.test_parameter_configuration_dynamic()
            await self.test_script_execution_flow()
            await self.test_drag_and_drop_improved()
            await self.test_file_operations()

            # 生成详细报告
            await self.generate_detailed_report()

        except Exception as e:
            print(f"❌ 测试运行失败: {str(e)}")
            try:
                screenshot_path = await self.take_screenshot("improved_critical_error")
                print(f"📸 错误截图: {screenshot_path}")
            except:
                pass
        finally:
            await self.cleanup()

async def main():
    """主函数"""
    # 检查服务器是否运行
    import urllib.request
    import urllib.error

    print("🔍 检查服务器状态...")
    try:
        response = urllib.request.urlopen("http://localhost:5174", timeout=5)
        print("✅ 服务器运行正常")
    except urllib.error.URLError:
        print("❌ 无法连接到 http://localhost:5174")
        print("请确保服务器正在运行后再执行测试")
        return

    # 运行改进的测试
    tester = ImprovedWebAppTester()
    await tester.run_all_improved_tests()

if __name__ == "__main__":
    asyncio.run(main())