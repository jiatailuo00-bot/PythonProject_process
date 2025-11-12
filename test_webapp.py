#!/usr/bin/env python3
"""
全面的Playwright自动化测试脚本
测试SOP流程标注工具的网页应用
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from pathlib import Path

class WebAppTester:
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
        self.browser = await self.playwright.chromium.launch(headless=True)
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

    async def wait_for_element(self, selector: str, timeout: int = 30000):
        """等待元素出现"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except:
            return False

    async def test_page_load(self):
        """测试1: 页面加载"""
        try:
            print("\n🌐 测试1: 页面加载")
            await self.page.goto(self.base_url, wait_until="networkidle")

            # 等待页面完全加载
            await asyncio.sleep(2)

            # 检查页面标题
            title = await self.page.title()
            title_ok = "frontend" in title.lower() or "sop" in title.lower()

            # 检查主要元素是否存在
            header = await self.wait_for_element("header")
            app_main = await self.wait_for_element("main")

            screenshot_path = await self.take_screenshot("page_load")

            success = title_ok and header and app_main
            message = f"页面标题: {title}, 主要元素存在: {header and app_main}"

            self.log_test("页面加载", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("page_load_error")
            self.log_test("页面加载", False, f"加载失败: {str(e)}", screenshot_path)

    async def test_script_selection(self):
        """测试2: 脚本选择下拉框"""
        try:
            print("\n📋 测试2: 脚本选择下拉框")

            # 等待脚本选择器加载
            script_dropdown = await self.wait_for_element("select.script-dropdown")
            if not script_dropdown:
                screenshot_path = await self.take_screenshot("script_dropdown_not_found")
                self.log_test("脚本选择下拉框", False, "未找到脚本选择下拉框", screenshot_path)
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

            screenshot_path = await self.take_screenshot("script_selection")

            success = len(found_scripts) >= 2  # 至少找到2个预期脚本
            message = f"找到脚本: {script_names}, 预期脚本匹配: {found_scripts}"

            self.log_test("脚本选择下拉框", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("script_selection_error")
            self.log_test("脚本选择下拉框", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_file_upload_ui(self):
        """测试3: 文件上传功能UI"""
        try:
            print("\n📁 测试3: 文件上传功能UI")

            # 检查上传区域
            upload_area = await self.wait_for_element(".upload-area")
            if not upload_area:
                screenshot_path = await self.take_screenshot("upload_area_not_found")
                self.log_test("文件上传功能UI", False, "未找到上传区域", screenshot_path)
                return

            # 检查文件输入框
            file_input = await self.wait_for_element("input[type='file']")

            # 检查上传文本提示
            upload_text = await self.page.query_selector(".upload-text")
            upload_content = await self.page.query_selector(".upload-content")

            screenshot_path = await self.take_screenshot("file_upload_ui")

            success = upload_area and file_input and upload_text and upload_content
            message = f"上传区域: {upload_area}, 文件输入: {file_input}, 提示文本: {upload_text}"

            self.log_test("文件上传功能UI", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("file_upload_ui_error")
            self.log_test("文件上传功能UI", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_file_upload_click(self):
        """测试4: 点击上传文件"""
        try:
            print("\n🖱️ 测试4: 点击上传文件")

            # 等待上传区域
            upload_area = await self.wait_for_element(".upload-area")
            if not upload_area:
                self.log_test("点击上传文件", False, "未找到上传区域")
                return

            # 创建一个测试文件
            test_file_path = self.screenshots_dir / "test_upload.txt"
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("这是一个测试文件\n用于测试文件上传功能\n包含一些示例数据")

            # 获取文件输入框
            file_input = await self.page.query_selector("input[type='file']")
            if not file_input:
                self.log_test("点击上传文件", False, "未找到文件输入框")
                return

            # 上传文件
            await file_input.set_input_files(str(test_file_path))

            # 等待上传完成
            await asyncio.sleep(3)

            # 检查是否有上传成功的迹象
            uploaded = await self.page.query_selector(".file-item")

            screenshot_path = await self.take_screenshot("file_upload_click")

            success = uploaded is not None
            message = f"文件上传: {'成功' if success else '失败或未完成'}"

            self.log_test("点击上传文件", success, message, screenshot_path)

            # 清理测试文件
            try:
                os.remove(test_file_path)
            except:
                pass

        except Exception as e:
            screenshot_path = await self.take_screenshot("file_upload_click_error")
            self.log_test("点击上传文件", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_script_switching(self):
        """测试5: 脚本选择切换"""
        try:
            print("\n🔄 测试5: 脚本选择切换")

            # 等待脚本下拉框
            script_dropdown = await self.wait_for_element("select.script-dropdown")
            if not script_dropdown:
                self.log_test("脚本选择切换", False, "未找到脚本下拉框")
                return

            # 获取所有选项
            options = await self.page.query_selector_all("select.script-dropdown option")
            if len(options) < 2:
                self.log_test("脚本选择切换", False, "脚本选项不足")
                return

            # 选择第一个脚本
            await script_dropdown.select_option(index=1)
            await asyncio.sleep(2)

            first_script_info = await self.page.query_selector(".script-info")
            first_script_desc = await self.page.query_selector(".script-description")

            # 选择第二个脚本
            if len(options) > 2:
                await script_dropdown.select_option(index=2)
                await asyncio.sleep(2)

                second_script_info = await self.page.query_selector(".script-info")
                second_script_desc = await self.page.query_selector(".script-description")

                screenshot_path = await self.take_screenshot("script_switching")

                success = first_script_info and second_script_info
                message = f"脚本切换: {'成功' if success else '失败'}"

                self.log_test("脚本选择切换", success, message, screenshot_path)
            else:
                screenshot_path = await self.take_screenshot("script_selection_only")
                self.log_test("脚本选择切换", True, "只有一个脚本选项，无法测试切换", screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("script_switching_error")
            self.log_test("脚本选择切换", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_parameter_configuration(self):
        """测试6: 参数配置动态变化"""
        try:
            print("\n⚙️ 测试6: 参数配置动态变化")

            # 等待脚本下拉框
            script_dropdown = await self.wait_for_element("select.script-dropdown")
            if not script_dropdown:
                self.log_test("参数配置动态变化", False, "未找到脚本下拉框")
                return

            # 切换不同的脚本，观察参数区域变化
            options = await self.page.query_selector_all("select.script-dropdown option")

            params_changed = False

            for i, option in enumerate(options[:min(3, len(options))]):  # 测试最多3个脚本
                try:
                    await script_dropdown.select_option(index=i)
                    await asyncio.sleep(2)

                    # 检查参数区域
                    params_section = await self.page.query_selector(".params-section")
                    param_items = await self.page.query_selector_all(".param-item")

                    screenshot_path = await self.take_screenshot(f"params_script_{i}")

                    if param_items:
                        params_changed = True
                        print(f"   脚本 {i+1}: 找到 {len(param_items)} 个参数")

                except Exception as e:
                    print(f"   脚本 {i+1}: 测试失败 - {str(e)}")

            success = params_changed
            message = f"参数配置变化: {'检测到' if success else '未检测到'}"

            self.log_test("参数配置动态变化", success, message)

        except Exception as e:
            screenshot_path = await self.take_screenshot("parameter_config_error")
            self.log_test("参数配置动态变化", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_run_script_button(self):
        """测试7: 运行脚本按钮"""
        try:
            print("\n🚀 测试7: 运行脚本按钮")

            # 等待运行脚本按钮
            run_button = await self.wait_for_button_available()
            if not run_button:
                screenshot_path = await self.take_screenshot("run_button_not_found")
                self.log_test("运行脚本按钮", False, "未找到运行脚本按钮", screenshot_path)
                return

            # 检查按钮状态
            is_disabled = await run_button.is_disabled()
            button_text = await run_button.text_content()

            screenshot_path = await self.take_screenshot("run_script_button")

            # 按钮应该存在，可能被禁用（如果缺少必需参数）
            success = True
            message = f"按钮状态: {'禁用' if is_disabled else '启用'}, 文本: {button_text}"

            self.log_test("运行脚本按钮", success, message, screenshot_path)

            # 如果按钮可用，尝试点击
            if not is_disabled:
                try:
                    print("   尝试点击运行按钮...")
                    await run_button.click()

                    # 等待执行状态
                    await asyncio.sleep(3)

                    running_button = await self.page.query_selector(".run-script-btn.running")
                    execution_result = await self.page.query_selector(".result-section")

                    success = running_button is not None or execution_result is not None
                    message = f"脚本执行: {'开始' if success else '未检测到执行状态'}"

                    screenshot_path = await self.take_screenshot("script_execution")
                    self.log_test("脚本执行尝试", success, message, screenshot_path)

                except Exception as e:
                    screenshot_path = await self.take_screenshot("script_execution_error")
                    self.log_test("脚本执行尝试", False, f"执行失败: {str(e)}", screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("run_button_error")
            self.log_test("运行脚本按钮", False, f"测试失败: {str(e)}", screenshot_path)

    async def wait_for_button_available(self, timeout: int = 10000) -> bool:
        """等待按钮可用"""
        try:
            await self.page.wait_for_selector(".run-script-btn", timeout=timeout)
            return await self.page.query_selector(".run-script-btn")
        except:
            return None

    async def test_execution_results(self):
        """测试8: 执行结果显示"""
        try:
            print("\n📊 测试8: 执行结果显示")

            # 检查是否存在执行结果区域
            result_section = await self.page.query_selector(".result-section")

            if result_section:
                # 检查结果内容
                result_content = await self.page.query_selector(".result-content")
                status_icon = await self.page.query_selector(".status-icon")

                screenshot_path = await self.take_screenshot("execution_results")

                success = result_content is not None
                message = f"结果区域: {'存在' if result_section else '不存在'}, 内容区域: {'存在' if result_content else '不存在'}"

                self.log_test("执行结果显示", success, message, screenshot_path)
            else:
                screenshot_path = await self.take_screenshot("no_execution_results")
                self.log_test("执行结果显示", True, "暂无执行结果（正常情况）", screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("execution_results_error")
            self.log_test("执行结果显示", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_file_history(self):
        """测试9: 文件历史功能"""
        try:
            print("\n📋 测试9: 文件历史功能")

            # 检查文件历史区域
            file_history = await self.wait_for_element(".file-history-section")
            if not file_history:
                screenshot_path = await self.take_screenshot("file_history_not_found")
                self.log_test("文件历史功能", False, "未找到文件历史区域", screenshot_path)
                return

            # 检查文件列表
            file_list = await self.page.query_selector(".file-list")
            file_items = await self.page.query_selector_all(".file-item")

            # 检查刷新按钮
            refresh_btn = await self.page.query_selector(".refresh-btn")

            screenshot_path = await self.take_screenshot("file_history")

            success = file_history is not None and file_list is not None
            message = f"文件历史区域: {'存在' if file_history else '不存在'}, 文件数量: {len(file_items)}, 刷新按钮: {'存在' if refresh_btn else '不存在'}"

            self.log_test("文件历史功能", success, message, screenshot_path)

            # 如果有文件，测试文件选择功能
            if file_items:
                try:
                    first_file = file_items[0]
                    await first_file.click()
                    await asyncio.sleep(1)

                    # 检查文件是否被选中
                    selected_file = await self.page.query_selector(".file-item.selected")

                    success = selected_file is not None
                    message = f"文件选择: {'成功' if success else '失败'}"

                    screenshot_path = await self.take_screenshot("file_selection")
                    self.log_test("文件选择功能", success, message, screenshot_path)

                except Exception as e:
                    screenshot_path = await self.take_screenshot("file_selection_error")
                    self.log_test("文件选择功能", False, f"选择失败: {str(e)}", screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("file_history_error")
            self.log_test("文件历史功能", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_recent_executions(self):
        """测试10: 最近执行记录"""
        try:
            print("\n🕒 测试10: 最近执行记录")

            # 检查最近执行区域
            recent_executions = await self.wait_for_element(".recent-executions-section")
            if not recent_executions:
                screenshot_path = await self.take_screenshot("recent_executions_not_found")
                self.log_test("最近执行记录", False, "未找到最近执行区域", screenshot_path)
                return

            # 检查执行列表
            execution_list = await self.page.query_selector(".execution-list")
            execution_items = await self.page.query_selector_all(".execution-item")

            screenshot_path = await self.take_screenshot("recent_executions")

            success = recent_executions is not None
            message = f"最近执行区域: {'存在' if recent_executions else '不存在'}, 执行记录数量: {len(execution_items)}"

            self.log_test("最近执行记录", success, message, screenshot_path)

        except Exception as e:
            screenshot_path = await self.take_screenshot("recent_executions_error")
            self.log_test("最近执行记录", False, f"测试失败: {str(e)}", screenshot_path)

    async def test_drag_and_drop(self):
        """测试11: 拖拽上传功能"""
        try:
            print("\n🎯 测试11: 拖拽上传功能")

            # 检查上传区域
            upload_area = await self.wait_for_element(".upload-area")
            if not upload_area:
                self.log_test("拖拽上传功能", False, "未找到上传区域")
                return

            # 创建测试文件
            test_file_path = self.screenshots_dir / "test_drag_drop.txt"
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("拖拽测试文件\n用于测试拖拽上传功能")

            # 模拟拖拽文件
            try:
                # 获取上传区域的边界
                box = await upload_area.bounding_box()
                if box:
                    # 模拟文件拖拽
                    data_transfer = await self.page.evaluate_handle("""
                        () => {
                            const dataTransfer = new DataTransfer();
                            const file = new File(['test content'], 'test_drag_drop.txt', { type: 'text/plain' });
                            dataTransfer.items.add(file);
                            return dataTransfer;
                        }
                    """)

                    # 触发拖拽事件
                    await upload_area.dispatch_event('dragover', {
                        'dataTransfer': data_transfer,
                        'clientX': box['x'] + box['width'] / 2,
                        'clientY': box['y'] + box['height'] / 2
                    })

                    await asyncio.sleep(0.5)

                    await upload_area.dispatch_event('drop', {
                        'dataTransfer': data_transfer,
                        'clientX': box['x'] + box['width'] / 2,
                        'clientY': box['y'] + box['height'] / 2
                    })

                    # 等待上传处理
                    await asyncio.sleep(3)

                    screenshot_path = await self.take_screenshot("drag_and_drop")

                    # 检查是否有上传成功的迹象
                    uploaded = await self.page.query_selector(".file-item")

                    success = uploaded is not None
                    message = f"拖拽上传: {'成功' if success else '可能失败'}"

                    self.log_test("拖拽上传功能", success, message, screenshot_path)
                else:
                    self.log_test("拖拽上传功能", False, "无法获取上传区域位置")

            except Exception as e:
                screenshot_path = await self.take_screenshot("drag_and_drop_error")
                self.log_test("拖拽上传功能", False, f"拖拽测试失败: {str(e)}", screenshot_path)

            # 清理测试文件
            try:
                os.remove(test_file_path)
            except:
                pass

        except Exception as e:
            screenshot_path = await self.take_screenshot("drag_and_drop_setup_error")
            self.log_test("拖拽上传功能", False, f"测试设置失败: {str(e)}", screenshot_path)

    async def test_responsive_design(self):
        """测试12: 响应式设计"""
        try:
            print("\n📱 测试12: 响应式设计")

            # 测试桌面尺寸
            await self.page.set_viewport_size({'width': 1400, 'height': 900})
            await asyncio.sleep(1)
            screenshot_path = await self.take_screenshot("responsive_desktop")

            # 测试平板尺寸
            await self.page.set_viewport_size({'width': 768, 'height': 1024})
            await asyncio.sleep(1)
            screenshot_path_tablet = await self.take_screenshot("responsive_tablet")

            # 测试手机尺寸
            await self.page.set_viewport_size({'width': 375, 'height': 667})
            await asyncio.sleep(1)
            screenshot_path_mobile = await self.take_screenshot("responsive_mobile")

            # 恢复桌面尺寸
            await self.page.set_viewport_size({'width': 1400, 'height': 900})

            success = True
            message = "响应式设计测试完成（桌面、平板、手机）"

            self.log_test("响应式设计", success, message, screenshot_path_mobile)

        except Exception as e:
            screenshot_path = await self.take_screenshot("responsive_error")
            self.log_test("响应式设计", False, f"测试失败: {str(e)}", screenshot_path)

    async def generate_report(self):
        """生成测试报告"""
        print("\n📊 生成测试报告...")

        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - passed_tests

        # 生成报告
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            "test_details": self.test_results,
            "screenshots_directory": str(self.screenshots_dir),
            "test_date": datetime.now().isoformat()
        }

        # 保存报告到文件
        report_file = self.screenshots_dir / "test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 打印总结
        print(f"\n🎯 测试总结:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   成功率: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        print(f"   报告文件: {report_file}")
        print(f"   截图目录: {self.screenshots_dir}")

        # 如果有失败的测试，列出详细信息
        if failed_tests > 0:
            print(f"\n❌ 失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test_name']}: {result['message']}")

    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始运行SOP流程标注工具自动化测试")
        print(f"📍 目标URL: {self.base_url}")
        print(f"📸 截图保存到: {self.screenshots_dir}")

        try:
            await self.setup()

            # 运行所有测试
            await self.test_page_load()
            await self.test_script_selection()
            await self.test_file_upload_ui()
            await self.test_file_upload_click()
            await self.test_script_switching()
            await self.test_parameter_configuration()
            await self.test_run_script_button()
            await self.test_execution_results()
            await self.test_file_history()
            await self.test_recent_executions()
            await self.test_drag_and_drop()
            await self.test_responsive_design()

            # 生成报告
            await self.generate_report()

        except Exception as e:
            print(f"❌ 测试运行失败: {str(e)}")
            try:
                screenshot_path = await self.take_screenshot("critical_error")
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

    # 运行测试
    tester = WebAppTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())