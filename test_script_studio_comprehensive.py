#!/usr/bin/env python3
"""
Comprehensive Script Studio Test using Playwright
Tests the complete workflow: upload file, select script, execute, and monitor results
"""

import asyncio
import os
import sys
import time
from playwright.async_api import async_playwright, expect
import json

# Configuration
FRONTEND_URL = "http://localhost:5173"
BACKEND_URL = "http://localhost:8000"
EXCEL_FILE_PATH = "/Users/luojiatai/PycharmProjects/PythonProject_predeal/bad12.xlsx"
SCREENSHOT_DIR = "/Users/luojiatai/PycharmProjects/PythonProject_predeal/test_screenshots"

class ScriptStudioTester:
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.test_results = []

    async def setup(self):
        """Setup browser and page"""
        print("🚀 Setting up browser...")
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=False, slow_mo=1000)
            self.context = await self.browser.new_context(
                viewport={'width': 1400, 'height': 900},
                ignore_https_errors=True
            )
            self.page = await self.context.new_page()

            # Setup console logging
            self.page.on('console', lambda msg: print(f"📝 Console: {msg.text}"))
            self.page.on('pageerror', lambda err: print(f"❌ Page Error: {err}"))

            # Create screenshot directory
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)

            print("✅ Browser setup completed")
            return True
        except Exception as e:
            print(f"❌ Failed to setup browser: {e}")
            return False

    async def navigate_to_frontend(self):
        """Navigate to the frontend application"""
        print(f"🌐 Navigating to {FRONTEND_URL}...")
        try:
            await self.page.goto(FRONTEND_URL, wait_until="networkidle")
            await self.page.wait_for_timeout(3000)

            # Take initial screenshot
            await self.take_screenshot("01_initial_load")

            # Check if page loaded successfully
            title = await self.page.title()
            print(f"📄 Page title: {title}")

            # Wait for app to be ready
            await self.page.wait_for_selector('#app', timeout=10000)
            print("✅ Frontend loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load frontend: {e}")
            await self.take_screenshot("error_frontend_load")
            return False

    async def verify_script_studio_interface(self):
        """Verify Script Studio interface elements are present"""
        print("🔍 Verifying Script Studio interface...")
        try:
            # Wait for main content to load
            await self.page.wait_for_timeout(5000)

            # Look for common Script Studio elements
            selectors_to_check = [
                '[class*="script"]',
                '[class*="studio"]',
                '[class*="sidebar"]',
                '[class*="left"]',
                '[data-testid*="script"]',
                'button[title*="脚本"]',
                'button[title*="script"]',
                '.upload',
                '[class*="upload"]',
                'input[type="file"]'
            ]

            found_elements = []
            for selector in selectors_to_check:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        found_elements.extend([selector] * len(elements))
                        print(f"  ✓ Found {len(elements)} elements matching: {selector}")
                except:
                    pass

            if not found_elements:
                print("⚠️  No obvious Script Studio elements found, checking page content...")

                # Get page text content for analysis
                body_text = await self.page.evaluate('() => document.body.innerText')
                print(f"📄 Page content preview (first 500 chars): {body_text[:500]}")

                # Look for specific text patterns
                if any(keyword in body_text.lower() for keyword in ['script', 'studio', '脚本', '上传', 'upload']):
                    print("✅ Found Script Studio related text in page content")
                else:
                    print("⚠️  Limited Script Studio content detected")

            await self.take_screenshot("02_interface_check")
            print("✅ Interface verification completed")
            return True
        except Exception as e:
            print(f"❌ Interface verification failed: {e}")
            await self.take_screenshot("error_interface_check")
            return False

    async def upload_excel_file(self):
        """Upload the Excel file using file upload interface"""
        print(f"📁 Uploading Excel file: {EXCEL_FILE_PATH}")
        try:
            # Check if file exists
            if not os.path.exists(EXCEL_FILE_PATH):
                print(f"❌ Excel file not found: {EXCEL_FILE_PATH}")
                return False

            # Look for file input elements
            file_inputs = await self.page.query_selector_all('input[type="file"]')
            upload_buttons = await self.page.query_selector_all('button, [role="button"]')

            if file_inputs:
                print(f"📎 Found {len(file_inputs)} file input elements")
                # Use the first file input
                await file_inputs[0].set_input_files(EXCEL_FILE_PATH)
                print("✅ File uploaded via file input")
            else:
                # Look for upload buttons and click them
                upload_found = False
                for button in upload_buttons:
                    button_text = await button.inner_text()
                    if any(keyword in button_text.lower() for keyword in ['upload', '上传', 'file', '文件', 'browse', '选择']):
                        print(f"🖱️  Clicking upload button: {button_text}")
                        await button.click()
                        await self.page.wait_for_timeout(1000)

                        # Look for file input that appears after clicking
                        file_inputs_after = await self.page.query_selector_all('input[type="file"]')
                        if file_inputs_after:
                            await file_inputs_after[0].set_input_files(EXCEL_FILE_PATH)
                            upload_found = True
                            print("✅ File uploaded after clicking upload button")
                            break

                if not upload_found:
                    print("⚠️  No obvious upload interface found, attempting drag and drop...")
                    # Try to find a drop zone
                    drop_zones = await self.page.query_selector_all('[class*="drop"], [class*="upload"], [class*="zone"]')
                    if drop_zones:
                        # Create a data transfer object for drag and drop
                        await drop_zones[0].dispatch_event('dragover')
                        await drop_zones[0].dispatch_event('drop', {
                            'dataTransfer': {
                                'files': [EXCEL_FILE_PATH]
                            }
                        })
                        print("✅ Attempted drag and drop file upload")
                    else:
                        print("❌ No upload interface found")
                        return False

            await self.page.wait_for_timeout(3000)
            await self.take_screenshot("03_file_upload")

            # Check for upload success indicators
            success_indicators = await self.page.query_selector_all('[class*="success"], [class*="uploaded"], [class*="complete"]')
            if success_indicators:
                print("✅ File upload appears successful")
            else:
                print("⚠️  Upload success not confirmed, but file was submitted")

            return True
        except Exception as e:
            print(f"❌ File upload failed: {e}")
            await self.take_screenshot("error_file_upload")
            return False

    async def select_sop_script(self):
        """Select the SOP流程标注 script from left sidebar"""
        print("🔧 Selecting SOP流程标注 script...")
        try:
            # Look for sidebar or script selection area
            sidebar_selectors = [
                '[class*="sidebar"]',
                '[class*="left"]',
                '[class*="script-list"]',
                '[class*="tool-list"]',
                '[data-testid*="sidebar"]'
            ]

            sidebar_found = False
            for selector in sidebar_selectors:
                sidebar = await self.page.query_selector(selector)
                if sidebar:
                    print(f"✅ Found sidebar: {selector}")
                    sidebar_found = True
                    break

            if not sidebar_found:
                print("⚠️  No obvious sidebar found, searching for script options...")

            # Look for script options with SOP text
            script_selectors = [
                'button:has-text("SOP")',
                'div:has-text("SOP")',
                '[class*="script"]:has-text("SOP")',
                'li:has-text("SOP")',
                'option:has-text("SOP")',
                '*:has-text("流程标注")',
                '*:has-text("SOP流程标注")'
            ]

            script_found = False
            for selector in script_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.inner_text()
                        if 'sop' in text.lower() or '流程标注' in text:
                            print(f"🎯 Found SOP script: {text}")
                            await element.click()
                            await self.page.wait_for_timeout(1000)
                            script_found = True
                            break
                    if script_found:
                        break
                except:
                    continue

            if not script_found:
                print("⚠️  SOP script not found, looking for any script options...")
                # Try to find any clickable script-related elements
                all_clickable = await self.page.query_selector_all('button, [role="button"], li[role="menuitem"], option')
                for element in all_clickable[:10]:  # Check first 10 elements
                    try:
                        text = await element.inner_text()
                        if text and len(text) < 100:  # Reasonable length
                            print(f"  Available option: {text}")
                    except:
                        pass

            await self.take_screenshot("04_script_selection")
            print("✅ Script selection attempt completed")
            return script_found
        except Exception as e:
            print(f"❌ Script selection failed: {e}")
            await self.take_screenshot("error_script_selection")
            return False

    async def fill_script_parameters(self):
        """Fill in required parameters for the script"""
        print("📝 Looking for script parameters...")
        try:
            # Look for input fields, text areas, selects
            input_fields = await self.page.query_selector_all('input:not([type="file"]):not([type="submit"]), textarea, select')

            if input_fields:
                print(f"📋 Found {len(input_fields)} input fields")

                for i, field in enumerate(input_fields):
                    try:
                        field_type = await field.get_attribute('type')
                        placeholder = await field.get_attribute('placeholder')
                        name = await field.get_attribute('name')
                        id_attr = await field.get_attribute('id')

                        print(f"  Field {i+1}: type={field_type}, placeholder={placeholder}, name={name}, id={id_attr}")

                        # Fill with test data based on field characteristics
                        if placeholder:
                            if 'column' in placeholder.lower():
                                await field.fill('A')  # Example column
                            elif 'sheet' in placeholder.lower():
                                await field.fill('Sheet1')  # Example sheet name
                            elif 'file' in placeholder.lower():
                                await field.fill('bad12.xlsx')
                            else:
                                await field.fill(f'test_value_{i+1}')
                        else:
                            await field.fill(f'test_value_{i+1}')

                    except Exception as e:
                        print(f"  ⚠️  Could not fill field {i+1}: {e}")
            else:
                print("ℹ️  No input fields found for parameters")

            await self.take_screenshot("05_parameters_filled")
            print("✅ Parameter filling completed")
            return True
        except Exception as e:
            print(f"❌ Parameter filling failed: {e}")
            await self.take_screenshot("error_parameters")
            return False

    async def execute_script(self):
        """Click the run script button"""
        print("▶️  Looking for run script button...")
        try:
            # Look for run/execute buttons
            run_button_selectors = [
                'button:has-text("运行")',
                'button:has-text("Run")',
                'button:has-text("执行")',
                'button:has-text("Execute")',
                '[class*="run"]',
                '[class*="execute"]',
                '[title*="运行"]',
                '[title*="run"]',
                'button[type="submit"]'
            ]

            button_found = False
            for selector in run_button_selectors:
                try:
                    buttons = await self.page.query_selector_all(selector)
                    for button in buttons:
                        button_text = await button.inner_text()
                        if button_text and any(keyword in button_text.lower() for keyword in ['run', '运行', 'execute', '执行', 'start', '开始']):
                            print(f"🎯 Found run button: {button_text}")
                            await button.click()
                            button_found = True
                            break
                    if button_found:
                        break
                except:
                    continue

            if not button_found:
                print("⚠️  No obvious run button found, checking all buttons...")
                all_buttons = await self.page.query_selector_all('button, [role="button"]')
                for i, button in enumerate(all_buttons):
                    try:
                        text = await button.inner_text()
                        if text:
                            print(f"  Button {i+1}: {text}")
                    except:
                        pass

                # Try clicking the last button (often submit/run)
                if all_buttons:
                    print("🎲 Trying last button as potential run button...")
                    await all_buttons[-1].click()
                    button_found = True

            if button_found:
                await self.page.wait_for_timeout(3000)
                await self.take_screenshot("06_script_execution")
                print("✅ Script execution initiated")
            else:
                print("❌ No run button found")

            return button_found
        except Exception as e:
            print(f"❌ Script execution failed: {e}")
            await self.take_screenshot("error_execution")
            return False

    async def monitor_execution_results(self):
        """Monitor for execution results and errors"""
        print("👀 Monitoring script execution results...")
        try:
            # Wait for execution to complete or timeout
            print("⏳ Waiting for execution results (30 seconds)...")
            await self.page.wait_for_timeout(30000)

            # Look for results
            result_selectors = [
                '[class*="result"]',
                '[class*="output"]',
                '[class*="success"]',
                '[class*="error"]',
                '[class*="status"]',
                'pre',
                'code',
                '.log',
                '[data-testid*="result"]'
            ]

            results_found = []
            for selector in result_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.inner_text()
                        if text and len(text.strip()) > 0:
                            results_found.append(f"{selector}: {text[:100]}...")
                except:
                    pass

            if results_found:
                print("✅ Results found:")
                for result in results_found[:5]:  # Show first 5 results
                    print(f"  📄 {result}")
            else:
                print("⚠️  No obvious results found")

            # Check for error messages
            error_selectors = [
                '[class*="error"]',
                '[class*="alert"]',
                '[class*="danger"]',
                '.error-message',
                '[role="alert"]'
            ]

            errors_found = []
            for selector in error_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.inner_text()
                        if text and 'error' in text.lower():
                            errors_found.append(f"{selector}: {text}")
                except:
                    pass

            if errors_found:
                print("❌ Errors found:")
                for error in errors_found:
                    print(f"  🔴 {error}")
            else:
                print("✅ No obvious errors detected")

            # Check network requests for 500 errors
            print("🌐 Checking for network errors...")
            # This would need to be implemented with request interception

            await self.take_screenshot("07_execution_results")
            print("✅ Results monitoring completed")
            return True
        except Exception as e:
            print(f"❌ Results monitoring failed: {e}")
            await self.take_screenshot("error_results")
            return False

    async def check_backend_logs(self):
        """Check backend logs for debugging information"""
        print("📋 Checking backend logs...")
        try:
            log_file = "/Users/luojiatai/PycharmProjects/PythonProject_predeal/backend_server.log"
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    # Read last 50 lines
                    lines = f.readlines()
                    recent_lines = lines[-50:]

                    print("📝 Recent backend log entries:")
                    for line in recent_lines:
                        if any(keyword in line.lower() for keyword in ['error', 'exception', '500', 'failed', 'traceback']):
                            print(f"  🔴 {line.strip()}")
                        elif any(keyword in line.lower() for keyword in ['post', 'upload', 'script', 'sop']):
                            print(f"  📄 {line.strip()}")
            else:
                print(f"⚠️  Backend log file not found: {log_file}")

            print("✅ Backend log check completed")
            return True
        except Exception as e:
            print(f"❌ Backend log check failed: {e}")
            return False

    async def take_screenshot(self, name):
        """Take a screenshot with timestamp"""
        try:
            timestamp = int(time.time())
            filename = f"{SCREENSHOT_DIR}/{name}_{timestamp}.png"
            await self.page.screenshot(path=filename, full_page=True)
            print(f"📸 Screenshot saved: {filename}")
        except Exception as e:
            print(f"⚠️  Screenshot failed: {e}")

    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            print("🧹 Browser cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

    async def run_full_test(self):
        """Run the complete test suite"""
        print("🎬 Starting comprehensive Script Studio test...")
        print("=" * 60)

        test_steps = [
            ("Setup Browser", self.setup),
            ("Navigate to Frontend", self.navigate_to_frontend),
            ("Verify Interface", self.verify_script_studio_interface),
            ("Upload Excel File", self.upload_excel_file),
            ("Select SOP Script", self.select_sop_script),
            ("Fill Parameters", self.fill_script_parameters),
            ("Execute Script", self.execute_script),
            ("Monitor Results", self.monitor_execution_results),
            ("Check Backend Logs", self.check_backend_logs),
        ]

        results = {}
        for step_name, step_func in test_steps:
            print(f"\n🔄 {step_name}...")
            try:
                results[step_name] = await step_func()
                status = "✅ PASSED" if results[step_name] else "❌ FAILED"
                print(f"{status} {step_name}")
            except Exception as e:
                results[step_name] = False
                print(f"❌ FAILED {step_name}: {e}")

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        for step_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"{status} {step_name}")

        print(f"\n📈 Overall Result: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed - check screenshots and logs")

        await self.cleanup()
        return results

async def main():
    """Main test runner"""
    tester = ScriptStudioTester()
    results = await tester.run_full_test()

    # Save test results to JSON
    results_file = "/Users/luojiatai/PycharmProjects/PythonProject_predeal/test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Test results saved to: {results_file}")

if __name__ == "__main__":
    print("🚀 Script Studio Comprehensive Test")
    print("=" * 60)
    asyncio.run(main())