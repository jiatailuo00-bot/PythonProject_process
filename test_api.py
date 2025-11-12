#!/usr/bin/env python3
"""
API测试脚本 - 测试脚本执行功能
"""

import asyncio
import json
from playwright.async_api import async_playwright

async def test_api():
    """测试脚本执行API"""
    print("🚀 开始API测试...")

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()

    try:
        # 1. 打开网页
        print("1️⃣ 打开网页...")
        await page.goto("http://localhost:5174")
        await page.wait_for_load_state("networkidle")
        print("   ✅ 网页加载完成")

        # 2. 检查脚本列表
        print("\n2️⃣ 获取脚本列表...")
        scripts_response = await page.evaluate("""
            async () => {
                const response = await fetch('/api/scripts');
                return await response.json();
            }
        """)

        print(f"   发现 {len(scripts_response)} 个脚本:")
        for script in scripts_response:
            print(f"   - {script['name']} ({script['id']})")

        # 3. 尝试执行一个简单的脚本
        print("\n3️⃣ 测试脚本执行...")

        # 首先检查文件列表
        files_response = await page.evaluate("""
            async () => {
                const response = await fetch('/api/upload/list');
                return await response.json();
            }
        """)

        if files_response['files']:
            test_file = files_response['files'][0]
            print(f"   使用测试文件: {test_file['filename']}")

            # 尝试执行"同步最新客户消息"脚本
            script_id = "update_latest_customer_message"
            payload = {
                "params": {
                    "excel_path": test_file['path'],
                    "context_column": "最终传参上下文",
                    "latest_customer_column": "最新客户消息"
                }
            }

            print(f"   执行脚本: {script_id}")
            print(f"   参数: {json.dumps(payload['params'], indent=2, ensure_ascii=False)}")

            try:
                execution_response = await page.evaluate(f"""
                    async () => {{
                        const scriptId = '{script_id}';
                        const payload = {json.dumps(payload)};

                        const response = await fetch(`/api/scripts/${{scriptId}}/run`, {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json',
                            }},
                            body: JSON.stringify(payload)
                        }});

                        const result = await response.json();
                        return {{
                            status: response.status,
                            ok: response.ok,
                            result: result
                        }};
                    }}
                """)

                print(f"   响应状态: {execution_response['status']}")
                print(f"   响应成功: {execution_response['ok']}")

                if execution_response['ok']:
                    print("   ✅ 脚本执行成功!")
                    print(f"   结果: {json.dumps(execution_response['result'], indent=2, ensure_ascii=False)}")
                else:
                    print("   ❌ 脚本执行失败!")
                    print(f"   错误: {execution_response['result']}")

            except Exception as e:
                print(f"   ❌ API调用异常: {e}")
        else:
            print("   ⚠️ 没有可用的测试文件，无法测试脚本执行")

        # 4. 测试错误处理
        print("\n4️⃣ 测试错误处理...")
        try:
            error_response = await page.evaluate("""
                async () => {
                    const response = await fetch('/api/scripts/nonexistent/run', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({params: {}})
                    });

                    return {
                        status: response.status,
                        result: await response.text()
                    };
                }
            """)

            print(f"   不存在脚本的响应状态: {error_response['status']}")
            if error_response['status'] == 404:
                print("   ✅ 错误处理正常")
            else:
                print("   ❌ 错误处理异常")

        except Exception as e:
            print(f"   ❌ 错误处理测试异常: {e}")

        print("\n🎉 API测试完成!")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    finally:
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_api())