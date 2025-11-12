#!/usr/bin/env python3
"""
简单API测试脚本 - 直接测试后端API
"""

import asyncio
import json
import aiohttp

async def test_api_direct():
    """直接测试后端API"""
    print("🚀 开始直接API测试...")

    base_url = "http://localhost:8000"

    async with aiohttp.ClientSession() as session:
        try:
            # 1. 检查健康状态
            print("1️⃣ 检查健康状态...")
            async with session.get(f"{base_url}/api/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"   ✅ 服务健康: {health}")
                else:
                    print(f"   ❌ 健康检查失败: {response.status}")
                    return

            # 2. 获取脚本列表
            print("\n2️⃣ 获取脚本列表...")
            async with session.get(f"{base_url}/api/scripts") as response:
                if response.status == 200:
                    scripts = await response.json()
                    print(f"   发现 {len(scripts)} 个脚本:")
                    for script in scripts:
                        print(f"   - {script['name']} ({script['id']}) - {script['category']}")
                else:
                    print(f"   ❌ 获取脚本列表失败: {response.status}")
                    return

            # 3. 获取文件列表
            print("\n3️⃣ 获取文件列表...")
            async with session.get(f"{base_url}/api/upload/list") as response:
                if response.status == 200:
                    files_data = await response.json()
                    files = files_data['files']
                    print(f"   发现 {len(files)} 个文件:")
                    for file in files[:3]:  # 只显示前3个
                        print(f"   - {file['filename']} ({file['size']} bytes)")
                else:
                    print(f"   ❌ 获取文件列表失败: {response.status}")
                    return

            # 4. 测试简单脚本执行
            if files:
                print("\n4️⃣ 测试脚本执行...")
                test_file = files[0]
                script_id = "update_latest_customer_message"

                print(f"   使用文件: {test_file['filename']}")
                print(f"   执行脚本: {script_id}")

                payload = {
                    "params": {
                        "excel_path": test_file['path'],
                        "context_column": "最终传参上下文",
                        "latest_customer_column": "最新客户消息"
                    }
                }

                print(f"   参数: {json.dumps(payload['params'], indent=2, ensure_ascii=False)}")

                try:
                    async with session.post(
                        f"{base_url}/api/scripts/{script_id}/run",
                        json=payload
                    ) as response:
                        print(f"   响应状态: {response.status}")

                        if response.status == 200:
                            result = await response.json()
                            print("   ✅ 脚本执行成功!")
                            print(f"   消息: {result.get('message', 'N/A')}")
                            if result.get('data'):
                                print(f"   数据: {json.dumps(result['data'], indent=2, ensure_ascii=False)}")
                        else:
                            error_text = await response.text()
                            print(f"   ❌ 脚本执行失败!")
                            print(f"   错误: {error_text}")

                except Exception as e:
                    print(f"   ❌ 请求异常: {e}")
            else:
                print("\n4️⃣ ⚠️ 没有可用的测试文件，跳过脚本执行测试")

            # 5. 测试错误处理
            print("\n5️⃣ 测试错误处理...")
            try:
                async with session.post(
                    f"{base_url}/api/scripts/nonexistent/run",
                    json={"params": {}}
                ) as response:
                    print(f"   不存在脚本的响应状态: {response.status}")
                    if response.status == 404:
                        print("   ✅ 404错误处理正常")
                    else:
                        error_text = await response.text()
                        print(f"   ❌ 错误处理异常: {error_text}")
            except Exception as e:
                print(f"   ❌ 错误处理测试异常: {e}")

            print("\n🎉 API测试完成!")

        except aiohttp.ClientError as e:
            print(f"\n❌ 网络连接错误: {e}")
            print("请确保后端服务运行在 http://localhost:8000")
        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_direct())