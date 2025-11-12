#!/usr/bin/env python3
"""
简化的SOP脚本测试
直接使用API测试SOP功能，绕过复杂的UI自动化
"""

import asyncio
import json
import sys
from pathlib import Path
import httpx

BASE_URL = "http://localhost:8000"
BAD12_FILE = Path("./bad12.xlsx").absolute()

async def test_sop_simple():
    """简化的SOP测试"""
    print("🚀 开始SOP简化测试...")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1. 检查服务状态
            print("📍 检查API服务...")
            response = await client.get(f"{BASE_URL}/api/health")
            if response.status_code != 200:
                print(f"❌ API服务异常: {response.status_code}")
                return False
            print("✅ API服务正常")

            # 2. 上传文件
            print(f"📍 上传文件 {BAD12_FILE}...")
            if not BAD12_FILE.exists():
                print(f"❌ 文件不存在: {BAD12_FILE}")
                return False

            with open(BAD12_FILE, "rb") as f:
                files = {"file": (BAD12_FILE.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                response = await client.post(f"{BASE_URL}/api/upload/single", files=files)

            if response.status_code != 200:
                print(f"❌ 文件上传失败: {response.status_code}")
                print(f"响应: {response.text}")
                return False

            upload_result = response.json()
            file_path = upload_result["path"]
            print(f"✅ 文件上传成功: {file_path}")

            # 3. 运行SOP脚本（简化参数）
            print("📍 运行SOP脚本...")

            # 使用最小参数配置
            script_params = {
                "corpus_path": file_path,
                "similarity": 0.9
            }

            response = await client.post(
                f"{BASE_URL}/api/scripts/run_sop_pipeline/run",
                json={"params": script_params}
            )

            print(f"📊 响应状态: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 脚本执行完成")
                print(f"成功: {result.get('success', False)}")
                print(f"消息: {result.get('message', '')}")

                if result.get('data'):
                    print(f"输出数据: {json.dumps(result['data'], indent=2, ensure_ascii=False)}")

                if result.get('logs'):
                    logs = result['logs']
                    print(f"日志长度: {len(logs)} 字符")
                    # 显示最后几行日志
                    log_lines = logs.split('\n')[-5:]
                    for line in log_lines:
                        if line.strip():
                            print(f"  日志: {line}")

                return result.get('success', False)
            else:
                print(f"❌ 脚本执行失败: {response.status_code}")
                print(f"错误响应: {response.text}")
                return False

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("🧪 SOP脚本简化测试")
    print("=" * 40)

    success = await test_sop_simple()

    if success:
        print("\n✅ SOP功能测试成功")
    else:
        print("\n❌ SOP功能测试失败")

if __name__ == "__main__":
    asyncio.run(main())