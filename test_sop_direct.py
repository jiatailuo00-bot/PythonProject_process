#!/usr/bin/env python3
"""
直接使用API测试SOP脚本功能
绕过前端界面，直接调用API进行测试
"""

import asyncio
import json
import time
from pathlib import Path
import httpx

BASE_URL = "http://localhost:8000"
BAD12_FILE = Path("./bad12.xlsx").absolute()

async def test_sop_via_api():
    """
    直接通过API测试SOP脚本功能
    1. 检查API服务状态
    2. 上传bad12.xlsx文件
    3. 调用SOP脚本
    4. 验证结果
    """

    print("🚀 开始SOP API测试...")

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # 步骤1：检查API服务状态
            print("📍 步骤1: 检查API服务状态")
            response = await client.get(f"{BASE_URL}/api/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API服务正常: {health_data}")
            else:
                raise Exception(f"API服务异常: {response.status_code}")

            # 步骤2：获取脚本列表
            print("📍 步骤2: 获取脚本列表")
            response = await client.get(f"{BASE_URL}/api/scripts")
            if response.status_code != 200:
                raise Exception(f"获取脚本列表失败: {response.status_code}")

            scripts = response.json()
            sop_script = None
            for script in scripts:
                if script["id"] == "run_sop_pipeline":
                    sop_script = script
                    break

            if not sop_script:
                raise Exception("未找到SOP流程标注脚本")

            print(f"✅ 找到SOP脚本: {sop_script['name']}")

            # 步骤3：上传bad12.xlsx文件
            print(f"📍 步骤3: 上传文件 {BAD12_FILE}")

            if not BAD12_FILE.exists():
                raise Exception(f"文件不存在: {BAD12_FILE}")

            with open(BAD12_FILE, "rb") as f:
                files = {"file": (BAD12_FILE.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                response = await client.post(f"{BASE_URL}/api/upload/single", files=files)

            if response.status_code != 200:
                raise Exception(f"文件上传失败: {response.status_code} - {response.text}")

            upload_result = response.json()
            uploaded_file_path = upload_result["path"]
            print(f"✅ 文件上传成功: {uploaded_file_path}")

            # 步骤4：运行SOP脚本
            print("📍 步骤4: 运行SOP脚本")

            # 准备脚本参数
            script_params = {
                "corpus_path": uploaded_file_path,
                "output_dir": str(BAD12_FILE.parent),
                "output_filename": f"{BAD12_FILE.stem}_sop_result_{int(time.time())}.xlsx",
                "similarity": 0.9,
                "batch_size": 100
            }

            print(f"📋 脚本参数: {json.dumps(script_params, indent=2, ensure_ascii=False)}")

            # 调用脚本执行API
            response = await client.post(
                f"{BASE_URL}/api/scripts/run_sop_pipeline/run",
                json={"params": script_params}
            )

            if response.status_code != 200:
                error_info = response.text
                print(f"❌ 脚本执行失败: {response.status_code}")
                print(f"错误详情: {error_info}")
                return False

            result = response.json()
            print("✅ 脚本执行完成")

            # 步骤5：验证执行结果
            print("📍 步骤5: 验证执行结果")

            print(f"📊 执行结果:")
            print(f"  成功: {result.get('success', False)}")
            print(f"  消息: {result.get('message', '')}")
            print(f"  数据: {json.dumps(result.get('data', {}), indent=2, ensure_ascii=False)}")

            if result.get("success"):
                print("✅ SOP脚本执行成功!")

                # 检查输出文件
                output_file = result.get("data", {}).get("output_file")
                if output_file and Path(output_file).exists():
                    file_size = Path(output_file).stat().st_size
                    print(f"📁 输出文件: {output_file} ({file_size} bytes)")
                else:
                    print("⚠️ 输出文件未找到或为空")

                # 显示日志摘要
                logs = result.get("logs", "")
                if logs:
                    log_lines = logs.split('\n')
                    print(f"📝 日志摘要 ({len(log_lines)} 行):")
                    for i, line in enumerate(log_lines[-10:], 1):  # 显示最后10行
                        print(f"  {i:2d}: {line}")

                return True
            else:
                print("❌ SOP脚本执行失败")
                return False

        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            return False

async def main():
    """主函数"""
    print("🧪 SOP脚本API测试")
    print("=" * 50)

    success = await test_sop_via_api()

    if success:
        print("\n✅ 所有测试通过")
        exit(0)
    else:
        print("\n❌ 测试失败")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())