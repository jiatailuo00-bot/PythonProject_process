#!/usr/bin/env python3
"""
快速API测试 - 验证基础功能
"""

import requests
import json

def test_api():
    """测试基础API功能"""
    print("🚀 快速API测试...")

    base_url = "http://localhost:8000"

    try:
        # 1. 检查健康状态
        print("1️⃣ 健康检查...")
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ 服务健康")
        else:
            print(f"   ❌ 健康检查失败: {response.status_code}")
            return

        # 2. 获取脚本列表
        print("\n2️⃣ 获取脚本列表...")
        response = requests.get(f"{base_url}/api/scripts", timeout=5)
        if response.status_code == 200:
            scripts = response.json()
            print(f"   发现 {len(scripts)} 个脚本")
            for script in scripts:
                print(f"   - {script['name']} ({script['id']})")
        else:
            print(f"   ❌ 获取脚本列表失败: {response.status_code}")
            return

        # 3. 获取文件列表
        print("\n3️⃣ 获取文件列表...")
        response = requests.get(f"{base_url}/api/upload/list", timeout=5)
        if response.status_code == 200:
            files_data = response.json()
            files = files_data['files']
            print(f"   发现 {len(files)} 个文件")
        else:
            print(f"   ❌ 获取文件列表失败: {response.status_code}")
            return

        # 4. 测试错误处理
        print("\n4️⃣ 测试错误处理...")
        response = requests.post(
            f"{base_url}/api/scripts/nonexistent/run",
            json={"params": {}},
            timeout=5
        )
        if response.status_code == 404:
            print("   ✅ 404错误处理正常")
        else:
            print(f"   ❌ 错误处理异常: {response.status_code}")

        print("\n🎉 基础API测试完成!")

    except requests.exceptions.ConnectionError:
        print("\n❌ 连接错误: 请确保后端服务运行在 http://localhost:8000")
    except requests.exceptions.Timeout:
        print("\n❌ 请求超时")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    test_api()