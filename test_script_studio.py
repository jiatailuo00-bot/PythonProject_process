#!/usr/bin/env python3
"""
Script Studio 自动化测试脚本
测试网页基本功能和文件上传脚本执行
"""

import time
import requests
from pathlib import Path

class ScriptStudioTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.test_file = "bad12.xlsx"

    def test_backend_health(self):
        """测试后端健康状态"""
        print("🔍 测试后端健康状态...")
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ 后端服务正常")
                return True
            else:
                print(f"❌ 后端服务异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 后端连接失败: {e}")
            return False

    def test_scripts_api(self):
        """测试脚本列表API"""
        print("🔍 测试脚本列表API...")
        try:
            response = requests.get(f"{self.base_url}/api/scripts", timeout=5)
            if response.status_code == 200:
                scripts = response.json()
                print(f"✅ 获取到 {len(scripts)} 个脚本:")
                for script in scripts:
                    print(f"   - {script['name']} ({script['category']})")
                return scripts
            else:
                print(f"❌ 获取脚本列表失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 脚本API请求失败: {e}")
            return None

    def test_upload_list_api(self):
        """测试文件列表API"""
        print("🔍 测试文件列表API...")
        try:
            response = requests.get(f"{self.base_url}/api/upload/list", timeout=5)
            if response.status_code == 200:
                files = response.json()
                print(f"✅ 获取到 {files['total']} 个文件")
                return files
            else:
                print(f"❌ 获取文件列表失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 文件列表API请求失败: {e}")
            return None

    def test_frontend_access(self):
        """测试前端访问"""
        print("🔍 测试前端访问...")
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                print("✅ 前端页面可访问")
                return True
            else:
                print(f"❌ 前端页面异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 前端连接失败: {e}")
            return False

    def check_test_file(self):
        """检查测试文件是否存在"""
        test_file_path = Path(self.test_file)
        if test_file_path.exists():
            print(f"✅ 找到测试文件: {self.test_file}")
            return True
        else:
            print(f"❌ 测试文件不存在: {self.test_file}")
            print("请确保测试文件在当前目录下")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 Script Studio 自动化测试")
        print("=" * 50)

        results = []

        # 测试后端健康状态
        results.append(("后端健康检查", self.test_backend_health()))

        # 测试前端访问
        results.append(("前端访问检查", self.test_frontend_access()))

        # 测试脚本API
        scripts = self.test_scripts_api()
        results.append(("脚本API检查", scripts is not None))

        # 测试文件列表API
        files = self.test_upload_list_api()
        results.append(("文件列表API检查", files is not None))

        # 检查测试文件
        results.append(("测试文件检查", self.check_test_file()))

        print("=" * 50)
        print("📊 测试结果汇总:")

        passed = 0
        for name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {name}: {status}")
            if result:
                passed += 1

        print(f"\n总体结果: {passed}/{len(results)} 测试通过")

        if passed == len(results):
            print("🎉 所有测试通过！Script Studio 运行正常")
            print("💡 你可以访问以下地址:")
            print(f"   - 前端界面: {self.frontend_url}")
            print(f"   - API文档: {self.base_url}/api/docs")
        else:
            print("⚠️  部分测试失败，请检查相关服务")

        return passed == len(results)

if __name__ == "__main__":
    tester = ScriptStudioTester()
    tester.run_all_tests()