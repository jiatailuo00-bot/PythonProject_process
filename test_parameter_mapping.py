#!/usr/bin/env python3
"""
Quick test to verify script parameter mapping is working
"""
import requests
import json

def test_parameter_mapping():
    base_url = "http://localhost:8000"

    print("🔧 Testing parameter mapping...")

    # Test 1: Check if scripts are available
    print("\n1. Testing scripts API...")
    response = requests.get(f"{base_url}/api/scripts")
    if response.status_code == 200:
        scripts = response.json()
        print(f"✅ Found {len(scripts)} scripts")
        for script in scripts:
            print(f"   - {script['name']} (ID: {script['id']})")
            params = [p['name'] for p in script['parameters']]
            print(f"     Parameters: {params}")
    else:
        print(f"❌ Scripts API failed: {response.status_code}")
        return False

    # Test 2: Upload a test file
    print("\n2. Testing file upload...")
    file_path = "../bad12.xlsx"
    try:
        with open(file_path, 'rb') as f:
            files = {'file': ('bad12.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post(f"{base_url}/api/upload/single", files=files)

        if response.status_code == 200:
            upload_result = response.json()
            print(f"✅ File uploaded successfully: {upload_result['path']}")
            uploaded_path = upload_result['path']
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except FileNotFoundError:
        print(f"❌ Test file not found: {file_path}")
        return False

    # Test 3: Test each script with proper parameters
    print("\n3. Testing script execution with parameters...")

    test_cases = [
        {
            "script_id": "run_sop_pipeline",
            "params": {
                "corpus_path": uploaded_path,
                "output_filename": "test_sop_result.xlsx"
            }
        },
        {
            "script_id": "update_latest_customer_message",
            "params": {
                "excel_path": uploaded_path,
                "context_column": "最终传参上下文",
                "latest_customer_column": "最新客户消息",
                "output_filename": "test_updated.xlsx"
            }
        },
        {
            "script_id": "process_waxu_badcase",
            "params": {
                "input_file": uploaded_path,
                "output_file": "test_badcase_result.xlsx"
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n   Testing {test_case['script_id']}...")
        print(f"   Parameters: {test_case['params']}")

        response = requests.post(
            f"{base_url}/api/scripts/{test_case['script_id']}/run",
            json=test_case['params']
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result.get('message', 'No message')}")
            print(f"   Success status: {result.get('success', False)}")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            print(f"   Response: {response.text}")

    print("\n🎯 Test completed!")
    return True

if __name__ == "__main__":
    test_parameter_mapping()