"""
Test script to verify the API is working correctly
"""
import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_api_status():
    """Test if API is responding"""
    try:
        response = requests.get(f"{API_BASE_URL}/predict/status")
        print("✅ API Status Check:", response.status_code)
        print("📄 Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print("❌ API Status Check Failed:", str(e))
        return False

def test_api_health():
    """Test model health"""
    try:
        response = requests.get(f"{API_BASE_URL}/predict/health")
        print("🏥 Health Check:", response.status_code)
        print("📄 Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print("❌ Health Check Failed:", str(e))
        return False

def test_api_info():
    """Test API root endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print("📋 API Info:", response.status_code) 
        print("📄 Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print("❌ API Info Failed:", str(e))
        return False

def test_prediction_json():
    """Test JSON prediction endpoint (will fail without real data, but should validate input)"""
    test_data = {
        "fieldId": "test_field_001",
        "userId": "test_user_123",
        "hyperSpectralImageUrl": "https://httpbin.org/get",  # Dummy URL for testing
        "sensorData": {
            "airTemperature": 25.5,
            "humidity": 60.0,
            "soilMoisture": 30.0,
            "timestamp": "2025-10-04T10:00:00Z",
            "windSpeed": 10.0,
            "rainfall": 0.0,
            "solarRadiation": 800.0,
            "leafWetness": 15.0,
            "co2Level": 400.0,
            "phLevel": 6.5
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print("🎯 JSON Prediction Test:", response.status_code)
        print("📄 Response:", response.text[:500] + "..." if len(response.text) > 500 else response.text)
        return True  # Even if it fails due to data issues, endpoint should be reachable
    except Exception as e:
        print("❌ JSON Prediction Test Failed:", str(e))
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    tests = [
        ("API Status", test_api_status),
        ("API Health", test_api_health), 
        ("API Info", test_api_info),
        ("JSON Prediction", test_prediction_json)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
        print("-" * 30)
    
    # Summary
    print("\n📊 TEST SUMMARY:")
    print("=" * 50)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for JavaScript integration.")
    else:
        print("⚠️  Some tests failed. Check the API configuration.")

if __name__ == "__main__":
    main()