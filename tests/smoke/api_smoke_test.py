import requests
import time
import sys

def test_api():
    base_url = "http://127.0.0.1:8000"
    max_retries = 5
    retry_delay = 3
    
    print(f"🔍 Testing Health Endpoint at {base_url} (with retries)...")
    connected = False
    for i in range(max_retries):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Health: OK - {resp.json()}")
                connected = True
                break
        except Exception:
            print(f"⏳ Attempt {i+1}/{max_retries}: Server not ready yet, waiting {retry_delay}s...")
            time.sleep(retry_delay)

    if not connected:
        print(f"❌ API not reachable after {max_retries} attempts. Make sure 'make api' is running and active.")
        sys.exit(1)

    print("\n📊 Testing Metrics Endpoint...")
    # ... (rest of the tests)
    resp = requests.get(f"{base_url}/metrics")
    if resp.status_code == 200:
        print(f"Metrics: OK (Content Type: {resp.headers.get('Content-Type')})")
        if "prediction_requests_total" in resp.text:
            print("✅ Prometheus metrics found!")
    else:
        print(f"❌ Metrics failed: {resp.status_code}")

    print("\n🤖 Testing Prediction Endpoint (Mock data)...")
    payload = {
        "node_id": 12345,
        "features": [0.1] * 165,
        "neighbors": [678, 910]
    }
    resp = requests.post(f"{base_url}/predict", json=payload)
    if resp.status_code == 200:
        print(f"Prediction: {resp.json()}")
        print("✅ End-to-end inference successful!")
    else:
        print(f"❌ Prediction failed: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    test_api()
