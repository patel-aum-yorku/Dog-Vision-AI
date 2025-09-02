#!/usr/bin/env python3
"""
Simple test script to verify the backend service works correctly.
Run this after installing dependencies to test the full functionality.
"""

import os
import sys
import time
import requests
from pathlib import Path


def test_health_endpoint():
    """Test the health check endpoint."""
    print("🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on localhost:8000?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_root_endpoint():
    """Test the root endpoint."""
    print("\n🏠 Testing root endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint works")
            print(f"   Message: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False


def test_predict_endpoint_without_file():
    """Test the predict endpoint without a file (should fail gracefully)."""
    print("\n📷 Testing predict endpoint (without file)...")
    
    try:
        response = requests.post("http://localhost:8000/predict", timeout=5)
        
        # Should return 422 (validation error) because no file was provided
        if response.status_code == 422:
            print("✅ Predict endpoint correctly rejects requests without files")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Predict endpoint error: {e}")
        return False


def test_predict_endpoint_with_invalid_file():
    """Test the predict endpoint with an invalid file."""
    print("\n📎 Testing predict endpoint (with invalid file)...")
    
    try:
        # Create a dummy text file
        files = {"file": ("test.txt", "This is not an image", "text/plain")}
        response = requests.post("http://localhost:8000/predict", files=files, timeout=10)
        
        # Should return 400 (bad request) because of invalid file format
        if response.status_code == 400:
            print("✅ Predict endpoint correctly rejects invalid file formats")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Predict endpoint error: {e}")
        return False


def create_test_image():
    """Create a simple test image for testing."""
    print("\n🎨 Creating test image...")
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple 224x224 RGB image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        test_image_path = "test_dog_image.jpg"
        image.save(test_image_path, "JPEG")
        
        print(f"✅ Test image created: {test_image_path}")
        return test_image_path
        
    except ImportError:
        print("⚠️  PIL not available - cannot create test image")
        return None
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return None


def test_predict_endpoint_with_image(image_path):
    """Test the predict endpoint with a real image."""
    print(f"\n🖼️  Testing predict endpoint (with image: {image_path})...")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful!")
            print(f"   Predicted breed: {data.get('predicted_breed', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def test_api_documentation():
    """Test if API documentation is accessible."""
    print("\n📚 Testing API documentation...")
    
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        
        if response.status_code == 200:
            print("✅ API documentation accessible at /docs")
            return True
        else:
            print(f"❌ API documentation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API documentation error: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Dog Vision AI Backend - API Testing")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_predict_endpoint_without_file,
        test_predict_endpoint_with_invalid_file,
        test_api_documentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)  # Small delay between tests
    
    # Test with image if possible
    test_image_path = create_test_image()
    if test_image_path:
        total += 1
        if test_predict_endpoint_with_image(test_image_path):
            passed += 1
        
        # Clean up test image
        try:
            os.remove(test_image_path)
            print(f"🧹 Cleaned up test image: {test_image_path}")
        except:
            pass
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The backend service is working correctly.")
        return 0
    else:
        print("🚨 Some tests failed. Check the server status and dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())