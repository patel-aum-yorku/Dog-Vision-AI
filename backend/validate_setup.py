#!/usr/bin/env python3
"""
Simple validation script to check the backend structure without dependencies.
This script validates the code structure and provides setup verification.
"""

import os
import sys


def check_file_structure():
    """Check if all required files exist."""
    print("🔍 Checking file structure...")
    
    required_files = [
        "app.py",
        "model_service.py", 
        "image_processor.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0


def check_model_file():
    """Check if the model file exists."""
    print("\n🤖 Checking model file...")
    
    model_path = "../20250421-07101745219413-10k-images-mobilenetv2-Adam.keras"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Make sure the model file is in the parent directory")
        return False


def check_code_syntax():
    """Check Python syntax of the main files."""
    print("\n🐍 Checking Python syntax...")
    
    python_files = ["app.py", "model_service.py", "image_processor.py"]
    
    all_valid = True
    for file in python_files:
        try:
            with open(file, 'r') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, file, 'exec')
            print(f"✅ {file} - Syntax OK")
            
        except SyntaxError as e:
            print(f"❌ {file} - Syntax Error: {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {file} - Error: {e}")
            all_valid = False
    
    return all_valid


def check_imports():
    """Check if imports can be resolved (without executing)."""
    print("\n📦 Checking import statements...")
    
    # These are the imports we expect to work once dependencies are installed
    expected_packages = {
        "fastapi": "FastAPI web framework",
        "uvicorn": "ASGI server",
        "tensorflow": "Machine learning framework", 
        "PIL": "Python Imaging Library (Pillow)",
        "numpy": "Numerical computing library"
    }
    
    for package, description in expected_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - Available ({description})")
        except ImportError:
            print(f"⚠️  {package} - Not installed ({description})")


def show_next_steps():
    """Show next steps for setup."""
    print("\n🚀 Next Steps:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Run the service:")
    print("   python app.py")
    print("\n3. Test the API:")
    print("   Open http://localhost:8000/docs in your browser")
    print("   Or use curl: curl http://localhost:8000/health")
    print("\n4. Upload an image for prediction:")
    print("   curl -X POST http://localhost:8000/predict -H 'Content-Type: multipart/form-data' -F 'file=@dog_image.jpg'")


def main():
    """Main validation function."""
    print("🔧 Dog Vision AI Backend - Structure Validation")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    structure_ok = check_file_structure()
    model_ok = check_model_file()
    syntax_ok = check_code_syntax()
    check_imports()
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY:")
    print(f"   File Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   Model File: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"   Code Syntax: {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    
    if structure_ok and model_ok and syntax_ok:
        print("\n🎉 Backend structure is valid! Ready for dependency installation.")
        show_next_steps()
        return 0
    else:
        print("\n🚨 Issues found. Please fix the problems above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())