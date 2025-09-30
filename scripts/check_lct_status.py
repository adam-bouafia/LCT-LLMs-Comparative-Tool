#!/usr/bin/env python3
"""
LCT Status Check Script
Verifies that the LLM Comparison Tool is properly configured and working
"""

import os
import sys
from pathlib import Path
import json

# Add project to path
project_root = Path(__file__).parent
app_src = project_root / "app" / "src"
sys.path.insert(0, str(app_src))

def check_virtual_environment():
    """Check if virtual environment is properly set up"""
    venv_paths = [
        project_root / "llm-experiment-runner" / ".venv",
        project_root / ".venv"
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists():
            python_path = venv_path / "bin" / "python"
            if python_path.exists():
                print(f"✅ Virtual environment found: {venv_path}")
                return str(python_path)
    
    print("❌ No virtual environment found")
    return None

def check_data_directory():
    """Check data directory structure"""
    data_dir = project_root / "data"
    models_dir = data_dir / "models"
    
    print(f"\n📁 Data Directory: {data_dir}")
    
    if not data_dir.exists():
        print("❌ Data directory missing")
        return False
    
    if not models_dir.exists():
        print("❌ Models directory missing")
        return False
    
    print(f"✅ Data structure exists")
    
    # Count models
    model_count = len([d for d in models_dir.iterdir() if d.is_dir()])
    print(f"   📦 Models: {model_count}")
    
    return True

def check_model_metadata():
    """Check if model metadata is properly configured"""
    try:
        from core.data_manager import DataManager
        
        dm = DataManager()
        models = dm.list_downloaded_models()
        
        print(f"\n🤖 Model Status:")
        
        if not models:
            print("⚠️  No models found")
            return False
        
        valid_models = 0
        for model in models:
            model_id = model.get("model_id", "Unknown")
            model_type = model.get("config", {}).get("model_type", "Unknown")
            
            if model_type != "Unknown":
                print(f"   ✅ {model_id}: {model_type}")
                valid_models += 1
            else:
                print(f"   ❌ {model_id}: Unknown type")
        
        if valid_models == len(models):
            print(f"✅ All {valid_models} models have proper metadata")
            return True
        else:
            print(f"⚠️  {len(models) - valid_models}/{len(models)} models need metadata fix")
            return False
    
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "torch", "transformers", "rich", "datasets", 
        "nltk", "sentence_transformers", "codecarbon"
    ]
    
    print(f"\n📦 Dependencies:")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if not missing:
        print("✅ All required packages installed")
        return True
    else:
        print(f"❌ Missing packages: {', '.join(missing)}")
        return False

def check_scripts():
    """Check if launch scripts exist and are executable"""
    scripts = [
        "run_lct.sh",
        "run_lct_cli.sh", 
        "start_lct.sh",
        "fix_model_metadata.py"
    ]
    
    print(f"\n🚀 Launch Scripts:")
    
    all_good = True
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            if os.access(script_path, os.X_OK):
                print(f"   ✅ {script}")
            else:
                print(f"   ⚠️  {script} (not executable)")
                all_good = False
        else:
            print(f"   ❌ {script} (missing)")
            all_good = False
    
    return all_good

def main():
    print("🔍 LLM Comparison Tool Status Check")
    print("=" * 50)
    
    checks = [
        ("Virtual Environment", check_virtual_environment),
        ("Data Directory", check_data_directory),
        ("Model Metadata", check_model_metadata),
        ("Dependencies", check_dependencies),
        ("Launch Scripts", check_scripts)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            if callable(check_func):
                result = check_func()
            else:
                result = check_func
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    passed = 0
    for name, result in results:
        if result:
            print(f"   ✅ {name}")
            passed += 1
        else:
            print(f"   ❌ {name}")
    
    print(f"\n🎯 Status: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 LLM Comparison Tool is ready to use!")
        print("   Run: ./start_lct.sh or ./run_lct.sh")
    else:
        print("\n⚠️  Some issues need to be resolved:")
        print("   1. Run Install Tools (option 9) if dependencies are missing")
        print("   2. Run fix_model_metadata.py if models show Unknown type")
        print("   3. Check file permissions for scripts")

if __name__ == "__main__":
    main()