#!/usr/bin/env python3
"""
Bike Sharing Demand - Web Application Starter
Script untuk menjalankan aplikasi web dengan pengecekan dependencies
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask',
        'flask_cors', 
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tensorflow',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'flask_cors':
                import flask_cors
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Please install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if all model files exist"""
    required_files = [
        'models/lstm_model.h5',
        'models/feature_scaler.pkl',
        'models/target_scaler.pkl',
        'models/feature_names.npy'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ Found: {file}")
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("Please run the complete_pipeline.ipynb notebook first!")
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 Bike Sharing Demand - Web Application Starter")
    print("=" * 60)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Cannot start app. Missing dependencies.")
        return
    
    # Check model files
    print("\n🤖 Checking model files...")
    if not check_model_files():
        print("\n❌ Cannot start app. Missing model files.")
        return
    
    print("\n✅ All checks passed!")
    print("\n🌐 Starting web application...")
    print("The app will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Start Flask app
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Web application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting web app: {e}")
    except FileNotFoundError:
        print("❌ Flask app file not found. Please check app.py exists.")

if __name__ == "__main__":
    main()
