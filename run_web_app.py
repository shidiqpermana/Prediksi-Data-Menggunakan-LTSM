"""
Bike Sharing Demand - Simple Web App Runner
Script sederhana untuk menjalankan web app setelah model dilatih
"""

import os
import sys
import subprocess
import time

def check_model_files():
    """Cek apakah file model sudah ada"""
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

def start_web_app():
    """Jalankan web app"""
    print("\n🌐 Starting Bike Sharing Demand Web Application...")
    print("="*60)
    
    try:
        # Start Flask app
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Web application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting web app: {e}")
    except FileNotFoundError:
        print("❌ Flask app file not found. Please check app.py exists.")

def main():
    """Main function"""
    print("🚀 Bike Sharing Demand - Web App Launcher")
    print("="*50)
    
    # Check if model files exist
    if not check_model_files():
        print("\n❌ Cannot start web app. Model files missing.")
        return
    
    print("\n✅ All model files found!")
    print("\n📋 Web App Features:")
    print("- Historical data visualization")
    print("- Interactive predictions")
    print("- 24-hour forecast")
    print("- Data insights")
    
    print("\n🌐 Starting web application...")
    print("The app will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    
    # Start web app
    start_web_app()

if __name__ == "__main__":
    main()
