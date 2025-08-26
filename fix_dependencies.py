#!/usr/bin/env python3
"""
Dependency Fix Script for Smart Bus Attendance System
=====================================================

This script fixes common dependency conflicts, especially the numpy/pandas
version incompatibility that causes "numpy.dtype size changed" errors.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    """Fix dependency conflicts"""
    print("🔧 Smart Bus Attendance System - Dependency Fix")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in a virtual environment")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 1: Uninstall conflicting packages
    print("\n📦 Step 1: Removing conflicting packages...")
    uninstall_cmd = "pip uninstall numpy pandas tensorflow mediapipe opencv-python -y"
    run_command(uninstall_cmd, "Uninstalling conflicting packages")
    
    # Step 2: Install numpy first (specific compatible version)
    print("\n📦 Step 2: Installing compatible numpy...")
    numpy_cmd = "pip install numpy==1.24.3"
    if not run_command(numpy_cmd, "Installing numpy 1.24.3"):
        print("❌ Failed to install numpy. Trying fallback version...")
        run_command("pip install numpy==1.21.6", "Installing numpy 1.21.6 (fallback)")
    
    # Step 3: Install other packages
    print("\n📦 Step 3: Installing other required packages...")
    packages = [
        "opencv-python>=4.5.0",
        "PyQt5>=5.15.0", 
        "pandas",
        "mediapipe>=0.8.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Step 4: Install tensorflow last (often causes the most conflicts)
    print("\n📦 Step 4: Installing tensorflow...")
    tf_cmd = "pip install tensorflow>=2.6.0"
    if not run_command(tf_cmd, "Installing tensorflow"):
        print("⚠️  TensorFlow installation failed. Trying CPU-only version...")
        run_command("pip install tensorflow-cpu", "Installing tensorflow-cpu")
    
    # Step 5: Verify installation
    print("\n✅ Step 5: Verifying installation...")
    verification_script = '''
import sys
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} - OK")
    
    import cv2
    print(f"✅ OpenCV {cv2.__version__} - OK")
    
    from PyQt5.QtWidgets import QApplication
    print("✅ PyQt5 - OK")
    
    import pandas as pd
    print(f"✅ Pandas {pd.__version__} - OK")
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe {mp.__version__} - OK")
    except Exception as e:
        print(f"⚠️  MediaPipe - Warning: {e}")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - OK")
    except Exception as e:
        print(f"⚠️  TensorFlow - Warning: {e}")
        
    print("\\n🎉 Verification complete!")
    
except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)
'''
    
    try:
        exec(verification_script)
        print("\n🎉 All dependencies fixed successfully!")
        print("You can now run: python main.py")
    except Exception as e:
        print(f"\n⚠️  Some issues remain: {e}")
        print("The basic GUI should still work. Try running: python main.py")

if __name__ == "__main__":
    main()
