#!/usr/bin/env python3
"""
🚌 Smart Bus Attendance System - Main Entry Point
=================================================

This is the main entry point for the Smart Bus Attendance System.
The system provides:
- Real-time face detection and recognition
- Student registration with live camera feed
- Attendance tracking for bus routes
- Modern dark-themed GUI interface

Usage:
    python main.py

Requirements:
    - Python 3.8+
    - OpenCV
    - PyQt5
    - MediaPipe
    - NumPy

Author: AI Assistant
Date: July 2025
"""

import sys
import os
from pathlib import Path

# Ensure we can import from our modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'PyQt5': 'PyQt5',
        'numpy': 'numpy',
        'pickle': 'built-in'
    }
    
    missing_packages = []
    incompatible_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'pickle':
                import pickle
            elif package == 'cv2':
                import cv2
            elif package == 'PyQt5':
                from PyQt5.QtWidgets import QApplication
            elif package == 'numpy':
                import numpy
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(install_name)
            print(f"❌ {package} - MISSING")
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                incompatible_packages.append(package)
                print(f"⚠️  {package} - VERSION CONFLICT")
            else:
                missing_packages.append(install_name)
                print(f"❌ {package} - ERROR: {e}")
    
    # Check mediapipe separately due to dependency conflicts
    try:
        import mediapipe
        print("✅ mediapipe - OK")
    except ImportError:
        missing_packages.append('mediapipe')
        print("❌ mediapipe - MISSING")
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            incompatible_packages.append('mediapipe')
            print("⚠️  mediapipe - VERSION CONFLICT (numpy compatibility issue)")
        else:
            print(f"❌ mediapipe - ERROR: {e}")
    except Exception as e:
        print(f"⚠️  mediapipe - WARNING: {e}")
        print("   (MediaPipe has dependency conflicts but may still work)")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    if incompatible_packages:
        print(f"\n⚠️  Package version conflicts detected: {', '.join(incompatible_packages)}")
        print("This is usually caused by numpy version incompatibility.")
        print("Try fixing with:")
        print("pip uninstall numpy pandas tensorflow mediapipe -y")
        print("pip install numpy==1.24.3 pandas mediapipe tensorflow")
        print("\nContinuing anyway - the system may still work for basic operations...")
        return True  # Continue anyway for basic functionality
    
    return True

def setup_directories():
    """Ensure required directories exist"""
    required_dirs = [
        'data/embeddings',
        'data/faces',
        'data/students',
        'data/attendance_logs',
        'models'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 {dir_path} - Ready")

def main():
    """Main entry point for the Smart Bus Attendance System"""
    print("🚌 Smart Bus Attendance System")
    print("=" * 50)
    print("Starting system initialization...\n")
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    if not check_requirements():
        print("\n❌ System requirements not met. Please install missing packages.")
        sys.exit(1)
    
    print("\n📂 Setting up directories...")
    setup_directories()
    
    print("\n🖥️  Launching GUI...")
    
    try:
        # Import and launch the main GUI
        from PyQt5.QtWidgets import QApplication
        from gui.main_gui import AttendanceApp
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Smart Bus Attendance System")
        app.setApplicationVersion("1.0.0")
        
        # Create and show main window
        window = AttendanceApp()
        window.show()
        
        print("✅ GUI launched successfully!")
        print("\n📋 Instructions:")
        print("1. Click 'Load AI Models' to enable face recognition")
        print("2. Click 'Register New Student' to add students")
        print("3. Click 'Start Boarding Session' to begin attendance")
        print("4. Students will be automatically recognized when they face the camera")
        print("\n🎯 System ready for operation!")
        
        # Start the application event loop
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"❌ Failed to import GUI components: {e}")
        print("Make sure all files are in the correct location:")
        print("- gui/main_gui.py")
        print("- gui/register_gui.py")
        print("- src/face_detection.py")
        print("- src/face_recognition.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
