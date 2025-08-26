"""
bus_face_attendance.gui

This package contains GUI modules for:

- main_gui.py : Main attendance marking interface
- register_gui.py : Student registration interface
- utils.py : Helper functions for GUI

Author: Your Name
Date: YYYY-MM-DD
"""

# Import specific classes and functions instead of using wildcard imports
from .main_gui import AttendanceApp
from .register_gui import RegistrationWindow

# Define what gets exported when using "from gui import *"
__all__ = ['AttendanceApp', 'RegistrationWindow']

