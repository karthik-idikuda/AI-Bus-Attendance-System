# AI Smart Bus Attendance System

## Overview
An automated attendance management solution leveraging real-time facial recognition technology. This system streamlines the tracking of student flow on school buses, providing accurate, instantaneous logs and ensuring student safety through automated verification.

## Features
-   **Real-time Recognition**: Instant student identification using FaceNet embeddings.
-   **Live Monitoring**: Interactive dashboard showing real-time camera feed and detection status.
-   **Automated Logging**: Timestamped attendance records generated automatically in CSV format.
-   **Registration Module**: easy-to-use interface for enrolling new students.
-   **Mask Detection**: Integrated safety checks for health compliance.

## Technology Stack
-   **Framework**: PyQt5 for the Graphical User Interface.
-   **Computer Vision**: OpenCV and MediaPipe for image processing.
-   **AI Models**: FaceNet for recognition, YOLOv5 for object detection.
-   **Database**: JSON and CSV for lightweight data persistence.

## System Architecture
1.  **Capture**: Video feed input from bus cameras.
2.  **Process**: Frame analysis for face detection and alignment.
3.  **Identify**: Feature extraction and matching against the student database.
4.  **Log**: Attendance record creation and storage.

## Quick Start
```bash
# Clone the repository
git clone https://github.com/Nytrynox/AI-based-Attendance-Management-System.git

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python main.py
```

## License
MIT License

## Author
**Karthik Idikuda**
