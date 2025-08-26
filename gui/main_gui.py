import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QTextEdit
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import face detection, recognition, attendance modules
from src.face_detection import detect_faces
from src.face_recognition import load_facenet_model, load_embeddings_db, get_face_embedding, recognize_face
from src.attendance import mark_attendance

# Import registration GUI
from gui.register_gui import RegistrationWindow

# ----------------------------
# Main Attendance GUI Class
# ----------------------------
class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚌 Smart Bus Attendance System")
        self.resize(1200, 800)
        
        # Set modern dark theme for the application
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                background-color: #2d2d2d;
                border: 2px solid #4a90e2;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 15px;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #4a90e2;
            }
            QPushButton {
                background-color: #4a90e2;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 2px solid #4a90e2;
                border-radius: 8px;
                padding: 8px;
                color: #ffffff;
                font-family: Consolas, monospace;
            }
        """)

        # Bus-specific settings
        self.bus_route = "Route A"
        self.boarding_enabled = False
        self.attendance_session_active = False
        
        # Initialize models and embeddings with error handling
        self.facenet_model = None
        self.embeddings_db = {}
        self.models_loaded = False
        
        # Attendance tracking
        self.session_attendance = {}  # Track who has boarded this session
        self.recent_recognitions = {}  # Prevent duplicate rapid recognitions

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera")

        # Setup UI elements
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface for bus attendance"""
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - Camera and controls
        left_panel = QVBoxLayout()
        
        # Camera feed with enhanced styling
        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setStyleSheet("""
            border: 3px solid #4a90e2; 
            background-color: #000000; 
            border-radius: 15px;
            margin: 10px;
        """)
        left_panel.addWidget(self.video_label)
        
        # Control buttons with enhanced colors
        controls_group = QGroupBox("🚌 Bus Controls")
        controls_layout = QVBoxLayout()
        
        self.route_label = QLabel(f"🚌 Route: {self.bus_route}")
        self.route_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4a90e2; margin: 5px;")
        
        self.status_label = QLabel("Status: System Ready")
        self.status_label.setStyleSheet("font-size: 14px; color: #00ff88; margin: 5px; font-weight: bold;")
        
        self.start_session_button = QPushButton("🟢 Start Boarding Session")
        self.start_session_button.clicked.connect(self.toggle_boarding_session)
        self.start_session_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 15px;
                font-size: 14px;
                border-radius: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.register_button = QPushButton("👤 Register New Student")
        self.register_button.clicked.connect(self.open_registration_window)
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; 
                color: white; 
                font-weight: bold; 
                padding: 12px;
                font-size: 14px;
                border-radius: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        
        self.load_models_button = QPushButton("🤖 Load AI Models")
        self.load_models_button.clicked.connect(self.load_models)
        self.load_models_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; 
                color: white; 
                font-weight: bold; 
                padding: 12px;
                font-size: 14px;
                border-radius: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #7b1fa2;
            }
        """)
        
        self.take_attendance_button = QPushButton("📋 Take Attendance")
        self.take_attendance_button.clicked.connect(self.take_single_attendance)
        self.take_attendance_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                font-weight: bold; 
                padding: 12px;
                font-size: 14px;
                border-radius: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        self.refresh_button = QPushButton("🔄 Refresh Database")
        self.refresh_button.clicked.connect(self.manual_refresh_database)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #607D8B; 
                color: white; 
                font-weight: bold; 
                padding: 12px;
                font-size: 14px;
                border-radius: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #455A64;
            }
        """)
        
        controls_layout.addWidget(self.route_label)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.start_session_button)
        controls_layout.addWidget(self.take_attendance_button)
        controls_layout.addWidget(self.register_button)
        controls_layout.addWidget(self.load_models_button)
        controls_layout.addWidget(self.refresh_button)
        controls_group.setLayout(controls_layout)
        left_panel.addWidget(controls_group)
        
        # Right panel - Attendance log with enhanced styling
        right_panel = QVBoxLayout()
        
        attendance_group = QGroupBox("📋 Today's Attendance")
        attendance_layout = QVBoxLayout()
        
        self.attendance_count_label = QLabel("Students Boarded: 0")
        self.attendance_count_label.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #4a90e2; 
            margin: 10px;
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 8px;
        """)
        
        self.attendance_log = QTextEdit()
        self.attendance_log.setReadOnly(True)
        self.attendance_log.setMinimumWidth(350)
        self.attendance_log.setStyleSheet("""
            background-color: #1a1a1a; 
            border: 2px solid #4a90e2;
            border-radius: 10px;
            color: #ffffff;
            font-family: Consolas, monospace;
            font-size: 12px;
            padding: 10px;
        """)
        
        attendance_layout.addWidget(self.attendance_count_label)
        attendance_layout.addWidget(self.attendance_log)
        attendance_group.setLayout(attendance_layout)
        right_panel.addWidget(attendance_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

        # Setup timer for live feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms interval ~ 33 FPS for smoother video
        
        self.status_label.setText("Status: 📹 Camera active - Basic face detection (Click 'Load AI Models' for face recognition)")
        self.load_attendance_log()

    def load_models(self):
        """Load ML models with error handling and enhanced feedback"""
        try:
            self.status_label.setText("Status: 🤖 Loading ML models...")
            self.status_label.setStyleSheet("font-size: 14px; color: #FF9800; margin: 5px; font-weight: bold;")
            self.load_models_button.setEnabled(False)
            self.update_attendance_log("🤖 Loading AI models for face recognition...")
            
            # Load models and embeddings
            self.facenet_model = load_facenet_model()
            self.embeddings_db = load_embeddings_db()
            self.models_loaded = True
            
            self.status_label.setText("Status: ✅ ML models loaded - Full face recognition active")
            self.status_label.setStyleSheet("font-size: 14px; color: #4CAF50; margin: 5px; font-weight: bold;")
            self.load_models_button.setText("✅ Models Loaded")
            self.load_models_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50; 
                    color: white; 
                    font-weight: bold; 
                    padding: 12px;
                    font-size: 14px;
                    border-radius: 10px;
                    margin: 5px;
                }
            """)
            self.update_attendance_log("✅ AI models loaded successfully - Face recognition enabled!")
            print("[INFO] ML models loaded successfully")
            
        except Exception as e:
            self.status_label.setText(f"Status: ❌ Model loading failed - {str(e)}")
            self.status_label.setStyleSheet("font-size: 14px; color: #f44336; margin: 5px; font-weight: bold;")
            self.load_models_button.setEnabled(True)
            self.update_attendance_log(f"❌ Failed to load AI models: {str(e)}")
            print(f"[ERROR] Failed to load models: {e}")
            # Continue with basic face detection

    def reload_embeddings_database(self):
        """Reload the embeddings database to get newly registered students"""
        try:
            print("[INFO] Reloading embeddings database...")
            old_count = len(self.embeddings_db)
            self.embeddings_db = load_embeddings_db()
            new_count = len(self.embeddings_db)
            
            if new_count > old_count:
                added_count = new_count - old_count
                self.update_attendance_log(f"🔄 Embeddings database updated! Added {added_count} new student(s). Total: {new_count}")
                print(f"[INFO] Embeddings database reloaded: {old_count} -> {new_count} entries")
            else:
                self.update_attendance_log(f"🔄 Embeddings database refreshed. Total students: {new_count}")
                print(f"[INFO] Embeddings database refreshed: {new_count} entries")
                
            return True
        except Exception as e:
            print(f"[ERROR] Failed to reload embeddings database: {e}")
            self.update_attendance_log(f"❌ Failed to reload embeddings database: {str(e)}")
            return False

    # ----------------------------
    # Update video feed frame
    # ----------------------------
    def update_frame(self):
        """Update video feed frame with enhanced face detection display"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Add session status overlay with enhanced styling
        overlay = frame.copy()
        alpha = 0.3
        
        if self.attendance_session_active:
            cv2.rectangle(overlay, (10, 10), (450, 70), (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "BOARDING SESSION ACTIVE", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Students can board the bus", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.rectangle(overlay, (10, 10), (450, 70), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "BOARDING SESSION INACTIVE", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Click 'Start Boarding' to begin", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Face detection with enhanced visual feedback
        if self.models_loaded and self.facenet_model is not None:
            # Advanced face recognition mode
            boxes = detect_faces(frame)
            print(f"[DEBUG] Detected {len(boxes)} faces")  # Debug print

            # Process each detected face
            for i, box in enumerate(boxes):
                x, y, w, h = box
                face_img = frame[y:y+h, x:x+w]

                # Skip if face_img too small
                if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    continue

                # Get embedding and recognize
                embedding = get_face_embedding(self.facenet_model, face_img)
                name, score = recognize_face(embedding, self.embeddings_db, threshold=0.5)

                # Enhanced visual display for recognition
                if name != "Unknown":
                    # Recognized student - get student info
                    student_info = self.get_student_info(name)
                    roll_no = student_info.get('roll_no', 'N/A') if student_info else 'N/A'
                    
                    # Green for recognized faces
                    color = (0, 255, 0)
                    label = f"{name} (ID: {roll_no})"
                    confidence_label = f"Confidence: {score:.2f}"
                    
                    # Enhanced border for recognized faces
                    cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), (0, 255, 0), 3)
                    
                    # Try to mark attendance if boarding is active
                    if self.boarding_enabled and name not in self.session_attendance:
                        if self.mark_student_attendance(name):
                            # Flash bright green border for successful boarding
                            cv2.rectangle(frame, (x-8, y-8), (x+w+8, y+h+8), (0, 255, 0), 8)
                        
                else:
                    # Unknown person - red border
                    color = (0, 0, 255)
                    label = "Unknown Face"
                    confidence_label = f"Confidence: {score:.2f}"
                    cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), (0, 0, 255), 3)
                
                # Draw main bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Enhanced text display with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y-40), (x + label_size[0] + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, confidence_label, (x + 5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add boarding status with enhanced visuals
                if name != "Unknown":
                    if name in self.session_attendance:
                        status_text = "ALREADY BOARDED"
                        status_color = (255, 165, 0)  # Orange
                    elif self.boarding_enabled:
                        status_text = "READY TO BOARD"
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_text = "BOARDING CLOSED"
                        status_color = (0, 0, 255)  # Red
                    
                    # Status background
                    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x, y+h+5), (x + status_size[0] + 10, y+h+25), status_color, -1)
                    cv2.putText(frame, status_text, (x + 5, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            # Basic face detection mode using Mediapipe
            boxes = detect_faces(frame)
            print(f"[DEBUG] Basic mode - Detected {len(boxes)} faces")  # Debug print

            # Draw rectangles around detected faces with enhanced styling
            for i, box in enumerate(boxes):
                x, y, w, h = box
                
                # Draw main detection box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                
                # Add face number
                face_label = f"Face #{i+1} Detected"
                label_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), (255, 0, 0), -1)
                cv2.putText(frame, face_label, (x + 5, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show AI models needed message
                models_text = "Load AI Models for Recognition"
                models_size = cv2.getTextSize(models_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(frame, (x, y+h+5), (x + models_size[0] + 10, y+h+25), (255, 100, 0), -1)
                cv2.putText(frame, models_text, (x + 5, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add system info overlay
        info_y = frame.shape[0] - 80
        cv2.rectangle(frame, (10, info_y), (300, frame.shape[0] - 10), (50, 50, 50), -1)
        cv2.putText(frame, f"Route: {self.bus_route}", (20, info_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Students Boarded: {len(self.session_attendance)}", (20, info_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Models: {'Loaded' if self.models_loaded else 'Basic Mode'}", (20, info_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Convert frame to QImage for PyQt display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), 1, 1)  # Qt.KeepAspectRatio, Qt.SmoothTransformation
        self.video_label.setPixmap(scaled_pixmap)

    # ----------------------------
    # Open registration window
    # ----------------------------
    def open_registration_window(self):
        """Open registration window with callback for database updates"""
        self.registration_window = RegistrationWindow(
            facenet_model=self.facenet_model if self.models_loaded else None,
            embeddings_db=self.embeddings_db if self.models_loaded else None,
            parent_callback=self.on_registration_completed  # Add callback
        )
        self.registration_window.show()

    def on_registration_completed(self, student_name):
        """Callback when a student registration is completed"""
        print(f"[INFO] Registration completed for {student_name}, reloading embeddings...")
        if self.models_loaded:
            success = self.reload_embeddings_database()
            if success:
                self.update_attendance_log(f"✅ {student_name} registered successfully and database updated!")
            else:
                self.update_attendance_log(f"⚠️  {student_name} registered but failed to update database - try reloading models")
        else:
            self.update_attendance_log(f"✅ {student_name} registered in basic mode")

    def manual_refresh_database(self):
        """Manually refresh the embeddings database"""
        if not self.models_loaded:
            self.update_attendance_log("⚠️  Please load AI models first before refreshing database")
            return
            
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText("🔄 Refreshing...")
        
        success = self.reload_embeddings_database()
        
        if success:
            self.refresh_button.setText("✅ Database Refreshed")
        else:
            self.refresh_button.setText("❌ Refresh Failed")
            
        # Reset button after 2 seconds
        QTimer.singleShot(2000, lambda: (
            self.refresh_button.setText("🔄 Refresh Database"),
            self.refresh_button.setEnabled(True)
        ))

    # ----------------------------
    # Toggle boarding session on/off
    # ----------------------------
    def toggle_boarding_session(self):
        """Toggle boarding session on/off with enhanced styling"""
        if not self.attendance_session_active:
            # Start boarding session
            self.attendance_session_active = True
            self.boarding_enabled = True
            self.session_attendance.clear()
            self.start_session_button.setText("🔴 End Boarding Session")
            self.start_session_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336; 
                    color: white; 
                    font-weight: bold; 
                    padding: 15px;
                    font-size: 14px;
                    border-radius: 10px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            self.status_label.setText("Status: 🚌 BOARDING SESSION ACTIVE - Students can now board")
            self.status_label.setStyleSheet("font-size: 14px; color: #4CAF50; margin: 5px; font-weight: bold;")
            self.update_attendance_log("🚌 === BOARDING SESSION STARTED ===")
        else:
            # End boarding session
            self.attendance_session_active = False
            self.boarding_enabled = False
            self.start_session_button.setText("🟢 Start Boarding Session")
            self.start_session_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50; 
                    color: white; 
                    font-weight: bold; 
                    padding: 15px;
                    font-size: 14px;
                    border-radius: 10px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.status_label.setText("Status: ⛔ Boarding session ended - No more students can board")
            self.status_label.setStyleSheet("font-size: 14px; color: #f44336; margin: 5px; font-weight: bold;")
            self.update_attendance_log("🚌 === BOARDING SESSION ENDED ===")

    # ----------------------------
    # Load today's attendance log
    # ----------------------------
    def load_attendance_log(self):
        """Load today's attendance log"""
        try:
            from src.attendance import get_attendance_file_path
            import csv
            
            file_path = get_attendance_file_path()
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    count = 0
                    for row in reader:
                        if len(row) >= 3:
                            count += 1
                            self.update_attendance_log(f"✅ {row[0]} - {row[2]}")
                    
                    self.attendance_count_label.setText(f"Students Boarded: {count}")
        except Exception as e:
            print(f"[ERROR] Failed to load attendance log: {e}")

    # ----------------------------
    # Update the attendance log display
    # ----------------------------
    def update_attendance_log(self, message):
        """Update the attendance log display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        self.attendance_log.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.attendance_log.textCursor()
        cursor.movePosition(cursor.End)
        self.attendance_log.setTextCursor(cursor)

    # ----------------------------
    # Mark attendance for a student with bus-specific logic
    # ----------------------------
    def mark_student_attendance(self, name):
        """Mark attendance for a student with bus-specific logic"""
        import time
        current_time = time.time()
        
        # Prevent rapid duplicate recognitions (within 5 seconds)
        if name in self.recent_recognitions:
            if current_time - self.recent_recognitions[name] < 5:
                return False
        
        # Check if boarding session is active
        if not self.boarding_enabled:
            self.update_attendance_log(f"❌ {name} - Boarding not allowed (session not active)")
            return False
        
        # Check if student already boarded this session
        if name in self.session_attendance:
            self.update_attendance_log(f"⚠️ {name} - Already boarded this session")
            return False
        
        # Get student information
        student_info = self.get_student_info(name)
        roll_no = student_info.get('roll_no', 'ID-NotAvailable') if student_info else 'ID-NotAvailable'
        
        # Mark attendance
        try:
            mark_attendance(name, roll_no, f"Bus-{self.bus_route}")
            self.session_attendance[name] = current_time
            self.recent_recognitions[name] = current_time
            
            # Update UI
            count = len(self.session_attendance)
            self.attendance_count_label.setText(f"Students Boarded: {count}")
            self.update_attendance_log(f"🎯 {name} (ID: {roll_no}) - BOARDED SUCCESSFULLY!")
            
            return True
        except Exception as e:
            self.update_attendance_log(f"❌ {name} - Error marking attendance: {e}")
            return False

    # ----------------------------
    # Take attendance for a single person (manual check)
    # ----------------------------
    def take_single_attendance(self):
        """Take attendance for currently visible person"""
        if not self.models_loaded or self.facenet_model is None:
            self.update_attendance_log("❌ Cannot take attendance - AI models not loaded")
            return
        
        # Capture current frame
        ret, frame = self.cap.read()
        if not ret:
            self.update_attendance_log("❌ Cannot capture camera frame")
            return
        
        # Detect faces
        from src.face_detection import detect_faces
        boxes = detect_faces(frame)
        
        if len(boxes) == 0:
            self.update_attendance_log("❌ No face detected in camera")
            return
        elif len(boxes) > 1:
            self.update_attendance_log("❌ Multiple faces detected - ensure only one person is visible")
            return
        
        # Process the single detected face
        box = boxes[0]
        x, y, w, h = box
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.shape[0] < 10 or face_img.shape[1] < 10:
            self.update_attendance_log("❌ Face too small to process")
            return
        
        # Get embedding and recognize
        embedding = get_face_embedding(self.facenet_model, face_img)
        name, score = recognize_face(embedding, self.embeddings_db, threshold=0.5)
        
        if name == "Unknown":
            self.update_attendance_log(f"❌ Unknown face detected (confidence: {score:.2f})")
        else:
            # Get student information
            student_info = self.get_student_info(name)
            if student_info:
                roll_no = student_info.get('roll_no', 'N/A')
                self.update_attendance_log(f"✅ {name} (ID: {roll_no}) - Present (confidence: {score:.2f})")
                
                # Mark attendance if not already marked
                if not self.is_already_present_today(name):
                    if mark_attendance(name, roll_no, f"Manual-{self.bus_route}"):
                        self.update_attendance_log(f"📝 Attendance marked for {name}")
                    else:
                        self.update_attendance_log(f"⚠️ Failed to mark attendance for {name}")
                else:
                    self.update_attendance_log(f"ℹ️ {name} already marked present today")
            else:
                self.update_attendance_log(f"✅ {name} recognized but no student data found (confidence: {score:.2f})")

    # ----------------------------
    # Get student information from JSON file
    # ----------------------------
    def get_student_info(self, name):
        """Get student information by name from JSON file"""
        import json
        
        students_file = "data/students/students.json"
        if os.path.exists(students_file):
            try:
                with open(students_file, 'r') as f:
                    students = json.load(f)
                    return students.get(name, None)
            except Exception as e:
                print(f"[ERROR] Failed to read student data: {e}")
        return None

    # ----------------------------
    # Check if student is already present today
    # ----------------------------
    def is_already_present_today(self, name):
        """Check if student is already marked present today"""
        try:
            from src.attendance import is_already_present
            return is_already_present(name)
        except Exception as e:
            print(f"[ERROR] Failed to check attendance status: {e}")
            return False

    # ----------------------------
    # Close event - release camera
    # ----------------------------
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

# ----------------------------
# Main entry point
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec_())
