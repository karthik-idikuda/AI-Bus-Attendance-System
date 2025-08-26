import sys
import cv2
import os
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import face recognition modules
from src.face_detection import detect_faces
from src.face_recognition import get_face_embedding
from src.utils import save_pickle, create_folder

# ----------------------------
# Registration Window Class
# ----------------------------
class RegistrationWindow(QWidget):
    def __init__(self, facenet_model=None, embeddings_db=None, parent_callback=None):
        super().__init__()
        self.setWindowTitle("👤 Register New Student")
        self.resize(900, 700)
        
        # Store the callback function to notify parent when registration is complete
        self.parent_callback = parent_callback
        
        # Set modern dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                margin: 5px;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 2px solid #4a90e2;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                color: #ffffff;
                margin: 5px;
            }
            QLineEdit:focus {
                border-color: #66b3ff;
            }
            QPushButton {
                background-color: #4a90e2;
                border: none;
                color: white;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)

        # Use shared model instances or load if not provided
        self.facenet_model = facenet_model
        self.embeddings_db = embeddings_db if embeddings_db is not None else {}
        self.models_available = facenet_model is not None

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera for registration")

        # Setup UI elements
        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setStyleSheet("""
            border: 3px solid #4a90e2; 
            background-color: #000000; 
            border-radius: 15px;
            margin: 10px;
        """)
        
        # Input fields
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter student name (e.g., John Doe)")
        
        self.roll_no_input = QLineEdit(self)
        self.roll_no_input.setPlaceholderText("Enter student roll number/ID (e.g., STU001)")
        
        # Buttons with enhanced styling
        self.capture_button = QPushButton("📸 Capture Face", self)
        self.capture_button.clicked.connect(self.capture_face)
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                font-weight: bold; 
                padding: 15px;
                font-size: 14px;
                border-radius: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        self.register_button = QPushButton("✅ Register Student", self)
        self.register_button.clicked.connect(self.register_student)
        self.register_button.setEnabled(False)  # Disabled until face is captured
        
        # Status label with enhanced styling
        self.status_label = QLabel(self)
        if self.models_available:
            self.status_label.setText("Status: 🤖 Ready for registration with face recognition")
            self.register_button.setStyleSheet("""
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
        else:
            self.status_label.setText("Status: ⚠️  ML models not loaded - Basic registration only")
            self.register_button.setText("✅ Register Student (Basic)")
            self.register_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800; 
                    color: white; 
                    font-weight: bold; 
                    padding: 15px;
                    font-size: 14px;
                    border-radius: 10px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #f57c00;
                }
            """)
        
        self.status_label.setStyleSheet("""
            font-size: 14px; 
            color: #4a90e2; 
            padding: 10px;
            background-color: #2d2d2d;
            border-radius: 8px;
            margin: 5px;
        """)
        
        # Face capture storage
        self.captured_face = None

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(QLabel("📹 Live Camera Feed - Position your face in the frame"))
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("👤 Student Information:"))
        layout.addWidget(self.name_input)
        layout.addWidget(self.roll_no_input)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.register_button)
        self.setLayout(layout)

        # Timer for live feed with higher FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33 FPS

    # ----------------------------
    # Update video feed frame
    # ----------------------------
    def update_frame(self):
        """Update video feed frame with enhanced face detection"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Use MediaPipe face detection for better accuracy
        try:
            boxes = detect_faces(frame, confidence_threshold=0.7)
        except Exception as e:
            print(f"[ERROR] MediaPipe face detection failed: {e}")
            # Fallback to OpenCV
            boxes = self._opencv_face_detection(frame)

        # Enhanced visual feedback
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                x, y, w, h = box
                
                # Draw main detection box with enhanced styling
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Add face number and quality indicator
                face_label = f"Face #{i+1} Detected"
                label_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), (0, 255, 0), -1)
                cv2.putText(frame, face_label, (x + 5, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Check face size quality
                face_area = w * h
                min_area = 100 * 100  # Minimum face size
                if face_area >= min_area:
                    quality_text = "Good Quality"
                    quality_color = (0, 255, 0)
                else:
                    quality_text = "Move Closer"
                    quality_color = (0, 165, 255)
                
                cv2.putText(frame, quality_text, (x, y+h+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2)
                
            # Instructions for single face
            if len(boxes) == 1:
                cv2.putText(frame, "Perfect! Click 'Capture Face' to proceed", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Multiple faces detected! Ensure only one person", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # No face detected
            cv2.putText(frame, "No face detected - Position yourself in front of camera", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Ensure good lighting and face the camera directly", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Add registration status overlay
        status_overlay = frame.copy()
        alpha = 0.3
        cv2.rectangle(status_overlay, (10, frame.shape[0]-80), (450, frame.shape[0]-10), (50, 50, 50), -1)
        cv2.addWeighted(status_overlay, alpha, frame, 1 - alpha, 0, frame)
        
        mode_text = "AI Mode" if self.models_available else "Basic Mode"
        cv2.putText(frame, f"Registration Mode: {mode_text}", (20, frame.shape[0]-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Faces Detected: {len(boxes)}", (20, frame.shape[0]-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Ready to Capture: {'Yes' if len(boxes) == 1 else 'No'}", (20, frame.shape[0]-20),
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

        self.current_frame = frame

    def _opencv_face_detection(self, frame):
        """Fallback OpenCV face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Convert to consistent format [x, y, w, h]
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, w, h])
        return boxes

    # ----------------------------
    # Capture face from current frame
    # ----------------------------
    def capture_face(self):
        """Capture and process the current face with enhanced validation"""
        if not hasattr(self, 'current_frame'):
            self.status_label.setText("Status: ❌ No camera feed available")
            return
        
        # Use MediaPipe face detection for better accuracy
        try:
            boxes = detect_faces(self.current_frame, confidence_threshold=0.7)
        except Exception as e:
            print(f"[ERROR] MediaPipe face detection failed: {e}")
            # Fallback to OpenCV
            boxes = self._opencv_face_detection(self.current_frame)
        
        if len(boxes) == 0:
            self.status_label.setText("Status: ❌ No face detected! Position your face properly and ensure good lighting.")
            return
        elif len(boxes) > 1:
            self.status_label.setText("Status: ❌ Multiple faces detected! Ensure only one person is visible in the camera.")
            return
        
        # Extract the face with improved processing
        x, y, w, h = boxes[0]
        
        # Add some padding around the face for better quality
        padding = 20
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(self.current_frame.shape[1] - x_padded, w + 2*padding)
        h_padded = min(self.current_frame.shape[0] - y_padded, h + 2*padding)
        
        face_img = self.current_frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        
        # Validate face quality
        if face_img.shape[0] < 80 or face_img.shape[1] < 80:
            self.status_label.setText("Status: ❌ Face too small! Move closer to the camera.")
            return
        
        # Check if face is too blurry
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        if laplacian_var < 100:  # Threshold for blur detection
            self.status_label.setText("Status: ⚠️  Image appears blurry. Ensure good lighting and hold steady.")
        
        # Resize face to standard size for consistency
        face_img = cv2.resize(face_img, (160, 160))
        
        self.captured_face = face_img
        self.register_button.setEnabled(True)
        self.status_label.setText("Status: ✅ Face captured successfully! Fill in student details and click Register.")
        
        print(f"[INFO] Face captured successfully. Size: {face_img.shape}, Quality score: {laplacian_var:.2f}")

    # ----------------------------
    # Register student by capturing image and saving embedding
    # ----------------------------
    def register_student(self):
        """Register student with enhanced validation and proper embedding storage"""
        name = self.name_input.text().strip()
        roll_no = self.roll_no_input.text().strip()
        
        # Enhanced validation
        if not name:
            self.status_label.setText("Status: ❌ Student name cannot be empty!")
            return
        
        if len(name) < 2:
            self.status_label.setText("Status: ❌ Student name must be at least 2 characters!")
            return
            
        if not roll_no:
            self.status_label.setText("Status: ❌ Roll number cannot be empty!")
            return
            
        if self.captured_face is None:
            self.status_label.setText("Status: ❌ Please capture face first!")
            return

        # Check if student already exists
        if self._student_exists(name):
            self.status_label.setText(f"Status: ⚠️  Student '{name}' already exists! Use a different name.")
            return

        try:
            self.status_label.setText("Status: 🔄 Processing registration...")
            
            # Create student data structure
            student_data = {
                'name': name,
                'roll_no': roll_no,
                'registration_date': __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            if self.models_available and self.facenet_model is not None:
                # Advanced registration with face embedding
                try:
                    print(f"[INFO] Generating face embedding for {name}...")
                    
                    # Generate embedding from captured face
                    embedding = get_face_embedding(self.facenet_model, self.captured_face)
                    print(f"[INFO] Generated embedding shape: {embedding.shape}")

                    # Save to embeddings database with name as key
                    self.embeddings_db[name] = embedding
                    
                    # Ensure directory exists and save embeddings
                    create_folder("data/embeddings")
                    embeddings_path = "data/embeddings/embeddings.pkl"
                    save_pickle(self.embeddings_db, embeddings_path)
                    
                    print(f"[INFO] Saved embeddings database with {len(self.embeddings_db)} entries to {embeddings_path}")
                    
                    # Also save student data separately
                    student_data['has_embedding'] = True
                    self._save_student_data(student_data)

                    print(f"[INFO] Registered {name} (Roll: {roll_no}) with face embedding successfully.")
                    self.status_label.setText(f"Status: ✅ {name} registered successfully with face recognition!")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to generate embedding: {e}")
                    self.status_label.setText(f"Status: ❌ Registration failed - {str(e)}")
                    return
            else:
                # Basic registration without face embedding
                student_data['has_embedding'] = False
                self._save_student_data(student_data)
                print(f"[INFO] Registered {name} (Roll: {roll_no}) in basic mode.")
                self.status_label.setText(f"Status: ✅ {name} registered (basic mode - no face recognition)")

            # Save captured face image for dataset reference
            create_folder(f"data/faces/{name}")
            img_path = f"data/faces/{name}/{name}.jpg"
            cv2.imwrite(img_path, self.captured_face)
            
            # Also save as pickle for consistency
            face_pkl_path = f"data/faces/{name}/{name}.pkl"
            save_pickle(self.captured_face, face_pkl_path)
            
            print(f"[INFO] Face image saved to {img_path} and {face_pkl_path}")
            
            # Show success message with instructions
            self.status_label.setText(f"Status: 🎉 {name} registered successfully! You can now close this window.")
            
            # Notify parent window about successful registration
            if self.parent_callback:
                try:
                    self.parent_callback(name)
                except Exception as e:
                    print(f"[WARNING] Failed to notify parent about registration: {e}")
            
            # Reset form for next registration
            self.name_input.clear()
            self.roll_no_input.clear()
            self.captured_face = None
            self.register_button.setEnabled(False)
            
            print(f"[SUCCESS] Registration completed for {name} (Roll: {roll_no})")
            
        except Exception as e:
            print(f"[ERROR] Registration failed: {e}")
            self.status_label.setText(f"Status: ❌ Registration failed - {str(e)}")

    def _student_exists(self, name):
        """Check if a student with this name already exists"""
        # Check in JSON file
        import json
        students_file = "data/students/students.json"
        if os.path.exists(students_file):
            try:
                with open(students_file, 'r') as f:
                    students = json.load(f)
                    if name in students:
                        return True
            except Exception as e:
                print(f"[WARNING] Could not read students file: {e}")
        
        # Check in embeddings database
        if name in self.embeddings_db:
            return True
            
        return False

    def _save_student_data(self, student_data):
        """Save student data to a JSON file"""
        import json
        
        create_folder("data/students")
        students_file = "data/students/students.json"
        
        # Load existing students data
        if os.path.exists(students_file):
            with open(students_file, 'r') as f:
                students = json.load(f)
        else:
            students = {}
        
        # Add new student
        students[student_data['name']] = student_data
        
        # Save updated data
        with open(students_file, 'w') as f:
            json.dump(students, f, indent=4)
        
        print(f"[INFO] Student data saved for {student_data['name']}")

    def get_student_info(self, name):
        """Get student information by name"""
        import json
        
        students_file = "data/students/students.json"
        if os.path.exists(students_file):
            with open(students_file, 'r') as f:
                students = json.load(f)
                return students.get(name, None)
        return None

    # ----------------------------
    # Close event - release camera
    # ----------------------------
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

# ----------------------------
# Main entry point for testing
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RegistrationWindow()
    window.show()
    sys.exit(app.exec_())
