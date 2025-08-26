import os
import csv
from datetime import datetime

# ----------------------------
# Function to get today's date as string
# ----------------------------
def get_today_date():
    return datetime.now().strftime("%Y%m%d")

# ----------------------------
# Function to get current time as string
# ----------------------------
def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

# ----------------------------
# Function to get attendance file path
# ----------------------------
def get_attendance_file_path():
    folder_path = os.path.join("data", "attendance_logs")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = f"attendance_{get_today_date()}.csv"
    return os.path.join(folder_path, filename)

# ----------------------------
# Function to check if student is already marked present
# ----------------------------
def is_already_present(name):
    file_path = get_attendance_file_path()
    if not os.path.exists(file_path):
        return False

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0 and row[0] == name:
                return True
    return False

# ----------------------------
# Function to mark attendance
# ----------------------------
def mark_attendance(name, student_id, gps_location=None):
    file_path = get_attendance_file_path()
    attendance_entry = [name, student_id, get_current_time()]

    if gps_location:
        attendance_entry.append(gps_location)

    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file did not exist
        if not file_exists:
            header = ["Name", "Student ID", "Time"]
            if gps_location:
                header.append("GPS Location")
            writer.writerow(header)

        # Check if already present
        if not is_already_present(name):
            writer.writerow(attendance_entry)
            print(f"[INFO] Attendance marked for {name}")
            return True
        else:
            print(f"[INFO] {name} already marked present today.")
            return False

# ----------------------------
# Function to get session-specific attendance
# ----------------------------
def get_session_attendance(session_id=None):
    """Get attendance for a specific session or today"""
    file_path = get_attendance_file_path()
    if not os.path.exists(file_path):
        return []
    
    attendance_list = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 3:
                attendance_list.append({
                    'name': row[0],
                    'student_id': row[1],
                    'time': row[2],
                    'location': row[3] if len(row) > 3 else None
                })
    return attendance_list

# ----------------------------
# Function to check if boarding session is active
# ----------------------------
def is_boarding_active():
    """Check if any boarding session is currently active"""
    # This could be enhanced to check a database or config file
    # For now, return True if attendance file exists for today
    return os.path.exists(get_attendance_file_path())

# ----------------------------
# Function to get attendance statistics
# ----------------------------
def get_attendance_stats():
    """Get attendance statistics for today"""
    attendance_list = get_session_attendance()
    return {
        'total_students': len(attendance_list),
        'attendance_times': [entry['time'] for entry in attendance_list],
        'students_list': [entry['name'] for entry in attendance_list]
    }

# ----------------------------
# Example usage (testing)
# ----------------------------
if __name__ == "__main__":
    # Test attendance marking
    mark_attendance("John Doe", "S123")
    mark_attendance("Jane Smith", "S456", gps_location="Lat:12.9716, Long:77.5946")
