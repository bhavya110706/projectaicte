import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Path to the folder containing known face images
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Function to load known faces and their encodings
def load_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_encodings, known_names

# Function to mark attendance in a CSV file
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Create a DataFrame if the file doesn't exist
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    # Read existing attendance file
    df = pd.read_csv(ATTENDANCE_FILE)

    # Check if the name is already recorded for today
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_entry = {"Name": name, "Date": date, "Time": time}
        df = df.append(new_entry, ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

# Load known faces
print("Loading known faces...")
known_encodings, known_names = load_known_faces()
print(f"Loaded {len(known_names)} known faces.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)
print("Starting video capture. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = small_frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        # Find the best match
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Mark attendance
        if name != "Unknown":
            mark_attendance(name)

        # Draw a box around the face
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
