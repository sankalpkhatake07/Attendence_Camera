import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
import csv

DATASET_PATH = "facedetectionApp/cropped_faces"
ATTENDANCE_PATH = "attendance/attendance.csv"

known_encodings = []
known_names = []

print("[INFO] Loading face dataset...")

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    for img in os.listdir(person_dir):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(person_dir, img)
            image = face_recognition.load_image_file(img_path)
            encs = face_recognition.face_encodings(image)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(person)

print(f"[INFO] {len(known_encodings)} face samples loaded.")

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_PATH):
    with open(ATTENDANCE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked_students = set()

cap = cv2.VideoCapture(0)
print("[INFO] Attendance system started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), face_encoding in zip(locations, encodings):

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )
        name = "Unknown"

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match = np.argmin(distances)
            if matches[best_match]:
                name = known_names[best_match]

        # Scale back
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mark attendance
        if name != "Unknown" and name not in marked_students:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            with open(ATTENDANCE_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, date, time])

            marked_students.add(name)
            print(f"[ATTENDANCE] Marked: {name}")

    cv2.imshow("Student Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

