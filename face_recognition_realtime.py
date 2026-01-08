import face_recognition
import cv2
import os
import numpy as np

# Path to cropped faces dataset
DATASET_PATH = "cropped_faces"

known_encodings = []
known_names = []

print("[INFO] Loading face dataset...")

# Load and encode all known faces
for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(person_dir, img_name)

            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

print(f"[INFO] Loaded {len(known_encodings)} face samples.")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    exit()

print("[INFO] Multi-face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )

        name = "Unknown"

        if known_encodings:
            face_distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Scale coordinates back to original frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Multi Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

