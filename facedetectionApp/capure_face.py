import cv2
import os

# Dataset folder
dataset_path = "dataset_images"

# Ask for person's name
person_name = input("Enter your name: ")

# Create folder for this person
person_folder = os.path.join(dataset_path, person_name)
if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(person_folder, f"{count}.jpg"), face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    # Quit with 'q' or after 50 images
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {person_name}.")

