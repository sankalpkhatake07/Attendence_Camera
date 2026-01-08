import cv2
import os

# Path to your dataset
dataset_path = "dataset_images"
# Folder to save cropped faces
faces_path = "cropped_faces"

# Create folder if it doesn't exist
if not os.path.exists(faces_path):
    os.makedirs(faces_path)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Loop through each person
for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue

    # Create folder for this person's cropped faces
    person_faces_folder = os.path.join(faces_path, person)
    if not os.path.exists(person_faces_folder):
        os.makedirs(person_faces_folder)

    # Process each image
    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not read {filename}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for i, (x, y, w, h) in enumerate(faces):
                face_img = image[y:y+h, x:x+w]
                face_file = os.path.join(person_faces_folder, f"{os.path.splitext(filename)[0]}_face{i+1}.jpg")
                cv2.imwrite(face_file, face_img)

            print(f"Processed {filename}, detected {len(faces)} face(s)")

print("Face cropping completed! Check 'cropped_faces' folder.")

