import cv2
import os

dataset_path = "dataset_images"  # Your dataset folder
output_path = "output_images"    # Folder to save processed images

# Create output folder if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Loop through each person in dataset
for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue

    # Process each image in the person's folder
    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not read {filename}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Create person's folder in output folder
            person_output = os.path.join(output_path, person)
            if not os.path.exists(person_output):
                os.makedirs(person_output)

            # Save processed image
            output_file = os.path.join(person_output, filename)
            cv2.imwrite(output_file, image)

            print(f"Processed {filename}, detected {len(faces)} face(s)")

print("Face detection completed! Check 'output_images' folder.")

