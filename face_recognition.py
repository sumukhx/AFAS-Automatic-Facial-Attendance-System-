from ultralytics import YOLO
from ultralytics.engine.results import Results  
from deepface import DeepFace
from PIL import Image
import shutil
import cv2
import os
import csv
import datetime
import time
import base64
import threading
from fpdf import FPDF


def faceRecognition(face_images, threshold=0.6):
    unknown_faces_dir = "./unknown/"
    known_faces_dir = "./known/"
    extracted_names = []
    confidences = []

    if not os.path.exists(unknown_faces_dir):
        os.makedirs(unknown_faces_dir)
    else:
        for file_or_folder in os.listdir(unknown_faces_dir):
            file_or_folder_path = os.path.join(unknown_faces_dir, file_or_folder)
            if os.path.isfile(file_or_folder_path):
                os.remove(file_or_folder_path)
            elif os.path.isdir(file_or_folder_path):
                shutil.rmtree(file_or_folder_path)

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    else:
        for file_or_folder in os.listdir(known_faces_dir):
            file_or_folder_path = os.path.join(known_faces_dir, file_or_folder)
            if os.path.isfile(file_or_folder_path):
                os.remove(file_or_folder_path)
            elif os.path.isdir(file_or_folder_path):
                shutil.rmtree(file_or_folder_path)

    for i, face_image in enumerate(face_images):
        img_path = face_image
        model = DeepFace.find(img_path=img_path, db_path="database", enforce_detection=False, model_name="Facenet512")

        if model and len(model[0]['identity']) > 0:
            distance = model[0]['distance'][0]
            if distance < threshold:
                name = model[0]['identity'][0].split('/')[1]
                known_faces_path = os.path.join(known_faces_dir, f"{i + 1}_{name}.jpg")
                shutil.copy(img_path, known_faces_path)
                extracted_names.append(name)
                confidences.append(1 - distance)  # Convert distance to confidence (higher is better)
            else:
                name = 'unknown'
                extracted_names.append(name)
                confidences.append(None)
        else:
            name = 'unknown'
            confidence = None
            unknown_faces_path = os.path.join(unknown_faces_dir, f"{i + 1}.jpg")
            shutil.copy(img_path, unknown_faces_path)
            extracted_names.append(name)
            confidences.append(confidence)
            
    return extracted_names, confidences

def faceExtraction(input_image, model, results):
    image = Image.open(input_image)
    detected_objects = []

    if hasattr(results, 'boxes') and hasattr(results, 'names'):
        for box in results.boxes.xyxy:
            object_id = int(box[-1])
            object_name = results.names.get(object_id)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            detected_objects.append((object_name, (x1, y1, x2, y2)))

    if os.path.exists("faces"):
        shutil.rmtree("faces")
    os.makedirs("faces")

    face_images = []
    for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):
        object_image = image.crop((x1, y1, x2, y2))
        face_path = f"faces/face{i}.jpg"
        object_image.save(face_path)
        face_images.append(face_path)
        
    return detected_objects, face_images

def faceDetection(input_image):
    model = YOLO('best.pt')
    results: Results = model.predict(input_image)[0]
    return faceExtraction(input_image, model, results)

def save_to_csv(name, manager, group):
    print(f"Saving to CSV: Name={name}, Manager={manager}, Group={group}")  # Debugging log
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"{date_str}.csv"
    fieldnames = ['Name', 'Manager', 'Group', 'Timestamp']

    # Check if the file exists and read existing records
    records_to_write = []
    file_exists = os.path.isfile(csv_filename)
    if file_exists:
        with open(csv_filename, mode='r') as file:
            reader = csv.DictReader(file)
            existing_records = [row['Name'] for row in reader]
            # Check if the name already exists in the records
            if name in existing_records:
                print(f"Record for {name} already exists for today. Skipping.")
                return  # Skip writing this record if it already exists

    # Append the new record to the CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Name': name,
            'Manager': manager,
            'Group': group,
            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    
def update_names_and_details(name, manager, group):
    details_file = "names_and_details.txt"

    with open(details_file, 'a') as file:
        file.write(f"Name: {name}\n")
        file.write(f"Manager: {manager}\n")
        file.write(f"Group: {group}\n")
        file.write("\n")  # Blank line for separation

def capture_and_recognize():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    recognized_faces = {}
    start_times = {}

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image from webcam")
            break

        frame_path = "current_frame.jpg"
        cv2.imwrite(frame_path, frame)

        detected_faces, face_images = faceDetection(frame_path)

        if detected_faces:
            names, confidences = faceRecognition(face_images)
            for name, confidence in zip(names, confidences):
                if name != 'unknown':
                    if name not in recognized_faces:
                        recognized_faces[name] = 0
                        start_times[name] = time.time()
                    else:
                        if time.time() - start_times[name] >= 4:
                            if recognized_faces[name] == 0:
                                recognized_faces[name] += 1
                                # Read the details from the text file
                                face_dir = os.path.join("database", name)
                                with open(os.path.join(face_dir, 'details.txt'), 'r') as f:
                                    details = f.readlines()
                                    manager = details[0].strip().split(": ")[1]
                                    group = details[1].strip().split(": ")[1]
                                save_to_csv(name, manager, group)

            # Draw bounding boxes and labels
            for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_faces):
                color = (0, 255, 0) if names[i] != 'unknown' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{names[i]} ({confidences[i]:.2f})" if confidences[i] is not None else names[i]
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cam.release()
    cv2.destroyAllWindows()

def add_new_face(name, manager, group):
    save_face_images(name, manager, group)
    update_names_and_details(name, manager, group)

def get_attendance_records():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"{date_str}.csv"
    attendance_records = []

    print(f"Checking for file: {csv_filename}")  # Debug print

    if os.path.exists(csv_filename):
        print(f"File {csv_filename} exists")  # Debug print
        with open(csv_filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                print(f"Read row: {row}")  # Debug print
                attendance_records.append(row)
    else:
        print(f"File {csv_filename} does not exist")  # Debug print

    print(f"Attendance records: {attendance_records}")  # Debug print
    return attendance_records


def save_captured_image(name, image_data, capture_count, manager=None, group=None):
    face_dir = os.path.join("database", name)
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
        # Save the details in a text file if manager and group are provided
        if manager and group:
            with open(os.path.join(face_dir, 'details.txt'), 'w') as f:
                f.write(f"Manager: {manager}\n")
                f.write(f"Group: {group}\n")

    image_path = os.path.join(face_dir, f"{name}_{capture_count:02d}.jpg")
    with open(image_path, "wb") as img_file:
        img_file.write(image_data)
    print(f"Saved {image_path}")
    
def add_new_pass(name, manager, group, days_valid, purpose):
    current_date = datetime.datetime.now()
    expiry_date = current_date + datetime.timedelta(days=days_valid)

    face_dir = os.path.join("database", name)
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
        
    # Save the details in a text file
    with open(os.path.join(face_dir, 'details.txt'), 'w') as f:
        f.write(f"Manager: {manager}\n")
        f.write(f"Group: {group}\n")
        f.write(f"Purpose: {purpose}\n")
        f.write(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Capture 4 images
    capture_images(name)

    # Generate the pass document
    generate_pass_document(name, manager, group, purpose, expiry_date)

    # Schedule the deletion of the pass
    schedule_pass_deletion(name, expiry_date)

def generate_pass_document(name, manager, group, purpose, expiry_date):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Visitor Pass", ln=True, align="C")
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Manager: {manager}", ln=True)
    pdf.cell(200, 10, txt=f"Group: {group}", ln=True)
    pdf.cell(200, 10, txt=f"Purpose: {purpose}", ln=True)
    pdf.cell(200, 10, txt=f"Expiry Date: {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Add one of the captured images
    image_path = os.path.join("database", name, f"{name}_01.jpg")
    if os.path.exists(image_path):
        pdf.image(image_path, x=10, y=pdf.get_y(), w=100)
    
    # Save the PDF
    pdf_output_path = os.path.join("database", name, "pass_document.pdf")
    pdf.output(pdf_output_path)
    print(f"Pass document created at {pdf_output_path}")

def capture_images(name):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    
    face_dir = os.path.join("database", name)
    for i in range(4):
        ret, frame = cam.read()
        if ret:
            image_path = os.path.join(face_dir, f"{name}_{i+1:02d}.jpg")
            cv2.imwrite(image_path, frame)
            time.sleep(1)  # Slight delay between captures
    cam.release()

def schedule_pass_deletion(name, expiry_date):
    current_time = datetime.datetime.now()
    wait_time = (expiry_date - current_time).total_seconds()

    # Use a separate thread to wait and delete the data
    threading.Timer(wait_time, delete_pass_data, [name]).start()

def delete_pass_data(name):
    face_dir = os.path.join("database", name)
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)
    print(f"Data for {name} deleted after pass expiry.")
    
    

    
    
    