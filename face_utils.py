import face_recognition
import cv2
import numpy as np
import os
import pickle
from datetime import datetime

def load_known_faces(known_faces_dir="known_faces"): 
    """
    Load known faces from directory and create encodings
    Returns: (known_face_encodings, known_face_names)
    """
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings (assuming one face per image)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                encoding = encodings[0]  # Take first face found
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])
    
    return known_face_encodings, known_face_names

def save_known_faces(encodings, names, filename="face_database.dat"):
    """Save face encodings to a file"""
    with open(filename, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

def load_saved_faces(filename="face_database.dat"):
    """Load saved face encodings"""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    except FileNotFoundError:
        return [], []

def recognize_faces_in_image(image_path, known_encodings, known_names, threshold=0.6):
    """
    Recognize faces in an image file
    Returns: image with annotations, names of recognized faces
    """
    unknown_image = face_recognition.load_image_file(image_path)
    
    # Find faces
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    # Convert to OpenCV format
    image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)
    recognized_names = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        name = "Unknown"
        confidence = 0
        
        if face_distances[best_match_index] < threshold:
            name = known_names[best_match_index]
            confidence = 1 - face_distances[best_match_index]
            recognized_names.append(name)
        
        # Draw rectangle and label
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} ({confidence:.2f})"
        cv2.putText(image, label, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    return image, recognized_names

def mark_attendance(name, attendance_dir="attendance"):
    """Record attendance with timestamp"""
    os.makedirs(attendance_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(attendance_dir, f"attendance_{today}.csv")
    
    with open(filename, "a") as f:
        timestamp = datetime.now().strftime("%H:%M:%S")
        f.write(f"{name},{timestamp}\n")

def real_time_recognition(known_encodings, known_names, threshold=0.6):
    """Real-time face recognition using webcam"""
    video_capture = cv2.VideoCapture(0)
    recognized_names = set()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Process every other frame to improve performance
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare faces
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            name = "Unknown"
            confidence = 0
            
            if face_distances[best_match_index] < threshold:
                name = known_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                
                if name not in recognized_names:
                    mark_attendance(name)
                    recognized_names.add(name)
            
            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()