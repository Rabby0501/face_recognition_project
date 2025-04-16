import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from face_utils import (  # Changed from "face.utils"
    load_known_faces, save_known_faces, load_saved_faces,
    recognize_faces_in_image, mark_attendance, real_time_recognition  # Fixed spelling
) 
import os
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# Initialize session state
if 'known_encodings' not in st.session_state:
    st.session_state.known_encodings = []
if 'known_names' not in st.session_state:
    st.session_state.known_names = []

# Load known faces
st.session_state.known_encodings, st.session_state.known_names = load_saved_faces()

# Sidebar
st.sidebar.title("Settings")

# Function to handle video frames
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Face recognition processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(st.session_state.known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            name = "Unknown"
            confidence = 0
            threshold = 0.6
            
            if len(face_distances) > 0 and face_distances[best_match_index] < threshold:
                name = st.session_state.known_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                mark_attendance(name)
            
            # Draw rectangle and label
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(img, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main App
st.title("Face Recognition System 👤")

tab1, tab2, tab3, tab4 = st.tabs(["Real-time Recognition", "Image Recognition", "Manage Faces", "Attendance Records"])

with tab1:
    st.header("Real-time Face Recognition")
    st.write("Using your webcam for live face recognition")
    
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with tab2:
    st.header("Recognize Faces in Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the image
        image, names = recognize_faces_in_image(
            "temp.jpg",
            st.session_state.known_encodings,
            st.session_state.known_names
        )
        
        # Display results
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Recognized Faces")
        
        if names:
            st.success(f"Recognized: {', '.join(names)}")
        else:
            st.warning("No known faces recognized")

with tab3:
    st.header("Manage Known Faces")
    
    # Add new face
    st.subheader("Add New Face")
    new_face_name = st.text_input("Person's Name")
    new_face_image = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    
    if st.button("Add Face") and new_face_name and new_face_image:
        # Save the uploaded file to known_faces directory
        os.makedirs("known_faces", exist_ok=True)
        filename = f"{new_face_name}.{new_face_image.name.split('.')[-1]}"
        filepath = os.path.join("known_faces", filename)
        
        with open(filepath, "wb") as f:
            f.write(new_face_image.getbuffer())
        
        # Reload known faces
        st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
        save_known_faces(st.session_state.known_encodings, st.session_state.known_names)
        st.success(f"Added {new_face_name} to known faces!")
    
    # List known faces
    st.subheader("Current Known Faces")
    if st.session_state.known_names:
        st.write(pd.DataFrame({"Name": st.session_state.known_names}))
    else:
        st.warning("No known faces in database")

with tab4:
    st.header("Attendance Records")
    
    # List available attendance files
    if os.path.exists("attendance"):
        attendance_files = os.listdir("attendance")
        selected_file = st.selectbox("Select date", attendance_files)
        
        if selected_file:
            filepath = os.path.join("attendance", selected_file)
            df = pd.read_csv(filepath, names=["Name", "Time"])
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                f"attendance_{selected_file.split('_')[-1]}",
                "text/csv"
            )
    else:
        st.warning("No attendance records yet")