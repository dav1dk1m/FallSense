import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set Streamlit page configuration
st.set_page_config(
    page_title="FallSense - Real-Time",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("⚠️ Fall Detection Using Real-Time Pose Estimation")

# Sidebar for user inputs
st.sidebar.header("Real-Time Fall Detection")

# Load the pre-trained fall detection model
@st.cache_resource
def load_fall_detection_model(model_path):
    model = load_model(model_path)
    return model

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_webcam(model):
    # Initialize Mediapipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.3,
        model_complexity=1
    )

    SEQ_LENGTH = 30
    num_landmarks = 33  # Mediapipe Pose has 33 landmarks
    feature_dim = num_landmarks * 3  # x, y, z for each landmark

    frames_buffer = []
    prediction_probs = []

    # Open webcam video stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error accessing webcam.")
        return

    stframe = st.empty()  # Create a placeholder for the video stream

    # Add a "Stop" button
    stop_button = st.sidebar.button("Stop Webcam", key="stop_button")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract pose landmarks
            landmarks = result.pose_landmarks.landmark
            coords = []
            for lm in landmarks:
                coords.extend([lm.x, lm.y, lm.z])
        else:
            coords = [0] * feature_dim  # Append zeros if no landmarks are detected

        frames_buffer.append(coords)

        # Once we have at least SEQ_LENGTH frames, start predicting
        if len(frames_buffer) >= SEQ_LENGTH:
            sequence = frames_buffer[-SEQ_LENGTH:]
            sequence_array = np.array(sequence)
            sequence_array = np.expand_dims(sequence_array, axis=0)

            # Predict
            prob = model.predict(sequence_array, verbose=0)[0][0]
            prediction_probs.append(prob)

            # Determine label
            label = "Fall Detected!" if prob > 0.5 else "No Fall"
            color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)

            # Put label on the frame
            cv2.putText(frame, f"{label} ({prob:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Check if the "Stop" button was clicked
        if stop_button:
            break

    # Release resources
    cap.release()
    pose.close()

# Main logic
st.sidebar.write("Click below to start real-time processing:")
if st.sidebar.button("Start Webcam", key="start_button"):
    # Load the model
    model_path = './Model/fall_detection_pose_model.h5' 
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload the model file.")
    else:
        with st.spinner('Loading fall detection model...'):
            model = load_fall_detection_model(model_path)

        st.write("Starting webcam... press **Stop Webcam** in the sidebar to end.")
        process_webcam(model)

