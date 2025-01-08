import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image
import time
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set Streamlit page configuration
st.set_page_config(
    page_title="FallSense",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("⚠️ Fall Detection Using Pose Estimation")

# Sidebar for user inputs
st.sidebar.header("Upload Video")

# File uploader allows only video files
uploaded_video = st.sidebar.file_uploader(
    "Upload a video file for fall detection", type=["mp4", "avi", "mov"]
)

# Load the pre-trained fall detection model
@st.cache_resource
def load_fall_detection_model(model_path):
    model = load_model(model_path)
    return model

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to process the video
def process_video(video_path, model):
    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
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

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video file.")
        return None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))  # float width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT )) # float height

    # Define the codec and create VideoWriter object to save the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec if needed
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()

    for frame_num in range(frame_count):
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
            # If no person detected, append zeros
            coords = [0]*feature_dim

        frames_buffer.append(coords)

        # Once we have at least SEQ_LENGTH frames, start predicting
        if len(frames_buffer) >= SEQ_LENGTH:
            # Get the last SEQ_LENGTH frames
            sequence = frames_buffer[-SEQ_LENGTH:]
            sequence_array = np.array(sequence)  # shape: (SEQ_LENGTH, feature_dim)
            sequence_array = np.expand_dims(sequence_array, axis=0)  # shape: (1, SEQ_LENGTH, feature_dim)

            # Predict
            prob = model.predict(sequence_array, verbose=0)[0][0]  # Get the probability
            prediction_probs.append(prob)

            # Determine label
            label = "Fall Detected!" if prob > 0.5 else "No Fall"
            color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)

            # Calculate position for right-top placement
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10  
            text_y = 30  

            # Put label on the frame
            cv2.putText(frame, f"{label}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


        # Write the annotated frame to the output video
        out.write(frame)

        # Update progress bar
        progress = (frame_num + 1) / frame_count
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_num + 1} of {frame_count}")

    # Release resources
    cap.release()
    out.release()
    pose.close()
    progress_bar.empty()
    status_text.empty()

    return temp_output.name, prediction_probs

# Main logic
if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Load the model
    model_path = './Model/fall_detection_pose_model.h5' 
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload the model file.")
    else:
        with st.spinner('Loading fall detection model...'):
            model = load_fall_detection_model(model_path)

        # Process the video
        with st.spinner('Processing video for fall detection...'):
            output_video_path, prediction_probs = process_video(tfile.name, model)

        if output_video_path:
            # Display the output video
            st.success('Video processing completed!')
            # To ensure the video is properly read, we need to read it as bytes
            with open(output_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            # Provide a download button for the annotated video
            st.download_button(
                label="Download Video",
                data=video_bytes,
                file_name='result_fall_detection.mp4',
                mime='video/mp4'
            )

        # Clean up temporary files if needed
        os.remove(tfile.name)
        if output_video_path and os.path.exists(output_video_path):
            os.remove(output_video_path)