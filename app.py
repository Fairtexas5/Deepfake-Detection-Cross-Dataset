import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import streamlit as st
import cv2
import torch
import numpy as np
import os
import tempfile
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import deque
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import urllib.request
import shutil

class CNNDeepfakeDetector:
    def __init__(self):
        st.info("Initializing CNN Deepfake Detector... This may take a moment.")

        # Initialize CNN model for deepfake detection
        with st.spinner("Loading CNN deepfake detection model..."):
            try:
                self.model = load_model('cnn_model.h5')
                st.success("CNN model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading CNN model: {e}")
                st.warning("Please make sure 'cnn_model.h5' is in the current directory.")
                self.model = None

    def classify_image(self, face_img):
        """Classify a face image as real or fake using CNN model"""
        try:
            if self.model is None:
                return "Model Not Loaded", 0.0

            # Resize to target size
            img_resized = cv2.resize(face_img, (128, 128))

            # Preprocess the image
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = self.model.predict(img_array)
            confidence = float(prediction[0][0])

            # In this model, <0.5 means Real, >=0.5 means Fake
            label = 'Real' if confidence < 0.5 else 'Fake'

            # Adjust confidence to be relative to the prediction
            if label == 'Fake':
                confidence = confidence  # Already between 0.5-1.0
            else:
                confidence = 1.0 - confidence  # Convert 0.0-0.5 to 0.5-1.0

            return label, confidence

        except Exception as e:
            st.error(f"Error in CNN classification: {e}")
            return "Error", 0.0

class DeepfakeDetector:
    def __init__(self):
        st.info("Initializing Deepfake Detector... This may take a moment.")

        # Initialize ViT model for deepfake detection
        with st.spinner("Loading deepfake detection model..."):
            self.image_processor = AutoImageProcessor.from_pretrained(
                'Adieee5/deepfake-detection-f3net-cross')
            self.model = AutoModelForImageClassification.from_pretrained(
                'Adieee5/deepfake-detection-f3net-cross')

        # Face detection model setup
        with st.spinner("Loading face detection model..."):
            model_file = "deploy.prototxt"
            weights_file = "res10_300x300_ssd_iter_140000.caffemodel"

            self.use_dnn = False
            if os.path.exists(model_file) and os.path.exists(weights_file):
                try:
                    self.face_net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
                    self.use_dnn = True
                    st.success("Using DNN face detector (better for close-up faces)")
                except Exception as e:
                    st.warning(f"Could not load DNN model: {e}")
                    self.use_dnn = False

            if not self.use_dnn:
                # Fallback to Haar cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    st.warning("Using Haar cascade face detector as fallback")
                else:
                    st.error(f"Cascade file not found: {cascade_path}")

        # Initialize CNN model
        self.cnn_detector = CNNDeepfakeDetector()

        # Face tracking/smoothing parameters
        self.face_history = {}  # Store face tracking data
        self.face_history_max_size = 10  # Store history for last 10 frames
        self.face_ttl = 5  # Number of frames a face can be missing before removing
        self.next_face_id = 0  # For assigning unique IDs to tracked faces

        # Result smoothing
        self.result_buffer_size = 5  # Number of classifications to average

        # Performance metrics
        self.processing_times = deque(maxlen=30)

        st.success("Models loaded successfully!")

    def detect_faces_haar(self, frame):
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert to list of (x,y,w,h,confidence) format for consistency
        return [(x, y, w, h, 0.8) for (x, y, w, h) in faces]

    def classify_frame(self, face_img, model_type="vit"):
        """Classify a face image as real or fake"""
        try:
            if model_type == "cnn":
                return self.cnn_detector.classify_image(face_img)

            # Default to ViT model
            # Resize image if too small
            h, w = face_img.shape[:2]
            if h < 224 or w < 224:
                scale = max(224/h, 224/w)
                face_img = cv2.resize(face_img, (int(w*scale), int(h*scale)))

            # Make sure we have valid image data
            if face_img.size == 0:
                return "Unknown", 0.0

            # Process with ViT model
            inputs = self.image_processor(images=face_img, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Get prediction and confidence
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()

            # The model has two classes: 0=Fake, 1=Real
            label = 'Real' if pred == 1 else 'Fake'
            confidence = probs[0][pred].item()

            return label, confidence

        except Exception as e:
            st.error(f"Error in classification: {e}")
            return "Error", 0.0

    def detect_faces_dnn(self, frame):
        """Detect faces using DNN method"""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Filter out weak detections
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")
                # Ensure box is within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:  # Valid face area
                    faces.append((x1, y1, w, h, confidence))

        return faces

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
        box1_x1, box1_y1, box1_w, box1_h = box1
        box2_x1, box2_y1, box2_w, box2_h = box2

        box1_x2, box1_y2 = box1_x1 + box1_w, box1_y1 + box1_h
        box2_x2, box2_y2 = box2_x1 + box2_w, box2_y1 + box2_h

        # Calculate area of intersection rectangle
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of both boxes
        box1_area = box1_w * box1_h
        box2_area = box2_w * box2_h

        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def track_faces(self, faces):
        matched_faces = []
        unmatched_detections = list(range(len(faces)))

        if not self.face_history:
            for face in faces:
                face_id = self.next_face_id
                self.next_face_id += 1
                self.face_history[face_id] = {
                    'positions': deque([face[:4]], maxlen=self.face_history_max_size),
                    'ttl': self.face_ttl,
                    'label': None,
                    'confidence': 0.0,
                    'result_history': deque(maxlen=self.result_buffer_size)
                }
                matched_faces.append((face_id, face))
            return matched_faces

        for face_id in list(self.face_history.keys()):
            last_pos = self.face_history[face_id]['positions'][-1]
            best_match = -1
            best_iou = 0.3
            for i in unmatched_detections:
                iou = self.calculate_iou(last_pos, faces[i][:4])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            if best_match != -1:
                matched_face = faces[best_match]
                self.face_history[face_id]['positions'].append(matched_face[:4])
                self.face_history[face_id]['ttl'] = self.face_ttl
                matched_faces.append((face_id, matched_face))
                unmatched_detections.remove(best_match)
            else:
                self.face_history[face_id]['ttl'] -= 1
                if self.face_history[face_id]['ttl'] <= 0:
                    del self.face_history[face_id]
                else:
                    predicted_face = (*last_pos, 0.5)
                    matched_faces.append((face_id, predicted_face))

        for i in unmatched_detections:
            face_id = self.next_face_id
            self.next_face_id += 1
            self.face_history[face_id] = {
                'positions': deque([faces[i][:4]], maxlen=self.face_history_max_size),
                'ttl': self.face_ttl,
                'label': None,
                'confidence': 0.0,
                'result_history': deque(maxlen=self.result_buffer_size)
            }
            matched_faces.append((face_id, faces[i]))

        return matched_faces

    def smooth_face_position(self, face_id):
        """Calculate smoothed position for a tracked face"""
        positions = self.face_history[face_id]['positions']

        if len(positions) == 1:
            return positions[0]

        # Weight recent positions more heavily
        total_weight = 0
        x, y, w, h = 0, 0, 0, 0

        for i, pos in enumerate(positions):
            # Exponential weighting - newer positions have more influence
            weight = 2 ** i  # Positions are stored newest to oldest
            total_weight += weight

            x += pos[0] * weight
            y += pos[1] * weight
            w += pos[2] * weight
            h += pos[3] * weight

        # Calculate weighted average
        x = int(x / total_weight)
        y = int(y / total_weight)
        w = int(w / total_weight)
        h = int(h / total_weight)

        return (x, y, w, h)

    def update_face_classification(self, face_id, label, confidence):
        """Update classification history for a face"""
        self.face_history[face_id]['result_history'].append((label, confidence))

        # Calculate the smoothed result
        if not self.face_history[face_id]['result_history']:
            return label, confidence

        real_votes = 0
        fake_votes = 0
        total_confidence = 0.0

        for result_label, result_conf in self.face_history[face_id]['result_history']:
            if result_label == "Real":
                real_votes += 1
                total_confidence += result_conf
            elif result_label == "Fake":
                fake_votes += 1
                total_confidence += result_conf

        # Determine majority vote
        if real_votes >= fake_votes:
            smoothed_label = "Real"
            label_confidence = real_votes / len(self.face_history[face_id]['result_history'])
        else:
            smoothed_label = "Fake"
            label_confidence = fake_votes / len(self.face_history[face_id]['result_history'])

        # Average confidence weighted by vote consistency
        avg_confidence = (total_confidence / len(self.face_history[face_id]['result_history'])) * label_confidence

        # Store the smoothed result
        self.face_history[face_id]['label'] = smoothed_label
        self.face_history[face_id]['confidence'] = avg_confidence

        return smoothed_label, avg_confidence

    def process_video(self, video_path, stframe, status_text, progress_bar, detector_type="dnn", model_type="vit"):
        """Process video with Streamlit output"""
        use_dnn_current = detector_type == "dnn" and self.use_dnn

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error: Cannot open video source")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = 250 if video_path != 0 else 0

        # Display video info
        if video_path != 0:  # If not webcam
            status_text.text(f"Video Info: {frame_width}x{frame_height}, {fps:.1f} FPS, {total_frames} frames")
        else:
            status_text.text(f"Webcam: {frame_width}x{frame_height}")

        # Reset tracking data for new video
        self.face_history = {}
        self.next_face_id = 0
        self.processing_times = deque(maxlen=30)

        frame_count = 0
        process_every_n_frames = 2  # Process every 2nd frame for better performance

        # For face detection stats
        face_stats = {"Real": 0, "Fake": 0, "Unknown": 0}

        # Main processing loop
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                status_text.text("End of video reached")
                break

            frame_count += 1

            if frame_count == 250:
                st.success("Video Processed Successfully!")
                break

            if video_path != 0:  # If not webcam, update progress
                progress = min(float(frame_count) / float(max(total_frames, 1)), 1.0)
                progress_bar.progress(progress)

            process_frame = (frame_count % process_every_n_frames == 0)

            # Store original frame for face extraction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if process_frame:
                # Detect faces using the appropriate method
                if use_dnn_current:
                    faces = self.detect_faces_dnn(frame)
                else:
                    faces = self.detect_faces_haar(frame)

                # Track faces across frames
                tracked_faces = self.track_faces(faces)

                # Process each tracked face
                for face_id, (x, y, w, h, face_confidence) in tracked_faces:
                    if face_id not in self.face_history:
                        continue

                    sx, sy, sw, sh = self.smooth_face_position(face_id)
                    # Draw rectangle around face with smoothed coordinates
                    cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

                    # Only process classification for real detections (not predicted)
                    if w > 20 and h > 20 and face_id in self.face_history:
                        try:
                            # Extract face using smoothed coordinates for better consistency
                            face = frame_rgb[sy:sy+sh, sx:sx+sw]

                            # Skip processing if face is too small after smoothing
                            if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                                continue

                            # Process only every N frames or if this is a new face
                            if frame_count % process_every_n_frames == 0 or \
                               len(self.face_history[face_id]['result_history']) == 0:
                                # Classify the face using the selected model
                                label, confidence = self.classify_frame(face, model_type)

                                # Update and smooth results
                                label, confidence = self.update_face_classification(face_id, label, confidence)
                            else:
                                # Use last stored result
                                label = self.face_history[face_id]['label'] or "Unknown"
                                confidence = self.face_history[face_id]['confidence']

                            # Update stats
                            if label in face_stats:
                                face_stats[label] += 1

                            # Display results
                            result_text = f"{label}: {confidence:.2f}"
                            text_color = (0, 255, 0) if label == "Real" else (0, 0, 255)

                            # Add text background for better visibility
                            cv2.rectangle(frame, (sx, sy+sh), (sx+len(result_text)*11, sy+sh+25), (0, 0, 0), -1)
                            cv2.putText(frame, result_text, (sx, sy+sh+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                            # Draw face ID
                            cv2.putText(frame, f"ID:{face_id}", (sx, sy-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        except Exception as e:
                            st.error(f"Error processing face: {e}")

            # Measure processing time
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            avg_time = sum(self.processing_times) / len(self.processing_times)
            effective_fps = 1.0 / avg_time if avg_time > 0 else 0

            # Add frame counter and progress
            if video_path != 0:  # If not webcam
                progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames} ({progress_percent:.1f}%)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"Frame: {frame_count}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show detector info and performance
            detector_name = "DNN" if use_dnn_current else "Haar Cascade"
            model_name = "ViT" if model_type == "vit" else "CNN"
            cv2.putText(frame, f"Detector: {detector_name} | Model: {model_name} | FPS: {effective_fps:.1f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show tracking info
            cv2.putText(frame, f"Tracked faces: {len(self.face_history)}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display the frame in Streamlit
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Update stats
            status_text.text(f"Real: {face_stats['Real']} | Fake: {face_stats['Fake']} | FPS: {effective_fps:.1f}")

            # Check if stop button is pressed
            if st.session_state.get('stop_button', False):
                break

        # Clean up
        cap.release()
        return face_stats

# Function to ensure sample video exists
def ensure_sample_video():
    sample_dir = "sample_videos"
    sample_path = os.path.join(sample_dir, "Sample.mp4")

    # Create directory if it doesn't exist
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # If sample video doesn't exist, download it
    if not os.path.exists(sample_path):
        try:
            with st.spinner("Downloading sample video..."):
                # URL to a public domain sample video that contains faces
                sample_url = "https://storage.googleapis.com/deepfake-demo/sample_deepfake.mp4"

                # Download the file
                with urllib.request.urlopen(sample_url) as response, open(sample_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

                st.success("Sample video downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download sample video: {e}")
            return None

    return sample_path

def main():
    st.set_page_config(page_title="Deepfake Detector", layout="wide")

    # App title and description
    st.title("Deepfake Detection App")
    st.markdown("""
    This app uses computer vision and deep learning to detect deepfake videos.
    Upload a video or use your webcam to detect if faces are real or manipulated.
    """)

    # Initialize session state for the detector and variables
    if 'detector' not in st.session_state:
        st.session_state.detector = None

    if 'stop_button' not in st.session_state:
        st.session_state.stop_button = False

    if 'use_sample' not in st.session_state:
        st.session_state.use_sample = False

    if 'sample_path' not in st.session_state:
        st.session_state.sample_path = None

    # Initialize the detector
    if st.session_state.detector is None:
        st.session_state.detector = DeepfakeDetector()

    # Create sidebar for options
    st.sidebar.title("Options")

    input_option = st.sidebar.radio(
        "Select Input Source",
        ["Upload Video", "Use Webcam", "Try Sample Video"]
    )

    detector_type = st.sidebar.selectbox(
        "Face Detector",
        ["DNN (better for close-ups)", "Haar Cascade (faster)"],
        index=0 if st.session_state.detector.use_dnn else 1
    )
    detector_option = "dnn" if "DNN" in detector_type else "haar"

    # Model selection option
    model_type = st.sidebar.selectbox(
        "Deepfake Detection Model",
        ["Vision Transformer (ViT)", "F3 Net Model"],
        index=0
    )
    model_option = "vit" if "Vision" in model_type else "cnn"

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Video display area
        video_placeholder = st.empty()

    with col2:
        # Status and controls
        status_text = st.empty()
        progress_bar = st.empty()

        # Results section
        st.subheader("Results")
        results_area = st.empty()

        # Stop button
        if st.button("Stop Processing"):
            st.session_state.stop_button = True

    # Process based on selected option
    if input_option == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_file is not None:
            st.session_state.stop_button = False

            # Save uploaded file to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            # Process the video
            face_stats = st.session_state.detector.process_video(
                video_path,
                video_placeholder,
                status_text,
                progress_bar,
                detector_option,
                model_option
            )

            # Display results
            results_df = {
                "Category": ["Real Faces", "Fake Faces"],
                "Count": [face_stats["Real"], face_stats["Fake"]]
            }
            results_area.dataframe(results_df)

            # Clean up temp file
            os.unlink(video_path)

    elif input_option == "Use Webcam":
        # Reset stop button
        st.session_state.stop_button = False

        if st.sidebar.button("Start Webcam"):
            # Process webcam feed
            face_stats = st.session_state.detector.process_video(
                0,  # 0 is the default camera
                video_placeholder,
                status_text,
                progress_bar,
                detector_option,
                model_option
            )

            # Display results after stopping
            results_df = {
                "Category": ["Real Faces", "Fake Faces"],
                "Count": [face_stats["Real"], face_stats["Fake"]]
            }
            results_area.dataframe(results_df)

    elif input_option == "Try Sample Video":
        # Reset stop button
        st.session_state.stop_button = False

        # Get or download the sample video
        sample_path = ensure_sample_video()

        if sample_path:
            if st.sidebar.button("Process Sample Video"):
                # Process the sample video
                face_stats = st.session_state.detector.process_video(
                    sample_path,
                    video_placeholder,
                    status_text,
                    progress_bar,
                    detector_option,
                    model_option
                )

                # Display results
                results_df = {
                    "Category": ["Real Faces", "Fake Faces"],
                    "Count": [face_stats["Real"], face_stats["Fake"]]
                }
                results_area.dataframe(results_df)
        else:
            st.sidebar.error("Failed to load sample video. Please try uploading your own video instead.")

if __name__ == "__main__":
    main()
