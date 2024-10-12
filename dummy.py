import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image
import numpy as np
import os
import time


model = YOLO("excavator.pt")


st.set_page_config(page_title="Advitiix Technovate", page_icon="✔", layout="wide")
st.image("alogo.jpg", width=300)
st.markdown("<h1 style='text-align: center; color: blue;'>Advitiix Technovate </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Detect Excavator Bucket Occupancy: Filled or Vacant</h3>", unsafe_allow_html=True)
st.markdown('<style>h1{animation: fadeIn 2s ease-in-out;}</style>', unsafe_allow_html=True)


if 'password_attempted' not in st.session_state:
    st.session_state['password_attempted'] = False
    st.session_state['password_correct'] = False

if not st.session_state['password_attempted']:
    st.info("Welcome! Please enter the password.")


password = st.text_input("Enter password to continue", type="password")


if password and not st.session_state['password_attempted']:
    if password == "intern@advitiix":
        st.session_state['password_correct'] = True
        st.session_state['password_attempted'] = True
    else:
        st.session_state['password_correct'] = False
        st.session_state['password_attempted'] = True


if st.session_state['password_correct']:
    st.success("Password is correct! Upload an image or video for detection.")
    
  
    confidence_threshold = st.slider('Confidence Threshold:', min_value=0, max_value=100, value=50)
    st.write(f"Selected confidence: {confidence_threshold}%")
    confidence = confidence_threshold / 100.0
    
    uploaded_file = st.file_uploader("Upload Files", type=["png", "jpg", "jpeg", "mp4"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension in [".png", ".jpg", ".jpeg"]:
           
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_np = np.array(image)

            with st.spinner('Processing image...'):
                results = model.predict(source=image_np, conf=confidence, show=False)
                detected_image = results[0].plot()
                st.image(detected_image, caption="Detected Objects", use_column_width=True)

      
            img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            cv2.imwrite(img_bytes.name, detected_image)

            with open(img_bytes.name, 'rb') as f:
                st.download_button("Download Image", f, file_name="detected_image.png")

        elif file_extension == ".mp4":
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_file.name, fourcc, fps, (width, height))
            stframe = st.empty()

            with st.spinner('Processing video...'):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict(source=frame, conf=confidence, show=False)
                    detected_frame = results[0].plot()
                    out.write(detected_frame)
                    stframe.image(detected_frame, channels="BGR")
                    time.sleep(0.05)  

            cap.release()
            out.release()

            st.markdown("### Download the detected video")
            with open(temp_video_file.name, 'rb') as f:
                st.download_button("Download Video", f, file_name="detected_video.mp4")

    st.markdown("<script>alert('Detection complete!');</script>", unsafe_allow_html=True)


elif st.session_state['password_attempted'] and not st.session_state['password_correct']:
    st.error("Incorrect password! Please try again.")
