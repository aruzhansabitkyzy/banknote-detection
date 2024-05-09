#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


# def infer_uploaded_video(conf, model):
#     """
#     Execute inference for uploaded video
#     :param conf: Confidence of YOLOv8 model
#     :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#     :return: None
#     """
#     source_video = st.sidebar.file_uploader(
#         label="Choose a video..."
#     )

#     if source_video:
#         st.video(source_video)

#     if source_video:
#         if st.button("Execution"):
#             with st.spinner("Running..."):
#                 try:
#                     tfile = tempfile.NamedTemporaryFile()
#                     tfile.write(source_video.read())
#                     vid_cap = cv2.VideoCapture(
#                         tfile.name)
#                     st_frame = st.empty()
#                     while (vid_cap.isOpened()):
#                         success, image = vid_cap.read()
#                         if success:
#                             _display_detected_frames(conf,
#                                                      model,
#                                                      st_frame,
#                                                      image
#                                                      )
#                         else:
#                             vid_cap.release()
#                             break
#                 except Exception as e:
#                     st.error(f"Error loading video: {e}")

import cv2
import streamlit as st
import tempfile

def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    # Create a temporary file to write the input video content
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())

                    # Open the input video file
                    vid_cap = cv2.VideoCapture(tfile.name)

                    # Get the original width and height of the input video frames
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Define the output video file name with the same extension as the input video
                    output_file_name = f"output.mp4"  # Output format will be mp4

                    # Define the codec and create a VideoWriter object to write the output video
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
                    out = cv2.VideoWriter(output_file_name, fourcc, 30.0, (width, height))

                    st_frame = st.empty()

                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            # Predict the objects in the image using YOLOv8 model
                            res = model.predict(image, conf=conf)

                            # Plot the detected objects on the video frame
                            res_plotted = res[0].plot()

                            # Resize the output frame to match the dimensions of the input video
                            res_plotted_resized = cv2.resize(res_plotted, (width, height))

                            # Write the processed frame to the output video file
                            out.write(res_plotted_resized)

                            # Display the processed frame
                            st_frame.image(res_plotted_resized, caption='Detected Video', channels="BGR", use_column_width=True)

                        else:
                            break

                    # Release the VideoCapture and VideoWriter objects
                    vid_cap.release()
                    out.release()

                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")