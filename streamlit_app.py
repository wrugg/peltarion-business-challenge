from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import numpy as np
import sidekick
import cv2
from cv2 import imread
from cv2 import rectangle
from cv2 import CascadeClassifier
import yaml
import requests

# send
import base64
import json
import pickle

def get_opencv_img_from_buffer(buffer, flags):
    # https://stackoverflow.com/questions/13329445/how-to-read-image-from-in-memory-buffer-stringio-or-from-url-with-opencv-pytho
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

def im2json(im):
    """Convert a Numpy array to JSON string"""
    # https://stackoverflow.com/questions/55892362/sending-opencv-image-in-json
    imdata = pickle.dumps(im)
    jstr = json.dumps({"image": base64.b64encode(imdata).decode('ascii')})
    return jstr

def im2encode(im):
    """Convert a Numpy array to JSON string"""
    # https://stackoverflow.com/questions/55892362/sending-opencv-image-in-json
    imdata = pickle.dumps(im)
    jstr = base64.b64encode(imdata).decode('ascii')
    return jstr

with open('config.yaml') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

with st.echo(code_location='below'):
    st.text(f"Hello, it s me, with opencv v. {cv2.__version__}")
    img_file_buffer = st.file_uploader(  
                                      "Upload an image with faces",
                                      type=["jpg"],
                                      accept_multiple_files=False
                                      )

    if img_file_buffer is not None:

        img = get_opencv_img_from_buffer(img_file_buffer,None)#'tmp/test1.jpg')

        st.text(f"Object type: {type(img)}")
        #st.image(img)

        # Load bbox classifier
        classifier = CascadeClassifier('models/haar_frontalface_default.xml')
        bboxes = classifier.detectMultiScale(img)
        # st.text(bboxes)

        # Print bboxes over image
        # print bounding box for each detected face
        sub_images = []
        for box in bboxes:
            # extract
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the img
            rectangle(img, (x, y), (x2, y2), (0,0,255), 1)

            sub_images.append( cv2.resize(img[y:y2, x:x2], (48,48) ))
        # show the image
        #imshow('face detection', pixels)
        st.image(img)

        # print sub images
        for si in sub_images:
            st.image(si)

        # Finally, query P model
        st.text("Now going to query the model, using the following config")
        

        url=configs["peltarion_endpoint_image"]["url"]
        token=configs["peltarion_endpoint_image"]["token"]
        
        header = {'Authorization': 'Bearer {}'.format(token), 'Content-Type': 'application/json'}
        payload = {"rows":
                    [{"image": im2encode(sub_images[0])}]
                    #[{"image": sub_images[0]}]
                  }

        st.text(f"Sending request to {url}")

        response = requests.request("POST", 
                                    url,
                                    headers=header,
                                    json=payload)
        
        st.title("Request URL")
        st.text(response.request.url)
        st.title("Request body")
        st.text(response.request.body)
        st.title("Request headers")
        st.text(response.request.headers)

        st.text(response)
