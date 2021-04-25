from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import numpy as np
import sidekick
import cv2
from cv2 import imread, imwrite
from cv2 import rectangle
from cv2 import CascadeClassifier
import yaml
import requests
import random
import os
import string
import shutil

# send
import base64
import json
import pickle

DEBUG_MODE = True

def get_opencv_img_from_buffer(buffer, flags):
    # https://stackoverflow.com/questions/13329445/how-to-read-image-from-in-memory-buffer-stringio-or-from-url-with-opencv-pytho
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

def im2encode(im):
    imdata = pickle.dumps(im)
    jstr = base64.b64encode(imdata).decode('utf-8')
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

        # Finally, query P model
        st.text("Now going to query the model, using the following config")
        

        url=configs["peltarion_endpoint_image"]["url"]
        token=configs["peltarion_endpoint_image"]["token"]
        st.text(f"URL: {url}")
        st.text(f"Token: {token}")
        
        if len(sub_images) >=1:
            # Not really optimize, good for an hackathon
            subfname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            subfname = f"tmp/{subfname}"
            os.makedirs(subfname)
            for si in sub_images:
                fname = f"{subfname}/tobesent.jpg"
                imwrite(fname, si)

                headers = {'Authorization': 'Bearer ' + token}
                files = {"image": open(fname, "rb")}
                x = requests.post(url, files = files, headers= headers)

                st.title("Image and response")
                st.image(si)
                st.text(x.text)
            
            if not DEBUG_MODE:
                shutil.rmtree(subfname, ignore_errors=True)
        else:
            st.text("No image to send to API")

            


