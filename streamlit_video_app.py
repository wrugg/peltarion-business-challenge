from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd

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

import base64
import json
import pickle
import tempfile

DEBUG_MODE = True

with open('config.yaml') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

# frontal face classifier
classifier = CascadeClassifier('models/haar_frontalface_default.xml')
url = configs["peltarion_endpoint_image"]["url"]
token = configs["peltarion_endpoint_image"]["token"]

def get_opencv_img_from_buffer(buffer, flags):
    # https://stackoverflow.com/questions/13329445/how-to-read-image-from-in-memory-buffer-stringio-or-from-url-with-opencv-pytho
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

def im2encode(im):
    imdata = pickle.dumps(im)
    jstr = base64.b64encode(imdata).decode('utf-8')
    return jstr

def process_image(img, verbose=False):
    bboxes = classifier.detectMultiScale(img)
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

    if verbose:
        st.title("Base Image and responses")
        st.image(img) # now has bboxes
    if verbose: 
        st.text("Now going to query the model, using the following config")
        st.text(f"URL: {url}")
        st.text(f"Token: {token}")
    
    # Loops over images in the frame
    if len(sub_images) >=1:
        # Not really optimize, good for an hackathon
        subfname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        subfname = f"tmp/{subfname}"
        os.makedirs(subfname)
        
        sentiments = None
        c = 0
        for si in sub_images:
            fname = f"{subfname}/tobesent.jpg"
            imwrite(fname, si)

            headers = {'Authorization': 'Bearer ' + token}
            files = {"image": open(fname, "rb")}
            x = requests.post(url, files = files, headers= headers)

            classes = json.loads(x.text).get("label", None)
            if verbose:
                
                st.text(classes)
                st.image(si)

                if classes:
                    maxkey = max(classes, key=classes.get)
                    st.text(f"Max is {maxkey} with value {classes[maxkey]}")
                
                st.text("-------------")

            if max(classes.values()) > 0.5 and sentiments is not None:
                c = c+1
                sentiments = sentiments + classes.values()
            elif max(classes.values()) > 0.5 and sentiments is None:
                c = c+1
                sentiments =  classes.values()
        
        if not DEBUG_MODE:
            shutil.rmtree(subfname, ignore_errors=True)
    else:
        st.text("No image to send to API")

    
    if sentiments:
        sentiments = [x / c for x in sentiments] 

    if verbose:
        st.text(f"Got sentiments: {sentiments} over {c} frames")

    return sentiments, c

with st.echo(code_location='below'):
    if DEBUG_MODE:
        st.text(f"Hello, it s me, with opencv v. {cv2.__version__}")
    file_buffer = st.file_uploader(  
                                      "Upload an image with faces",
                                      type=["mp4"],
                                      accept_multiple_files=False
                                      )
    MAX_FRAMES = 10
    VERBOSE = False
    if file_buffer is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(file_buffer.read())
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        i = 0
        frames_loc = []
        os.makedirs("tmp/video", exist_ok=True)
        while vf.isOpened():
            ret, frame = vf.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            if (i % 24*120) == 0:
                destpath = f"tmp/video/{i}.jpg"
                imwrite(destpath, frame)
                frames_loc.append(destpath) 
                
            elif len(frames_loc) > MAX_FRAMES:
                break
            i = i+1
        st.text(f"Processed video till {i}-th frame, written {len(frames_loc)} frames")

        img_list = []
        sent_list = []
        for framepath in frames_loc:
            img = get_opencv_img_from_buffer(open(framepath,"rb"),None)
            sent, c = process_image(img, VERBOSE)

            if sent is not None:
                img_list.append(framepath)
                sent_list.append(sent)


        # TODO: hardcoded!
        final_res = pd.DataFrame(sent_list, columns = ["other", "happiness", "sad", "neutral", "anger", "fear"])
        final_res["image"] = img_list

        st.subheader('Raw data')
        st.write(final_res)

        if DEBUG_MODE: 
            final_res.to_csv("sample.csv")
            


