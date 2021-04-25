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
from operator import add

import base64
import json
import pickle
import tempfile
import matplotlib.pyplot as plt
import matplotlib

pd.options.plotting.backend = 'matplotlib'
DEBUG_MODE = False
MAX_FRAMES = 16
VERBOSE = False
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
    bboxes_detected = classifier.detectMultiScale(img)
    # Print bboxes over image
    # print bounding box for each detected face
    sub_images = []
    bboxes = []
    for box in bboxes_detected:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the img
        rectangle(img, (x, y), (x2, y2), (0,0,255), 1)
        sub_images.append(cv2.resize(img[y:y2, x:x2], (48,48)))
        bboxes.append([x,y,x2,y2])

    if verbose:
        st.title("Base Image and responses")
        st.image(img) # now has bboxes
    if verbose: 
        st.text("Now going to query the model, using the following config")
        st.text(f"URL: {url}")
        st.text(f"Token: {token}")
    
    # Loops over images in the frame
    sentiments = None
    bbs = None
    c = 0
    if len(sub_images) >=1:
        # Not really optimize, good for an hackathon
        subfname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        subfname = f"tmp/{subfname}"
        os.makedirs(subfname)
        
        for si, bb in zip(sub_images, bboxes):
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
                sentiments =  [sum(x) for x in zip(list(sentiments), list(classes.values()) )]
            elif max(classes.values()) > 0.5 and sentiments is None:
                c = c+1
                sentiments =  list(classes.values())

                # TODO: adjust for multiple bboxes
                bbs = bb

        
        if not DEBUG_MODE:
            shutil.rmtree(subfname, ignore_errors=True)
    else:
        st.text("No image to send to API")

    
    if sentiments:
        sentiments = [x / c for x in sentiments] 

    if verbose:
        st.text(f"Got sentiments: {sentiments} over {c} frames")

    return sentiments, c, bbs

@st.cache
def read_data(pklname):
    dd = pd.read_pickle(pklname)
    if dd.index.name != "idx":
        dd["idx"] = list(range(0, dd.shape[0]))
    dd.set_index("idx", inplace=True)
    #df = df.round(2) * 100
    min_val = int(dd.index.min())
    max_val = int(dd.index.max())

    return dd, min_val, max_val

def make_plot(source):
    #https://discuss.streamlit.io/t/cache-matplotlib-figure/12035/2
    f, ax = plt.subplots(1,1, figsize=(16,9))
    source.plot(ax=ax)
    for col in source.columns:
        if col == "neutral":
            plt.fill_between(df.index, source[col], alpha=0.1)
        else:
            plt.fill_between(df.index, source[col], alpha=0.4)
    return f

@st.cache
def process_video(file_buffer):
    pklname = None
    if file_buffer is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(file_buffer.read())
        vf = cv2.VideoCapture(tfile.name)
        #stframe = st.empty()
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
        bbs_list = []
        for framepath in frames_loc:
            img = get_opencv_img_from_buffer(open(framepath,"rb"),None)
            sent, c, bbs = process_image(img, VERBOSE)

            if sent is not None:
                img_list.append(framepath)
                sent_list.append(sent)
                bbs_list.append(bbs)


        # TODO: hardcoded!
        final_res = pd.DataFrame(sent_list, columns = ["other", "happiness", "sad", "neutral", "anger", "fear"])
        final_res["image"] = img_list
        final_res["bbox"] = bbs_list

        if DEBUG_MODE:
            st.subheader('Raw data')
            st.write(final_res)

        if DEBUG_MODE: 
            final_res.to_csv("sample.csv")
        
        pklname = random.choice(string.ascii_uppercase + string.digits) + ".pkl"
        final_res.to_pickle(pklname)
        st.text("Video analyzed")

    return pklname
    


#with st.echo(code_location='none'): # code_location='below'
st.set_page_config(layout="wide")
'''
# Welcome to vcoach

Do you have an important meeting incoming? A sales pitch? 
Or do you just need to express the right emotions? 
*vcoach* analyzes your facial expression and helps by increasing your self-awareness.

*vcoach* is a deep-learning powered APP built to analyze the way you present, built for demonstration purposes for the peltarion business hackhathon.

## How to use
* upload a video file of you speaking (any mp4 from youtube, from your webcam or grab a sample file [here](https://drive.google.com/file/d/1G-kTP67D5n-loiu73COlGtXcek45G_G8/view?usp=sharing) or [here](https://drive.google.com/file/d/1QDdcOlD70Udz2Fruoygvo0jd6e8e0G2U/view?usp=sharing))
* wait some seconds and play with the output


'''

if DEBUG_MODE:
    st.text(f"Hello, it s me, with opencv v. {cv2.__version__}")
file_buffer = st.file_uploader(  
                                "Upload an mp4 file of someone speaking be analyzed",
                                type=["mp4"],
                                accept_multiple_files=False
                                )
pklname = None
pklname = process_video(file_buffer)
# Viz
if pklname:
    df, min_val, max_val = read_data(pklname)
    r = st.slider(label="select frame", min_value=min_val,
            max_value=max_val,value=0)

    st.text(r)
    
    
    imgloaded = get_opencv_img_from_buffer(open(df.loc[r, "image"],"rb"), None)

    #st.text(df.loc[r]["bbox"])
    x,y,x2,y2 = list(df.loc[r]["bbox"])
    rectangle(imgloaded, (x, y), (x2, y2), (0,0,255), 1)
    
    col1, col2, col3 = st.beta_columns(3)
    col1.header("Frame")
    col1.image(imgloaded)

    col2.header("Face expression")
    #st.text(plt.style.available)
    plt.style.use("fivethirtyeight") # fivethirtyeight
    display_sentiment = ["other","happiness","sad","anger","fear","neutral"]
    source = df[display_sentiment]
    
    f = make_plot(source)
    plt.axvline(x=r, color="black")
    plt.legend(loc=2, prop={'size': 24})
    col2.pyplot(fig=f)

    col3.header("Frame analysis")
    col3.write(df.loc[r, display_sentiment].astype(float).round(2)*100)
