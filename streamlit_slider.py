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
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_MODE = True

def get_opencv_img_from_buffer(buffer, flags):
    # https://stackoverflow.com/questions/13329445/how-to-read-image-from-in-memory-buffer-stringio-or-from-url-with-opencv-pytho
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

@st.cache
def read_data():
    df = pd.read_pickle("sample.pkl")
    df["Unnamed: 0"] = list(range(0, df.shape[0]))
    df.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
    df.set_index("idx", inplace=True)

    min_val = int(df.index.min())
    max_val = int(df.index.max())

    return df, min_val, max_val

with st.echo(code_location='below'):
    st.text("ciao")
    df, min_val, max_val = read_data()
    st.write(df)

    r = st.slider(label="select frame", min_value=min_val,
              max_value=max_val,value=0)

    st.text(r)
    st.write(df.loc[r])
    
    imgloaded = get_opencv_img_from_buffer(open(df.loc[r, "image"],"rb"), None)

    st.text(df.loc[r]["bbox"])
    x,y,x2,y2 = list(df.loc[r]["bbox"])
    rectangle(imgloaded, (x, y), (x2, y2), (0,0,255), 1)
    st.image(imgloaded)
    # bar_chart, area_chart, line_chart
    #st.area_chart(df[["other","happiness","sad","anger","fear"]])

    pd.options.plotting.backend = "seaborn"#"plotly" 
    f, ax = plt.subplots(1,1, figsize=(16,9))
    df[["other","happiness","sad","anger","fear"]].plot(ax=ax)
    st.pyplot(fig=f)

            


