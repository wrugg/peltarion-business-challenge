from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import numpy as np
import cv2
from cv2 import imread
from cv2 import rectangle
from cv2 import CascadeClassifier

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


def get_opencv_img_from_buffer(buffer, flags):
    # https://stackoverflow.com/questions/13329445/how-to-read-image-from-in-memory-buffer-stringio-or-from-url-with-opencv-pytho
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

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
        st.image(img)

        # Load bbox classifier
        classifier = CascadeClassifier('models/haar_frontalface_default.xml')
        bboxes = classifier.detectMultiScale(img)
        st.text(bboxes)

        # Print bboxes over image
        # print bounding box for each detected face
        sub_images = []
        for box in bboxes:
            # extract
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the img
            rectangle(img, (x, y), (x2, y2), (0,0,255), 1)

            sub_images.append( img[y:y2, x:x2] )
        # show the image
        #imshow('face detection', pixels)
        st.image(img)

        # print sub images
        for si in sub_images:
            st.image(si)
