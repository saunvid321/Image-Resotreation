import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import save_img
import os
from skimage.io import imsave
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import save_img
from mpl_toolkits.axes_grid1 import ImageGrid
import glob
# from tqdm import tqdm
import warnings;
warnings.filterwarnings('ignore')
import glob
from keras.preprocessing.image import img_to_array,load_img
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

import streamlit as st
import pandas as pd
import numpy as np


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import streamlit as st
import cv2
from PIL import Image,ImageEnhance

import pandas as pd
import glob
import random
import os
import time
import pandas as pd
import pydub
import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import streamlit as st
from image_processing import *
def main():
    """
    Image Restoration
    """
    st.title('Image Restoration 🤖')
    #st.text('Auto-Encoder and OpenCV')

    menu = ['Denoise','Clarification','Inpainting']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Denoise':
        st.subheader('Image Denoise')
        image_file = st.file_uploader("Upload Image",type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            input_image = Image.open(image_file)
            st.text('Original Image')
            input_image=noisy_image(input_image)

            st.image(input_image,width=250)

        if st.button("Process"):
            result_img = get_denoisy(input_image)
            st.image(result_img,clamp=True, channels='RGB',width=250)

    if choice == 'Clarification':
        st.subheader('Image Clarification')
        image_file = st.file_uploader("Upload Image",type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            input_image = Image.open(image_file)
            st.text('Original Image')
            input_image=pixalate_image(input_image)

            st.image(input_image,width=250)

        if st.button("Process"):
            result_img = get_depixalate(input_image)
            st.image(result_img,clamp=True, channels='RGB',width=250)
    if choice == 'Inpainting':
        st.subheader('Image Inpainting')
        image_file = st.file_uploader("Upload Image",type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            input_image = Image.open(image_file)
            st.text('Original Image')
            input_image=masked_image(input_image)

            st.image(input_image)

        if st.button("Process"):
            result_img = get_inpainted(input_image)
            st.image(result_img,clamp=True, channels='RGB',width=250)



if __name__ == '__main__':
    main()