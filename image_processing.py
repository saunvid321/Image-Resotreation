import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os

from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image as image_prep
from keras.preprocessing.image import save_img
from mpl_toolkits.axes_grid1 import ImageGrid
import glob
import warnings;

import glob
from keras.preprocessing.image import img_to_array,load_img
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

denoising=tf.keras.models.load_model('./denoising-11-5.h5')
depixalate=tf.keras.models.load_model('./pixalate_scale_23.h5')
Inpainting=tf.keras.models.load_model('./Inpainting-50.h5')


def noisy_image(img):
  img = image_prep.img_to_array(img)
  img = img/255.
  noise =  np.random.normal(loc=0, scale=1, size=img.shape)
  noisy = np.clip((img + noise*0.2),0,1)
  noisy = cv2.resize(noisy, (256, 256)) 
  return noisy

def pixalate_image(image, scale_percent = 23):

  image = image_prep.img_to_array(image)
  image = image/255.
  image = cv2.resize(image, (256, 256))
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)

  small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
  # scale back to original size
  width = int(small_image.shape[1] * 100 / scale_percent)
  height = int(small_image.shape[0] * 100 / scale_percent)
  dim = (width, height)

  low_res_image = cv2.resize(small_image, dim, interpolation = cv2.INTER_AREA)

  low_res_image = cv2.resize(low_res_image, (256, 256)) 

  return low_res_image

def masked_image(image):
  #mask = np.full((256,256,3), 255, np.uint8)
  image = image_prep.img_to_array(image)
  image = image/255.
  image=cv2.resize(image, (256, 256))
  for _ in range(np.random.randint(1, 10)):
    x1, x2 = np.random.randint(1, 256), np.random.randint(1, 256)
    y1, y2 = np.random.randint(1, 256), np.random.randint(1, 256)
    thickness = np.random.randint(1, 7)
    masked=cv2.line(image,(x1,y1),(x2,y2),(1,1,1),thickness)
  masked=cv2.resize(masked, (256, 256)) 
  return masked

def get_denoisy(noisy_images):
  noisy_images = np.array([noisy_images])
  noisy_pred=denoising.predict(noisy_images)
  return noisy_pred[0]

def get_depixalate(pixalate_images):
  pixalate_images = np.array([pixalate_images])
  depixalate_pred=depixalate.predict(pixalate_images)
  return depixalate_pred[0]

def get_inpainted(scratched_images):
  scratched_images = np.array([scratched_images])
  inpainted_pred=Inpainting.predict(scratched_images)
  return inpainted_pred[0]