#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pip', 'install pillow')
#get_ipython().run_line_magic('pip', 'install --upgrade pip')
#get_ipython().run_line_magic('pip', 'install tensorflow')
#get_ipython().run_line_magic('pip', 'install keras')
#get_ipython().run_line_magic('pip', 'install --upgrade tensorflow keras')
#get_ipython().run_line_magic('pip', 'install opencv-python')
#get_ipython().run_line_magic('pip', 'install graphviz')
#get_ipython().run_line_magic('pip', 'install pydot')


# In[2]:


import json
import shutil
from shutil import copy2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import collections


# In[3]:


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix


# In[5]:


import keras.layers as Layers
from keras.layers import BatchNormalization

import keras.activations as Actications
from keras.models import load_model
from keras.models import model_from_json
import keras.optimizers as Optimizer
import keras.metrics as Metrics
import keras.utils as Utils
from keras.utils import model_to_dot
# from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
from keras.preprocessing import image

import tensorflow as tf
from tensorflow import keras

# ### 예측하기

# In[39]:
def predict_age(image_path):

# 모델 파일 경로 설정
    try:
        #model_path = './myapp/static/models_and_weights/base_full_model0.h5'
        #model_path = './myapp/static/models_and_weights/deeper_CNN_20231211_205112.h5'
        #model_path = './myapp/static/models_and_weights/less_param_BW_CNN_20231214_035017.h5'
        model_path = './myapp/static/models_and_weights/less_param_BW_CNN_20231214_214355.h5'
    except OSError as e:
        print(f"Error loading the model: {e}")
# 모델 불러오기
    loaded_model = load_model(model_path)

# 모델 컴파일
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 이미지 파일 경로 설정
# [여기를 폴더내의 다른 사진 파일들로 시도해보세요]

# 이미지 불러오기 및 전처리
    img = image.load_img(image_path, target_size=(224, 224),color_mode="grayscale")
    #path = './myapp/static/val/0915_2001_21_00000029_D.png'
    #img=image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 이미지를 모델이 훈련된 스케일로 전처리
    
# 모델 예측
    predictions = loaded_model.predict(img_array)

# 예측 결과 출력
    class_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50+']
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0, class_idx]
    predicted_class = class_labels[class_idx]

    return predicted_class, "{:.4f}".format(confidence*100)

