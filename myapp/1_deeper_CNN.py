#!/usr/bin/env python
# coding: utf-8

# In[12]:


import json
import shutil
from shutil import copy2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import collections
import pickle


# In[11]:


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix


# In[1]:


import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import tensorflow.keras.optimizers as Optimizer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from tensorflow.keras.utils import model_to_dot
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


# In[4]:


def load_latest_model(directory):
    # 디렉토리 내의 모든 파일 리스트 가져오기
    all_files = os.listdir(directory)

    # .h5 확장자를 가진 파일들만 선택
    model_files = [f for f in all_files if f.endswith(".h5")]

    if not model_files:
        print("No model files found in the directory.")
        return None

    # 파일들의 수정 시간을 기준으로 정렬
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    # 가장 최근에 수정된 모델 파일 불러오기
    latest_model_path = os.path.join(directory, model_files[0])

    try:
        loaded_model = load_model(latest_model_path)
        print(f"Loaded the latest model: {latest_model_path}")
        return loaded_model
    except Exception as e:
        print(f"Failed to load the latest model. Error: {e}")
        return None

from datetime import datetime

def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"deeper_CNN_{timestamp}.h5"
    model_path = os.path.join('./models_and_weights/', model_filename)
    model.save(model_path)
    return model_filename

def load_model_with_filename(filename):
    if os.path.exists(filename):
        return load_model(filename)
    else:
        print(f"Model file {filename} does not exist.")
        return None


# In[ ]:


input_shape=(224,224,3)

# model= Sequential([
#     Conv2D(filters=36, kernel_size=7, activation='relu', input_shape=input_shape),
#     MaxPooling2D(pool_size=2),
#     Conv2D(filters=54, kernel_size=5, activation='relu', input_shape=input_shape),
#     MaxPooling2D(pool_size=2),
#     Flatten(),
#     Dense(2024, activation='relu'),
#      Dropout(0.5),
#     Dense(1024, activation='relu'),
#     Dropout(0.5),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     #20 is the number of outputs
#     Dense(8, activation='softmax')
# ])

# opt = keras.optimizers.Adam(learning_rate=0.00001)

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#filters= the depth of output image or kernels


# In[5]:


# model = load_model("./models_and_weights/base_full_model1.h5")
# 특정 디렉토리에서 최근에 저장된 모델 불러오기

# learning_rate = 0.001
# optimizer = Adam(learning_rate=learning_rate)

directory_path = "./models_and_weights"
model = load_latest_model(directory_path)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[9]:


# dimensions of the images
img_width, img_height = 224, 224

train_data_dir = './classification_data/Training'
test_data_dir = './classification_data/Validation'

# 50으로 해보기
epochs = 10
batch_size = 256


# In[13]:


# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        color_mode = 'rgb')

test_generator = test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = False,
        color_mode = 'rgb')


# In[1]:


# history = model.fit_generator(train_generator, steps_per_epoch= 13800//batch_size, epochs=epochs)
history = model.fit_generator(train_generator, epochs=epochs)


# In[ ]:


# 특정 위치에 history 객체를 파일에 저장

file_path = './history/history.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(history.history, file)


# In[ ]:


# # 저장된 history 객체를 불러오기
# with open('./history/history.pkl', 'rb') as file:
#     loaded_history = pickle.load(file)

# # 로드한 history 객체 사용
# print(loaded_history['loss'])
# print(loaded_history['val_loss'])

# # 파일에서 객체를 읽어오기
# with open('./history/history.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)

# print(loaded_data)


# In[ ]:


# 모델 저장
saved_model_filename = save_model(model)
print(f"Model saved as: {saved_model_filename}")


# In[25]:


get_ipython().run_cell_magic('time', '', 'test_generator.reset()\npred=model.predict_generator(test_generator, np.ceil(5050/batch_size))\n\n# `model.predict_generator` 함수는 제공된 데이터 제너레이터에서 배치 단위로 데이터를 가져와서 모델에 입력으로 전달하고 예측을 수행합니다. \n# 이때, `test_generator`로 전달된 제너레이터는 테스트 데이터셋에 대한 데이터를 생성하는 역할을 합니다. \n# `np.ceil(700/batch_size)`는 배치 크기에 기반하여 전체 데이터셋을 돌기 위해 필요한 에포크 수를 계산한 값입니다. \n# `test_generator`가 모든 데이터를 소진할 때까지 여러 배치에 걸쳐서 모델에 입력을 전달하게 됩니다.\n# `model.predict_generator`는 예측된 결과를 반환하는데, 이 결과는 각 이미지에 대한 모델의 출력입니다. \n# 만약 다중 클래스 분류 문제라면 각 클래스에 대한 확률이 담긴 배열이 반환됩니다. 이 배열의 크기는 `(전체 샘플 수, 클래스 수)`가 됩니다.\n# 따라서 `pred`는 모델이 테스트 데이터셋에 대해 예측한 결과를 담고 있는 배열입니다. 결과의 구조는 모델의 출력에 따라 다를 수 있습니다. \n# 예를 들어, 만약 모델이 softmax 활성화 함수를 사용했다면, 각 클래스에 대한 확률이 포함된 배열이 될 것입니다.\n')


# In[ ]:


def save_results(predictions, filenames, output_directory):
    # 결과를 저장할 디렉토리 생성
    # os.makedirs(output_directory, exist_ok=True)

    # 현재 시간을 이용한 고유한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results_{timestamp}.csv"

    # 결과 데이터프레임 생성
    results = pd.DataFrame({"Age": filenames, "Predictions": predictions})

    # 결과를 CSV 파일로 저장
    output_path = os.path.join(output_directory, output_filename)
    results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return results


# In[26]:


labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in np.argmax(pred, axis=1)]

filenames=test_generator.filenames
classes = [x.split('\\')[0] for x in filenames]

# results=pd.DataFrame({"Age":classes, "Predictions":predictions})
# results.to_csv('./results/basic_cnn_model_aug.csv',index=False)

# 예측 결과 저장
results = save_results(predictions, classes, "./results")


# In[27]:


results


# In[ ]:


# model.evaluate_generator 함수 사용
# train_generator: 모델을 평가할 데이터 제너레이터
# steps: 한 에포크 동안 제너레이터로부터 가져올 배치의 총 개수

# evaluation_results = model.evaluate_generator(train_generator, steps=40150//batch_size)

# evaluation_results는 모델 평가 결과를 담은 리스트
# 리스트의 구조는 [loss, accuracy] 또는 다중 메트릭을 사용한 경우에는 [metric1, metric2, ..., metricN]

# 예를 들어, 만약 모델이 categorical_crossentropy 손실 함수를 사용하고, accuracy 메트릭을 평가하는 경우:
# evaluation_results = [평가된 손실 값, 평가된 정확도]

# 만약 다중 메트릭을 사용하면 결과는 [metric1_value, metric2_value, ..., metricN_value] 형식이 됩니다.

# 이 결과를 출력하거나 다른 곳에서 활용할 수 있습니다.
# print("Evaluation Results:", evaluation_results)


# In[28]:


# evaluate the model
loss, acc = model.evaluate_generator(train_generator, steps=40150//batch_size)
print('Cross-entropy: ', loss)
print('Accuracy: ', acc)


# In[29]:


# evaluate the model
loss, acc = model.evaluate_generator(test_generator, steps=5050//batch_size)
print('Cross-entropy: ', loss)
print('Accuracy: ', acc)


# In[3]:


plt.plot(history.history['accuracy'])
plt.title('Base Model Accuracy over 100 Epochs')
plt.savefig('./images/base_model_aug_acc.png', bbox_inches='tight')

plt.plot(history.history['loss'])
plt.title('Base Model Loss over 100 Epochs')
plt.savefig('./images/base_model_aug_loss.png', bbox_inches='tight')

print("done")