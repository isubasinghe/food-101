
# coding: utf-8

# In[1]:


from itertools import cycle
import pickle as pk

import pandas as pd
import numpy as np


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import model_from_json
from keras import backend as k



from PIL import Image

# In[2]:


test_files = pd.read_json('./meta/test.json')
train_files = pd.read_json('./meta/train.json')


# In[3]:


img_width, img_height = 256, 256
batch_size = 100
epochs = 50000
no_labels = len(train_files.columns)


# In[4]:


class DataGenerator:
    
    def read_image(image_path, width, height):
        img = Image.open(image_path)
        img = img.resize((width, height))
        img = np.asarray(img)
        return img
    
    def __init__(self, files_df,path='.', batch_size=10):
        self.path = path
        self.files_df = files_df
        self.cycles = []
        for j in range(files_df.shape[0]//batch_size):
             self.cycles.append((j*batch_size, (j*batch_size)+(batch_size-1)))
        self.cycles = cycle(self.cycles)
        
    def next_batch(self):
        indexes = next(self.cycles)
        X = []
        Y = []
        for i in range(indexes[0], indexes[1]):
            row = self.files_df.loc[i]
            for j in range(len(row)):
                x = DataGenerator.read_image(self.path + "/" + row[j] + ".jpg", img_width, img_height)
                if x.shape != (256, 256, 3):
                    continue
                X.append(x)
                y = np.zeros(101)
                y[j] = 1
                Y.append(y)
        return (np.asarray(X), np.asarray(Y))


# In[5]:


dg = DataGenerator(train_files, path="./images", batch_size=10)


# In[6]:


X_train, Y_train = dg.next_batch()


# In[7]:


base_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
for layer in base_model.layers[:5]:
    layer.trainable = False
base_model.summary()


# In[8]:


x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(no_labels, activation="softmax")(x)
model = Model(input = base_model.input, output = predictions)
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# In[ ]:

model.load_weights("./model.h5")
print("Loaded model")

for i in range(50000000):
    if i%10 == 0:
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            print("Saved JSON")
        model.save_weights("model.h5")
        print("Saved H5")
    model.fit(x=X_train, y=Y_train)
    X_train, Y_train = dg.next_batch()


