import argparse
import glob
import io

from itertools import cycle
import pickle as pk

from flask import Flask, request, abort, jsonify
from threading import Lock
from gevent.wsgi import WSGIServer

import pandas as pd
import numpy as np

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import model_from_json
from keras import backend as k

import tensorflow as tf


from PIL import Image



CLASSES = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheese_plate',
 'cheesecake',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']


IM_WIDTH = 256
IM_HEIGHT = 256
NO_CLASSES = len(CLASSES)

def get_model():
    base_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (IM_WIDTH, IM_HEIGHT, 3))
    for layer in base_model.layers[:5]:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(NO_CLASSES, activation="softmax")(x)
    model = Model(input = base_model.input, output = predictions)
    model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    model.load_weights('./model.h5')
    return model

app = Flask(__name__)
model = None
graph = None
lock = Lock()

def read_image(im_path, width, height):
    img = Image.open(im_path)
    img = img.resize((width, height))
    img = np.asarray(img)
    return img




def predict(img):

    #img = read_image(img_path, IM_WIDTH, IM_HEIGHT)
    with lock:
        with graph.as_default():
            results = model.predict(np.asarray([img]))

    results = results[0]
    results = list(results)
    #print(results.shape)
    #print(np.argmax(results))
    #print('Food={}, confidence={:0.6f} file={}'.format(CLASSES[np.argmax(results)], results[np.argmax(results)], img_path))
    
    return CLASSES[np.argmax(results)], float(results[np.argmax(results)])

@app.route('/')
def index():
    return "Hello"

@app.route("/api", methods=["POST"])
def api():
    if request.files.get("image"):
        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        if image.mode != "RGB":
            image.mode = "RGB"

        image = image.resize((IM_WIDTH, IM_HEIGHT))
        image = np.asarray(image)
        data = {}
        data["results"], data["confidence"] = predict(image)
        return jsonify(data)
    else:
        abort(404)



def main():
    global model
    model = get_model()
    global graph
    graph = tf.get_default_graph()

    http_server = WSGIServer(('', 8080), app)
    http_server.serve_forever()
    #app.run('localhost', 8080)

if __name__ == '__main__':
    main()