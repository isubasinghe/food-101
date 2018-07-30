import argparse


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

def read_image(im_path, width, height):
    img = Image.open(im_path)
    img = img.resize((width, height))
    img = np.asarray(img)
    return img

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


def predict(img, num_out):
    if num_out < 1:
        return

    img = read_image(img, IM_WIDTH, IM_HEIGHT)
    model = get_model()
    results = model.predict(np.asarray([img]))
    results = results[0]
    results = list(results)
    #print(results.shape)
    print(np.argmax(results))
    print('Food={}, confidence={:0.3f}%'.format(CLASSES[np.argmax(results)], results[np.argmax(results)]))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--top', type=int)

    args = parser.parse_args()

    predict(args.image, args.top)
    #print(args.image)

if __name__ == '__main__':
    main()