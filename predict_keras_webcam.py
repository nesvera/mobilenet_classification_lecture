import keras
from keras.preprocessing import image
from keras.applications.mobilenetv2 import preprocess_input
from keras.models import Model, load_model

import numpy as np
import time
import cv2

import argparse
import os
import random

from utils.object_classes import classes

if __name__ == "__main__":

    # argument handler
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_model', required=True,
                        help='Path to the weights/model file .pb')

    args = parser.parse_args()
    model_path = args.input_model

    # check if the path exist and it is a file
    if os.path.isfile(model_path) == False:
        print('Invalid path! It is not a file!')
        exit(1)

    model = load_model(model_path)
    model.summary()

    cap = cv2.VideoCapture(0)

    while True:

        pred_start_time = time.time()

        # read a new frame from the webcam
        ret, frame = cap.read()

        # reshape the image to the neural network
        image = cv2.resize(frame, (224, 224))
        image_tensor = np.reshape(image, (-1, image.shape[0], image.shape[1], image.shape[2]))

        # forward pass in the nn
        preds = model.predict(image_tensor)
        
        # get class with highest probability
        output_max = preds.max()
        output_ind_class = preds.argmax()
        
        pred_end_time = time.time()
        pred_period = pred_end_time - pred_start_time
        pred_fps = 1./pred_period

        cv2.imshow("image", image)
        cv2.waitKey(1)

        print("---------------------------------")
        print('Predicted: ', classes[output_ind_class])
        print('FPS: ' + str(pred_fps))
        

        
