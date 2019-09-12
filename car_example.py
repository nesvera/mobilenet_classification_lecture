import argparse
import os
import time

import cv2
import numpy as np
from collections import deque

from utils.object_classes import classes

if __name__ == "__main__":

    # argument handler
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_model', required=True,
                        help='Path to the weights/model file .pb')

    args = parser.parse_args()
    model_path = args.input_model

    # check if the path exist and if it isa file
    if os.path.isfile(model_path) == False:
        print("Invalid path! It is not a file!")
        exit(1)

    # Load a model that was trained in tensorflow/keras
    model = cv2.dnn.readNetFromTensorflow(model_path)

    image_size = 224

    # create a FIFO buffer to average the predictions
    prediction_buffer = deque([], maxlen=10)

    # open camera
    cap = cv2.VideoCapture(0)

    pred_start_time = time.time()
    pred_period = 0
    pred_fps = 0

    while True:

        # read frame from the webcam
        ret, frame = cap.read()

        # here the image is resized just to be shown in the screen
        # if not, the raspberry will suffer to present it
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # Convert image to the necessary format, resize to the correct size (224,224),
        # and normalize each pixel of the image from (0->255) to (0->1)
        blob_image = cv2.dnn.blobFromImage(frame, 
                                           scalefactor= 1./255,               # normalization
                                           size=(image_size, image_size),     # input size of neural network
                                           mean=(0,0,0),                      # standardization
                                           swapRB=False) 

        model.setInput(blob_image)

        # Run a forward pass to compute the net output
        net_output = model.forward()

        # Get the prediction with best confidence
        output_max = net_output.max()
        output_ind_class = net_output.argmax()

        # add prediction to the buffer to find the mode
        prediction_buffer.append(output_ind_class) 

        pred_period = time.time() - pred_start_time
        pred_fps = 1./pred_period

        # check if the buffer is full
        if len(prediction_buffer) == prediction_buffer.maxlen:

            # find the mode of the predictions    
            prediction_mode = max(prediction_buffer, key=prediction_buffer.count)
        
            current_prediction = classes[output_ind_class]
            mode_predection = classes[prediction_mode]

            print("---------------------------------")
            print("Current prediction: " + current_prediction)
            print("Prediction mode: " + mode_predection)
            print("FPS: " + str(pred_fps))

            # do some actions based in the objects detected
            if mode_predection == 'background':
                pass
            
            elif mode_predection == 'traf_sign_60':
                print("Action: braap")

            elif mode_predection == 'traf_sign_free':
                print("Action: braaaaaaaaaaaaaaaaaaap")

            elif mode_predection == 'traf_sign_stop':
                print("Action: break")


        cv2.imshow("image", frame)
        cv2.waitKey(1)