import cv2
import argparse
import os
import time
import numpy as np

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

    # Load a model that was trained in tensorflow
    model = cv2.dnn.readNetFromTensorflow(model_path)

    image_size = 224

    # open webcam
    cap  = cv2.VideoCapture(0)

    pred_start_time = 0
    pred_period = 0
    pred_fps = 0

    while True:

        # read one frame from webcam
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        pred_start_time = time.time()

        # Convert 
        blob_image = cv2.dnn.blobFromImage(frame, 
                                           scalefactor=1.0, #1./128,          # standardization
                                           size=(image_size, image_size),     # input size of neural network
                                           mean=(0,0,0),#(128, 128, 128),     # normalization
                                           swapRB=False)                      # RGB -> BGR
        
        model.setInput(blob_image)

        # Run a forward pass to compute the net output
        net_output = model.forward()

        # Get prediction with best confidence
        output_max = net_output.max()
        output_ind_class = net_output.argmax()

        pred_object_class = classes[output_ind_class]

        pred_period = time.time() - pred_start_time
        pred_fps = 1./pred_period

        print("---------------------------------")
        print(pred_object_class)
        print("FPS: " + str(pred_fps))

        cv2.imshow('image', frame)
        cv2.waitKey(1)

