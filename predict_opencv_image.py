import cv2
import numpy as np

from imagenet_labels import imagenet_labels

import time

tf_pb_model_path = "/home/nesvera/Documents/neural_nets/classification/mobilenets/model/model.pb"
#img_path = "/home/nesvera/Documents/neural_nets/classification/mobilenets/images/cheetah.jpg"
img_path = "/home/nesvera/Documents/neural_nets/classification/mobilenets/images/tesla.jpg"

# Load a model imported from Tensorflow
#tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
model = cv2.dnn.readNetFromTensorflow(tf_pb_model_path)

image_size = 224

# Input image
img = cv2.imread(img_path)
rows, cols, channels = img.shape

pred_start_time = 0
pred_period = 0
pred_fps = 0

while True:

    pred_start_time = time.time()

    blob_image = cv2.dnn.blobFromImage(img, 1./128, (image_size, image_size), (128, 128, 128), swapRB=True)
    model.setInput(blob_image)

    # Runs a forward pass to compute the net output
    net_output = model.forward()
    output_max = net_output.max()
    output_ind_class = net_output.argmax()

    pred_period = time.time() - pred_start_time
    pred_fps = 1./pred_period

    print(output_max, output_ind_class)
    print(imagenet_labels[output_ind_class])
    print("FPS: " + str(pred_fps))

