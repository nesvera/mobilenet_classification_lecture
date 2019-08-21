import sys
sys.path.append("..")

import prepare_image

import cv2

if __name__ == "__main__":

    # Load some image
    image_path = "../../images/traffic_sign_60.jpg"
    image = cv2.imread(image_path)

    while True:

        # Apply blur effect in the image
        new_image = prepare_image.blur(image)

        # resize image to the input of the neural network
        new_image = cv2.resize(new_image, (224, 224))

        cv2.imshow('Image', image)
        cv2.imshow('New image', new_image)
        cv2.waitKey(0)    

    