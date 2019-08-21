import sys
sys.path.append("..")

import prepare_image

import cv2

if __name__ == "__main__":

    # Load some image
    image_path = "../../images/traffic_sign_60.jpg"
    image = cv2.imread(image_path)

    while True:

        # Apply Change color properties (brightness, contrast, saturation, hue)
        new_image = prepare_image.change_color_properties(image)

        # Apply rotation, zoom in or zoom out in the image
        new_image = prepare_image.rotate_and_scale(new_image)

        # Apply blur effect in the image
        new_image = prepare_image.blur(new_image)

        # Apply gaussian or sand&pepper noise
        new_image = prepare_image.noise(new_image)

        # resize image to the input of the neural network
        new_image = cv2.resize(new_image, (224, 224))

        cv2.imshow('Image', image)
        cv2.imshow('New image', new_image)
        cv2.waitKey(0)    

    