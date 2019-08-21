import argparse
import os
import sys
import cv2
import random
import numpy as np

def change_color_properties(image):
    """
    Change color properties (brightness, contrast, saturation, hue)
    
    """

    contrast_min = 0.5
    contrast_max = 1.5

    brightness_min = -(255/2.)
    brightness_max = (255/2.)

    hue_min = 0
    hue_max = 10

    saturation_min = -100.
    saturation_max = 100

    contrast_value = 0
    brightness_value = 0
    hue_value = 0
    saturation_value = 0

    # Randomly select each propertie to change and a value
    if random.random() < 0.5:
        contrast_value = random.uniform(contrast_min, contrast_max)
        #print("contrast: " + str(contrast_value))

    if random.random() < 0.5:
        brightness_value = random.uniform(brightness_min, brightness_max)
        #print("brightness: " + str(brightness_value))

    if random.random() < 0.5:
        hue_value = random.uniform(hue_min, hue_max)
        #print("hue: " + str(hue_value))

    if random.random() < 0.5:
        saturation_value = random.uniform(saturation_min, saturation_max)
        #print("saturation: " + str(saturation_value))

    # changes in bgr colorspace
    new_image = image.astype(float)
    new_image = contrast_value*new_image + brightness_value
    new_image = np.clip(new_image, a_min=0, a_max=255)
    new_image = new_image.astype('uint8')

    # changes in hsv colorspace
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
    new_image[:,:,0] = new_image[:,:,0] + hue_value
    new_image[:,:,1] = new_image[:,:,1] + saturation_value
    new_image = np.clip(new_image, a_min=0, a_max=255)
    new_image = new_image.astype('uint8')        
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    return new_image

def rotate_and_scale(image):
    """
    Apply random rotation to the image
    """

    angle_min = -90
    angle_max = 90
    
    center_offset_scale_max =  -0.1
    center_offset_scale_min =  0.1

    scale_min = 0.7
    scale_max = 1.3

    angle_value = 0
    center_offset_scale_value = 0
    scale_value = 1.

    new_image = image.copy()

    # Select randomly a angle and a offset to the image transformation
    if random.random() < 0.5:
        angle_value = random.uniform(angle_min, angle_max)
    
    if random.random() < 0.5:
        center_offset_scale_value = random.uniform(center_offset_scale_min, center_offset_scale_max)

    if random.random() < 0.5:
        scale_value = random.uniform(scale_min, scale_max)

    # Select a background to the final image transformated
    border_b_value = random.uniform(0, 255)
    border_g_value = random.uniform(0, 255)
    border_r_value = random.uniform(0, 255)
    
    cx = image.shape[0]/2. + image.shape[0]*center_offset_scale_value
    cy = image.shape[1]/2. + image.shape[1]*center_offset_scale_value

    im_width = image.shape[1]
    im_height = image.shape[0]
    
    M = cv2.getRotationMatrix2D((cx, cy), angle_value, scale_value)
    new_image = cv2.warpAffine(new_image, M, (im_width, im_height),
                            borderValue=(border_b_value, border_g_value, border_r_value))


    return new_image

def blur(image):

    blur_range = [1, 3, 5, 7, 9, 11]

    new_image = image.copy()

    if random.random() < 0.5:

        blur_value = random.choice(blur_range)
        new_image = cv2.GaussianBlur(new_image, (blur_value, blur_value), 0)

    return new_image

def noise(image):

    noise_types = ['salt_and_pepper', 'gaussian_noise']

    sap_percentage_min = 0
    sap_percentage_max = 0.2

    gauss_sigma_min = 0
    gauss_sigma_max = 30
    
    new_image = image.copy()

    if random.random() < 0.5:

        #noise = random.choice(noise_types)
        noise = random.choice(noise_types)
        
        if noise == 'salt_and_pepper':

            noise_amount = random.uniform(sap_percentage_min, sap_percentage_max)
            pixel_noise_amount = int(new_image.size*noise_amount)

            # generate a list of random pixel position in the image
            pixel_location = [ (random.randint(0, new_image.shape[0]-1),
                                random.randint(0, new_image.shape[1]-1),
                                random.randint(0, new_image.shape[2]-1)) for k in range(pixel_noise_amount)]

            for p in pixel_location:
                if random.random() < 0.5:
                    new_image[p] = 0
                else:
                    new_image[p] = 255

        elif noise == 'gaussian_noise':
            
            mean = 0
            gauss_sigma_value = random.uniform(gauss_sigma_min, gauss_sigma_max)
            gauss = np.random.normal(mean, gauss_sigma_value, new_image.shape)
            new_image = new_image.astype(float)
            new_image = new_image + gauss
            new_image = np.clip(new_image, 0, 255)
            new_image = new_image.astype('uint8')

    return new_image


def prepare_dataset(folder_path, image_size=(224,224), augmentation=False, multply=10):
    """

    :param folder_path: 
    :param image_size:
    """

    num_class = len(os.listdir(folder_path))

    # loop through the classes
    for ind_class, image_class in enumerate(os.listdir(folder_path)):

        image_class_path = folder_path + "/" + image_class

        num_images = len(os.listdir(image_class_path))

        # loop through the images
        for ind_image, image_filename in enumerate(os.listdir(image_class_path)):

            print("Preparing -> Class: " + str(ind_class) + " of " + str(num_class) + 
                  " -> Image: " + str(ind_image) + " of " + str(num_images))

            # load image
            image_filepath = image_class_path + "/" + image_filename
            image = cv2.imread(image_filepath)

            # Just resize the image
            if augmentation == False:
                new_image = cv2.resize(image, image_size)
                
                # save image
                cv2.imwrite(image_filepath, new_image)
                
            # Data augmentation
            else:

                # resize and save original image
                new_image = cv2.resize(image, image_size)
                cv2.imwrite(image_filepath, new_image)


                # Generate n modified copies of the image
                for n in range(multply):
                    new_image = image.copy()
                    
                    # change color properties (brightness, contrast, saturation, hue)
                    new_image = change_color_properties(new_image)

                    # rotate
                    new_image = rotate_and_scale(new_image)

                    new_image = blur(new_image)

                    new_image = noise(new_image)

                    new_image = cv2.resize(new_image, image_size)

                    filename, extension = image_filename.split(".")
                    image_filepath = image_class_path + "/" + filename + "_" + str(n) + "." + extension

                    cv2.imwrite(image_filepath, new_image)             

if __name__ == "__main__":

    # handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset', required=False,
                        help='Path to the directory of the dataset')

    args = parser.parse_args()

    # Check if the path is a real directory
    dataset_path = args.dataset
    if os.path.isdir(dataset_path) == False:
        print("Dataset path is not a directory")
        sys.exit(1)

    # try to find train and valid folder
    dataset_train_path = None
    dataset_validation_path = None

    for filename in os.listdir(dataset_path):

        # convert all string to lower case
        filename = filename.lower()

        if filename == 'train':
            dataset_train_path = dataset_path + "/" + filename

        elif filename == 'validation':
            dataset_validation_path = dataset_path + "/" + filename

    # check if the train directory was found
    if (dataset_train_path != None):
        # We are going to use augmentation in the train set to increase the number or images
        # trying to generalize the images
        prepare_dataset(dataset_train_path, image_size=(224,224), augmentation=True)

    else:
        print("Train folder was not found inside the folder!")


    # check if the validation directory was found
    if (dataset_validation_path != None):
        # We are NOT going to use augmentation in the validation set, because this set
        # will be used to evaluate the neural network. The images inside de validation
        # set must be valid examples of objects
        prepare_dataset(dataset_train_path, image_size=(224,224), augmentation=False)

    else:
        print("Validation folder was not found inside the folder!")

    