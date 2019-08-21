import argparse
import sys
import os
import random
import math
import shutil

import cv2

def split_data(input_path, output_path, validation_size=0.2):
    """
    Split images from a directory in train and validation datasets

    :param input_path: path to the dataset with all images
    
    Example:
    input_dataset
        cats
            cat1.png
            cat2.png
        dogs
            dogo1.png
            dogo2.png

    :param output_path: path to the directory where the train and validation folder will
    be placed
    
    Example:
    output_dataset
        train
            cats
                cat2.png
            dogs
                dogo1.png
        validation
            cats
                cat1.png
            dogs
                dogo2.png

    """

    # build train output folder
    output_train_folder_path = output_path + "/train"
    try:
        os.mkdir(output_train_folder_path)
    except FileExistsError:
        pass

    # build validation output folder
    output_validation_folder_path = output_path + "/validation"
    try:
        os.mkdir(output_validation_folder_path)
    except FileExistsError:
        pass

    # look all the files(folders) inside the input directory
    for class_folder in os.listdir(input_path):
    
        # check if the file is a directory
        input_class_folder_path = input_path + "/" + class_folder
        if os.path.isdir(input_class_folder_path) == True:

            output_train_class_folder_path = output_train_folder_path + "/" + class_folder
            output_validation_class_folder_path = output_validation_folder_path + "/" + class_folder

            # create class folders inside train and validation
            try:
                os.mkdir(output_train_class_folder_path)
                os.mkdir(output_validation_class_folder_path)
            except FileExistsError:
                pass

            # get name of all files(images) inside the class folder
            images = os.listdir(input_class_folder_path)

            # shuffle the names
            random.shuffle(images)

            # Split between train and validation
            n_validation = math.ceil(len(images)*validation_size)

            images_validation = images[0:n_validation]
            images_train = images[n_validation:]

            # Save train images
            for image_filename in images_train:

                # build path string from the input image and to output image
                image_filepath_in = input_class_folder_path + "/" + image_filename
                image_filepath_out = output_train_class_folder_path + "/" + image_filename
                
                shutil.copy(image_filepath_in, image_filepath_out)


            # Save validation images
            for image_filename in images_validation:
                
                # build path string from the input image and to output image
                image_filepath_in = input_class_folder_path + "/" + image_filename
                image_filepath_out = output_validation_class_folder_path + "/" + image_filename

                shutil.copy(image_filepath_in, image_filepath_out)
            
if __name__ == "__main__":

    # handle arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', required=False, 
                        help='Path to the directory of the dataset')
    parser.add_argument('-o', '--output_dataset', required=False,
                        help="Path to the directory where the augmented dataset will be placed")
                
    args = parser.parse_args()
    
    # Check if the paths are real directories
    dataset_input_path = args.input_dataset
    if os.path.isdir(dataset_input_path) == False:
        print("Input path is not a directory!")
        sys.exit(1)

    dataset_output_path = args.output_dataset
    if os.path.isdir(dataset_output_path) == False:
        print("Output path is not a directory!")
        sys.exit(1)

    split_data(dataset_input_path, dataset_output_path)

    

    
    