import cv2
import time
import datetime
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', required=True,
                        help='Path to the folder that will receive the images')
    
    args = parser.parse_args()

    folder_path = args.output_path

    # open webcam
    cap = cv2.VideoCapture(0)

    # how many frames it will save by second
    frames_saved_by_second = 15
    period_to_save = 1./frames_saved_by_second

    timer_start = time.time()

    images_saved_counter = 0

    while True:

        ret, frame = cap.read()

        # count elapsed time
        if (time.time()-timer_start) > period_to_save:

            timer_start = time.time()

            images_saved_counter += 1
            print(str(images_saved_counter) + " images saved.")

            # use year_month_time as the name of the image
            filename = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')
            image_path = folder_path + "/" + filename + ".jpg"

            cv2.imwrite(image_path, frame)

            cv2.imshow("Image", frame)
            cv2.waitKey(1)



