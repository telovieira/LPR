from detector.plate_detection import LicensePlateDetector
import argparse
import time

start_time = time.time()

# argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='image')
args = vars(ap.parse_args())


lpd = LicensePlateDetector(
    pth_weights='./cfg/yolov4-custom_final.weights',
    pth_cfg='./cfg/yolov4_custom.cfg',
    pth_model='./cfg/letters_numbers_model_keras_datagen_cargo2_mandator_v1.h5',
    pth_classes='./cfg/classes.txt'
)

img = args["image"]
p = lpd.detector(img)
print(p)
print('Time %s seconds : '% (time.time() - start_time))
