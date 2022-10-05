from detector.plate_detection_v2 import LicensePlateDetector
import argparse
import time

start_time = time.time()

# argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument('-u', '--url', required=False, help='url webcam', default=0)
args = vars(ap.parse_args())

lpd = LicensePlateDetector(
    pth_weights='/Users/telo/license_plates/cfg/yolov4_custom_best.weights',
    pth_cfg='/Users/telo/license_plates/cfg/yolov4_custom.cfg',
    pth_model='/Users/telo/license_plates/cfg/letters_numbers_model_aug_v2.h5',
    pth_classes='/Users/telo/license_plates/cfg/classes.txt'
)

url = args["url"]
#p = lpd.detector_webcam(0)
lpd.detect_plate_webcam(url)

print('Time %s seconds : '% (time.time() - start_time))
