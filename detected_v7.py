from detector.plate_detection_v7 import LoadProcessImage as LPI
import argparse
import cv2
import queue
import json
from datetime import datetime
import numpy as np

# argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='image')
args = vars(ap.parse_args())

img = args["image"]

lpi = LPI(img, 0.5)

print(lpi.to_json())
