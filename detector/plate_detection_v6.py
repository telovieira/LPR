#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:15:18 2021

@author: telo
"""

##https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/S
## tensorflow==2.6.0

## wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz
## ./configure --enable-optimizations
## make -j 2
## make install

## apt update
## apt install libopencv-dev python3-opencv
## pip install opencv-contrib-python==3.4.13.47 --force-reinstall

#import argparse
import cv2
import numpy as np
import string
import os
#import imutils
from datetime import datetime
import time
#import concurrent.futures
import logging
import logging.config
import queue
import json
import configparser
import functools

# Define constants
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TARGET_WIDTH = 55
TARGET_HEIGHT = 80

chars = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
    ]

# Log File Path
logging.config.fileConfig(fname='./cfg/filelog.conf', disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger(__name__)

# Config File path
configParser = configparser.RawConfigParser()
configFilePath = './cfg/config.conf'
configParser.read(configFilePath)

# Parameters config
pth_weights = configParser.get('config', 'pth_weights')
pth_cfg = configParser.get('config', 'pth_cfg')
plate_confidence = np.float64(configParser.get('config', 'plate_confidence'))
box_confidence = np.float64(configParser.get('config', 'box_confidence'))
moto_h_w_confidence = np.float64(configParser.get('config', 'moto_h_w_confidence'))

# CNN Yolov4 and Character recognition
net = cv2.dnn.readNetFromDarknet(pth_cfg, pth_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#logging.debug("A Debug Logging Message")
#logging.info("A Info Logging Message")
#logging.warning("A Warning Logging Message")
#logging.error("An Error Logging Message")
#logging.critical("A Critical Logging Message")


# Loadin and processing image
class LoadProcessImage():
    def __init__(self, frame, conf, name='load-process-image'):
        self.frame = frame
        self.conf = conf
        self.plate = []
        self.timing = 0
        self.is_moto = False

        logger.info("Started Load Process Image")
        dtn = datetime.now()
        start_time = time.time()
        ln = net.getLayerNames()
        output_layer_names = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        frame = cv2.imread(self.frame)
        if frame is not None:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layer_names)
            outputs = np.vstack(outputs)
            H, W = frame.shape[:2]
            boxes = []
            confidences = []
            classIDs = []
            for output in outputs:
                scores = output[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.conf:
                    x, y, w, h = output[:4] * np.array([W, H, W, H])
                    p0 = int(x - w//2), int(y - h//2)
                    #p1 = int(x + w//2), int(y + h//2)
                    boxes.append([*p0, int(w), int(h)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, 1-self.conf)
            if len(indices) > 0:
                plate = []
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    confidence = round(confidences[i],2)
                    if confidence >= plate_confidence: #box_confidence

                        ## Detect car or moto based on plate ratio height / width
                        #self.frame,dt,ind = (frame[y:y + h, x:x + w], dtn, i)
                        #cv2.imwrite('/mydrive/trdg/img_cropped/'+str(dtn)+'.jpeg',self.frame)
                        #self.timing = time.time() - start_time
                        #logger.info("Plate Box Confidence: {},  Datetime: {}, Box Index: {},  Timing: {}".format(confidence,dtn,i,self.timing))
                        #start_time = time.time()
                        # Plate segmentation and chard identification
                        plate.append((x,y,chars[classIDs[i]],confidences[i]))
                        #self.plate.append(plate)
            def compare(rect1, rect2):
              if abs(rect1[1] - rect2[1]) > 21:
                self.is_moto = True
                return rect1[1] - rect2[1]
              else:
                return rect1[0] - rect2[0]
            self.plate = sorted(plate, key=functools.cmp_to_key(compare))
        self.timing = time.time() - start_time
        logger.info("Plate#: {}, Datetime: {}, Timing: {}".format(self.plate,dtn,self.timing))
        logger.info("Stopped Load Process Image")

    # JSON output
    def to_json(self):
      js = {'date':str(datetime.now()),'timing':self.timing, 'plates':[]}
      if self.plate != 0:
        car = lambda x : "Moto" if self.is_moto else "Car/Truck"
        a = ''.join([percent[2] for ind, percent in enumerate(self.plate)])
        p = str(np.mean([percent[3] for ind, percent in enumerate(self.plate)]))
        js['plates'].append({'plate':a,'confidence':p,'vehicle': car(self.is_moto)})
      return js #json.dumps(js,indent=4)
