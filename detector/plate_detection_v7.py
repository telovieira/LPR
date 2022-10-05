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
from tensorflow.keras.models import load_model
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
logging.config.fileConfig(fname='/content/gdrive/MyDrive/license_plates/cfg/filelog.conf', disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger(__name__)

# Config File path
configParser = configparser.RawConfigParser()
configFilePath = '/content/gdrive/MyDrive/license_plates/cfg/config.conf'
configParser.read(configFilePath)

# Parameters config
pth_weights = configParser.get('config', 'pth_weights')
pth_cfg = configParser.get('config', 'pth_cfg')
pth_model = configParser.get('config', 'pth_model')
plate_confidence = np.float64(configParser.get('config', 'plate_confidence'))
box_confidence = np.float64(configParser.get('config', 'box_confidence'))
moto_h_w_confidence = np.float64(configParser.get('config', 'moto_h_w_confidence'))


# CNN Yolov4 and Character recognition
net = cv2.dnn.readNetFromDarknet(pth_cfg, pth_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Classifier char model
char_classifier = load_model(pth_model)

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
        self.is_moto = []
        self.moto_img = False
        self.is_mercosul = False

        logger.info("Started Load Process Image")
        start_time = time.time()
        ln = net.getLayerNames()
        output_layer_names = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        frame = cv2.imread(self.frame)
        if frame is not None:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
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
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.conf)
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    confidence = round(confidences[i],2)
                    if confidence >= box_confidence:
                        dtn = datetime.now()
                        ## Detect car or moto based on plate ratio height / width
                        moto = lambda x:  1 if x > moto_h_w_confidence else 0
                        self.moto_img = moto(h/w)
                        self.is_moto.append(self.moto_img)
                        self.frame,dt,ind = (frame[y:y + h, x:x + w], dtn, i)
                        self.is_mercosul = self.mercosul(self.frame)
                        #cv2.imwrite('/mydrive/trdg/img_cropped/'+str(dtn)+'.jpeg',self.frame)
                        self.timing = time.time() - start_time
                        logger.info("Plate Box Confidence: {},  Datetime: {}, Box Index: {}, Car: {}, Timing: {}".format(confidence,dtn,i,self.moto_img,self.timing))
                        start_time = time.time()
                        # Plate segmentation and chard identification
                        preprocessed_img = self.preprocess_image()
                        crop = self.watershed(preprocessed_img)
                        plate = self.prediction(crop)
                        plate = self.detector_tuning(plate)
                        self.plate.append(plate)
                        timing = time.time() - start_time
                        logger.info("Plate#: {}, Datetime: {}, Timing: {}".format(plate,dt,timing))
                        self.timing += timing
        logger.info("Stopped Load Process Image")

    # JSON output
    def to_json(self):
        js = {'date':str(datetime.now()),'timing':self.timing, 'plates':[]}
        for i in range(len(self.plate)):
            plate = self.plate[i]
            if plate != 0:
              car = lambda x : "Moto" if self.is_moto[i] else "Car/Truck"
              a = ''.join([percent[0] for ind, percent in enumerate(plate)])
              p = str(np.mean([percent[1] for ind, percent in enumerate(plate)]))
              js['plates'].append({'plate':a,'confidence':p, 'vehicle': car(self.is_moto[i])})
        return json.dumps(js,indent=4)

    def preprocess_image(self):
      im = self.frame
      height, width, _ = im.shape
      if self.moto_img == 0:
        im = cv2.resize(im, (990,325),interpolation = cv2.INTER_LINEAR)
        im = im[75:280,50:950]
      else:
        im = cv2.resize(im, (400,325),interpolation = cv2.INTER_LINEAR)
        im = im[80:300,30:370]
      return im

    def prediction(self, crop):
      vehicle_plate = []
      for i in range(len(crop)):
        im3 = cv2.resize(crop[i], (55, 80), interpolation = cv2.INTER_CUBIC)
        im3 = np.array([im3])

        #  Make prediction
        char_recognized = char_classifier.predict(im3)
        prob = char_recognized.max(axis = 1)[0]
        idx = char_recognized.argmax(axis = 1)[0]
        if prob >=0.0:
          vehicle_plate.append((chars[idx],prob))
      return vehicle_plate

    def watershed(self, box):
        gray = cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform ,0.1*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        markers = cv2.watershed(box,markers)
        box[markers == -1] = [0,0,0]
        #markers = np.zeros(dist_transform.shape, dtype=np.int32)
        dist_8u = sure_fg.astype('uint8')
        try:
          contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
          ret2, contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        def compare(rect1, rect2):
          if abs(rect1[1] - rect2[1]) > 50:
              return rect1[1] - rect2[1]
          else:
              return rect1[0] - rect2[0]
        sorted_contours = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
        crop = []
        delta = 5
        for cnt in sorted_contours:
          x,y,w,h = cnt

          if self.is_moto == 0:
            if not (2900 < w*h < 22000): continue
            if h < 100: continue
          else:
            if not (2900 < w*h < 22000): continue
            if h < 60: continue

          if x - delta > 0:
            m = thresh[y-delta:y+h+delta,x-delta:x+w+delta]
          else:
            m = thresh[y-delta:y+h+delta,x:x+w+delta]
          m = cv2.bitwise_not(m)
          m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)

          crop.append(m)

        return crop

    def mercosul(self, img):
      ## mask of blue color
      temp = img[0:25,0:800]
      ## https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/
      ## H :[100, 130], S: [100, 255], and V: [20, 255] =>> Blue
      hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
      lower_blue = np.array([100,100,20])
      upper_blue = np.array([130,255,255])
      mask = cv2.inRange(hsv, lower_blue, upper_blue)
      res = cv2.bitwise_and(temp,temp, mask= mask)
      if np.sum(res) > 9350:
        return True
      return False

    # Make correction in some identification char according to their plate position
    def detector_tuning(self, x):
      out = []
      if self.is_mercosul:
        for j, k in enumerate(x):
          if j in (0,1,2,4):
            if k[0] == '1':
              out.append(('I',k[1]))
            elif k[0] == '0':
              out.append(('O',k[1]))
            else:
              out.append(k)
          elif j in (3,5,6):
            if k[0] == 'I':
              out.append(('1',k[1]))
            elif k[0] == 'O':
              out.append(('0',k[1]))
            else:
              out.append(k)
      else:
        for j, k in enumerate(x):
          if j in (0,1,2):
            if k[0] == '1':
              out.append(('I',k[1]))
            elif k[0] == '0':
              out.append(('O',k[1]))
            else:
              out.append(k)
          elif j in (3,4,5,6):
            if k[0] == 'I':
              out.append(('1',k[1]))
            elif k[0] == 'O':
              out.append(('0',k[1]))
            else:
              out.append(k)
      return out
