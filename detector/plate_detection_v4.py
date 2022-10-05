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

#import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LETTERS = list(string.ascii_uppercase)
NUMBERS = [str(i) for i in range(10)]
mapping_character = {letter : idx+10 for idx,letter in enumerate(LETTERS)}
mapping_character.update({number : idx for idx,number in enumerate(NUMBERS)})
#NUM_CLASSES = len(smapping_character)

logging.config.fileConfig(fname='/content/gdrive/MyDrive/license_plates/cfg/filelog.conf', disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger(__name__)

configParser = configparser.RawConfigParser()
configFilePath = '/content/gdrive/MyDrive/license_plates/cfg/config.conf'
configParser.read(configFilePath)

pth_weights = configParser.get('config', 'pth_weights')
pth_cfg = configParser.get('config', 'pth_cfg')
pth_model = configParser.get('config', 'pth_model')
plate_confidence = np.float64(configParser.get('config', 'plate_confidence'))
box_confidence = np.float64(configParser.get('config', 'box_confidence'))
moto_h_w_confidence = np.float64(configParser.get('config', 'moto_h_w_confidence'))
#pth_classes = configParser.get('config', 'pth_classes')
url_webcam = configParser.get('config', 'url_webcam')
if url_webcam == "0":
    url_webcam = int(url_webcam)

net = cv2.dnn.readNetFromDarknet(pth_cfg, pth_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

char_classifier = load_model(pth_model)

#logging.debug("A Debug Logging Message")
#logging.info("A Info Logging Message")
#logging.warning("A Warning Logging Message")
#logging.error("An Error Logging Message")
#logging.critical("A Critical Logging Message")


# Define the thread that will continuously process frame
class LoadProcessImage():
    def __init__(self, frame, conf, name='load-process-image'):
        self.frame = frame
        self.conf = conf
        self.plate = []
        self.timing = 0
        self.is_moto = []
        self.moto_img = False
        self.run()

    def run(self):
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
                    boxes.append([*p0, int(w), int(h)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.conf-0.1)
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    confidence = round(confidences[i],2)
                    if confidence >= box_confidence:
                        dtn = datetime.now()
                        ## Detect car or moto based on ratio height / width
                        moto = lambda x: 1 if x > moto_h_w_confidence else 0
                        self.moto_img = moto(h/w)
                        self.is_moto.append(self.moto_img)
                        self.frame,dt,ind = (frame[y:y + h, x:x + w], dtn, i)
                        self.timing = time.time() - start_time
                        logger.info("Plate Box Confidence: {},  Datetime: {}, Box Index: {},  Car: {}, Timing: {}".format(confidence,dtn,i,self.moto_img,self.timing))
                        start_time = time.time()
                        plate = self.detector_v2()
                        self.plate.append(plate)
                        timing = time.time() - start_time
                        logger.info("Plate#: {}, Datetime: {}, Timing: {}".format(plate,dt,timing))
                        self.timing += timing

        logger.info("Stopped Load Process Image")

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

    def get_key(self, val):
        return [key for key, value in mapping_character.items() if val == value][0]

    def get_percentage(self, val):
        if val[0] < 0.80:
            return 0
        else:
            return val[0]
        #return val[0]

    def reco_char_cnn(self, img):
        image_data = img
        image_data = cv2.resize(image_data, (55, 80), interpolation = cv2.INTER_CUBIC)
        image_data = np.array([image_data])
        #image_data = np.stack(image_data, axis = 0)
        char_recognized = char_classifier.predict(image_data)
        return self.get_key(char_recognized.argmax(axis = 1)),char_recognized.max(axis = 1)
        #return self.get_key(char_recognized.argmax(axis = 1)),self.get_percentage(char_recognized.max(axis = 1))

    def skip(self,a, b, c, d):
      skip_list = [self.moto_img,(2.9,2.99),(1,1),(4.8,4.8),(0,28),(100,100)]
      if a > skip_list[1][skip_list[0]]:
        return True
      if not skip_list[2][skip_list[0]]<=b<=skip_list[3][skip_list[0]]:
        return True
      if skip_list[4][skip_list[0]] > 0:
        if c > skip_list[4][skip_list[0]]:
          return True
      if d < skip_list[4][skip_list[0]]:
        return True
      return False

    def plate_segmentation(self):
        try:
            box = self.frame
            # grayscale region within bounding box
            gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
            # resize image to 5 times as large as original for better readability
            scale_percent = 500 # percent of original size
            width = int(box.shape[1] * scale_percent / 100)
            height = int(box.shape[0] * scale_percent / 100)
            dim = (width, height)
            gray = cv2.resize(gray, dim, interpolation = cv2.INTER_CUBIC)
            # perform gaussian blur to smoothen image
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            # threshold the image using Otsus method to preprocess for tesseract
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            # create rectangular kernel for dilation
            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rect_kern)

            # apply dilation to make regions more clear
            dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
            #dilation = cv2.bitwise_not(opening)
            # find contours of regions of interest within license plate
            try:
                contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # sort contours left-to-right
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            # create copy of gray image
            #im2 = gray.copy()
            im2 = dilation.copy()

            # create blank string to hold license plate number
            plate_num = []
            # loop through contours and find individual letters and numbers in license plate
            for cnt in sorted_contours:
                x,y,w,h = cv2.boundingRect(cnt)
                #rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (255,0,0),6)
                height, width = im2.shape

                # if height of box is not tall enough relative to total height then skip
                heigh_h = height / float(h)

                # if height to width ratio is less than 1.2 skip
                ratio = h / float(w)

                # if width is not wide enough relative to total width then skip
                width_w = width / float(w)

                # if area is less than 100 pixels skip
                area = h * w

                if self.skip(heigh_h, ratio, width_w, area): continue

                # draw the rectangle
                #rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)

                # grab character region of image
                delta = 6
                roi = thresh[y-delta:y+h+delta, x-delta:x+w+delta]

                # perform bitwise not to flip image to black text on white background
                roi = cv2.bitwise_not(roi)

                # perform another blur on character region
                roi = cv2.medianBlur(roi, 1)

                try:
                    #plate_num.append(cv2.cvtColor(im2[y-delta:y+h+delta,x-delta:x+w+delta],cv2.COLOR_GRAY2RGB))
                    plate_num.append(cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB))
                except:
                    plate_num.append("?")
        except:
            return []
        return plate_num

    def detector_tuning(self, x):
      out = []
      for j, k in enumerate(x):
        if j in (3,4,5,6):
          if k[0] == 'I':
            out.append(('1',k[1]))
          elif k[0] == 'O':
            out.append(('0',k[1]))
          else:
            out.append(k)
        elif j in (0,1,2):
          if k[0] == '1':
            out.append(('I',k[1]))
          elif k[0] == '0':
            out.append(('O',k[1]))
          else:
            out.append(k)
        else:
          out.append(k)
      return out

    def detector_v2(self):
        ch = []
        plates = self.plate_segmentation()
        if len(plates) > 0:
            for j, k in enumerate(plates):
              x = self.reco_char_cnn(k)
              if x[1] >= plate_confidence:
                ch.append(x)
            if self.moto_img:
              odd = [val for ind, val in enumerate(ch) if ind % 2 != 0]
              even = [val for ind, val in enumerate(ch) if ind % 2 == 0]
              ch = odd + even
        return self.detector_tuning(ch)
