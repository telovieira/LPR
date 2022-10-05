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
pth_model = configParser.get('config', 'pth_model')
plate_confidence = np.float64(configParser.get('config', 'plate_confidence'))
box_confidence = np.float64(configParser.get('config', 'box_confidence'))
moto_h_w_confidence = np.float64(configParser.get('config', 'moto_h_w_confidence'))
url_webcam = configParser.get('config', 'url_webcam')
if url_webcam == "0":
    url_webcam = int(url_webcam)

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
                        #cv2.imwrite('/mydrive/trdg/img_cropped/'+str(dtn)+'.jpeg',self.frame)
                        self.timing = time.time() - start_time
                        logger.info("Plate Box Confidence: {},  Datetime: {}, Box Index: {}, Car: {}, Timing: {}".format(confidence,dtn,i,self.moto_img,self.timing))
                        start_time = time.time()
                        # Plate segmentation and chard identification
                        plate = self.plate_segmentation()
                        if len(plate) < 7:
                          plate = self.plate_segmentation2()
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
        return js #json.dumps(js,indent=4)

    # Change cut values according to moto or car plate
    def skip(self,a, b, c, d):
      skip_list = [self.moto_img,(2.9,3.31),(1,1),(4.8,4.8),(0,28),(100,100)]
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

    def plate_segmentation2(self):

      # Read the image and convert to grayscale
      image = self.frame
      height, width, _ = image.shape

      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Apply Gaussian blur and thresholding (normal or adaptive)
      blurred = cv2.GaussianBlur(gray, (5, 5), 0)
      thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

      # Perform connected components analysis on the thresholded images and
      # initialize the mask to hold only the components we are interested in
      (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

      mask = np.zeros(thresh.shape, dtype="uint8")

      # Loop over the unique components
      for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
          continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255

        # Ensure image ratio (h/height)
        if 0.29 < h/height < 0.69 and w > 3 and x > 2 and y > 0 and x < width - 10: #0.29 < h/height < 0.55 and w > 3 and x > 0 and y > 0:
          mask = cv2.add(mask, labelMask)

      # Find contours and get bounding box for each contour
      ret_img, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      boundingBoxes = [cv2.boundingRect(c) for c in cnts]

      # Sort the bounding boxes from left to right, top to bottom
      # sort by Y first, and then sort by X if Ys are similar
      def compare(rect1, rect2):
          if abs(rect1[1] - rect2[1]) > 10:
              return rect1[1] - rect2[1]
          else:
              return rect1[0] - rect2[0]
      boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

      vehicle_plate = []
      # Loop over the bounding boxes
      for rect in boundingBoxes:

          # Get the coordinates from the bounding box
          x,y,w,h = rect
          height, width = mask.shape

          #heigh_h = height / float(h)
          #ratio = h / float(w)
          #width_w = width / float(w)
          #area = h * w
          #if self.skip(heigh_h, ratio, width_w, area): continue

          # Crop the character from the mask
          # and apply bitwise_not because in our training data for pre-trained model
          # the characters are black on a white background
          crop = mask[y:y+h, x:x+w]
          crop = cv2.bitwise_not(crop)

          # Convert and resize image
          crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
          crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT), interpolation = cv2.INTER_CUBIC)
          crop = np.array([crop])

          # Make prediction
          char_recognized = char_classifier.predict(crop)
          prob = char_recognized.max(axis = 1)[0]
          idx = char_recognized.argmax(axis = 1)[0]
          vehicle_plate.append((chars[idx],prob))

      return self.detector_tuning(vehicle_plate)

    # Plate segmentation and char identification
    def plate_segmentation(self):
        try:
            # get image frame detect on yolov4 algorithm
            box = self.frame

            # grayscale region within bounding box
            gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)

            # resize image to 5 times as large as original for better readability
            scale_percent = 500 # percent of original size
            width = int(box.shape[1] * scale_percent / 100)
            height = int(box.shape[0] * scale_percent / 100)
            dim = (width, height)
            rs = cv2.resize(gray, dim, interpolation = cv2.INTER_CUBIC)

            # perform gaussian blur to smoothen image
            gblur = cv2.GaussianBlur(rs, (3,3), 0)

            # threshold the image using Otsus method to preprocess for tesseract
            ret, thresh = cv2.threshold(gblur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            #_,labels = cv2.connectedComponents(thresh)

            # create rectangular kernel for dilation
            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

            # apply dilation to make regions more clear
            dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

            # find contours of regions of interest within license plate
            ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # sort contours left-to-right
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            # create copy of gray image
            im2 = dilation.copy()

            # create blank string to hold license plate number
            vehicle_plate = []

            # loop through contours and find individual letters and numbers in license plate
            for cnt in sorted_contours:
                x,y,w,h = cv2.boundingRect(cnt)
                height, width = im2.shape

                # if height of box is not tall enough relative to total height then skip
                heigh_h = height / float(h)

                # if height to width ratio is less than XX skip
                ratio = h / float(w)

                # if width is not wide enough relative to total width then skip
                width_w = width / float(w)

                # if area is less than 100 pixels skip
                area = h * w

                # skip
                if self.skip(heigh_h, ratio, width_w, area): continue
                if not 0.37 < h/height < 0.69: continue

                # grab character region of image
                delta = 6
                roi = thresh[y-delta:y+h+delta, x-delta:x+w+delta]

                # perform bitwise_not to flip image to black text on white background
                roi = cv2.bitwise_not(roi)

                # perform another blur on character region
                roi = cv2.medianBlur(roi, 1)

                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                roi = cv2.resize(roi, (TARGET_WIDTH, TARGET_HEIGHT), interpolation = cv2.INTER_CUBIC)
                roi = np.array([roi])

                # Make prediction
                char_recognized = char_classifier.predict(roi)
                prob = char_recognized.max(axis = 1)[0]
                idx = char_recognized.argmax(axis = 1)[0]
                vehicle_plate.append((chars[idx],prob))

        except:
            return []
        if self.moto_img:
          odd = [val for ind, val in enumerate(vehicle_plate) if ind % 2 != 0]
          even = [val for ind, val in enumerate(vehicle_plate) if ind % 2 == 0]
          vehicle_plate = odd + even
        return self.detector_tuning(vehicle_plate)

    # Make correction in some identification char according to their plate position
    def detector_tuning(self, x):
      out = []
      for j, k in enumerate(x):
        if j in (3,5,6):
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
