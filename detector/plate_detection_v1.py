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

detection_buffer = queue.Queue()

LETTERS = list(string.ascii_uppercase)
NUMBERS = [str(i) for i in range(10)]
mapping_character = {letter : idx+10 for idx,letter in enumerate(LETTERS)}
mapping_character.update({number : idx for idx,number in enumerate(NUMBERS)})
#NUM_CLASSES = len(smapping_character)

logging.config.fileConfig(fname='./cfg/filelog.conf', disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger(__name__)

configParser = configparser.RawConfigParser()
configFilePath = './cfg/config.conf'
configParser.read(configFilePath)

pth_weights = configParser.get('config', 'pth_weights')
pth_cfg = configParser.get('config', 'pth_cfg')
pth_model = configParser.get('config', 'pth_model')
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
        self.run()

    def run(self):
        logger.info("Started Load Process Image")
        start_time = time.time()
        ln = net.getLayerNames()
        output_layer_names = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #print("getUnconnectedOutLayers {}".format(time.time() - start_time))
        #while 1:
        #frame = self.frame.last_frame
        frame = cv2.imread(self.frame)
        if frame is not None:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            #print("setInput = {} sec".format(blob))
            outputs = net.forward(output_layer_names)
            #print("forward = {} sec".format(outputs))
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
                    #cv2.rectangle(frame, p0, p1, WHITE, 1)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.conf-0.1)
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    confidence = round(confidences[i],2)
                    if confidence >= 0.8:
                        dtn = datetime.now()
                        #cv2.imshow("Rectangle",frame[y:y + h, x:x + w])
                        #cv2.waitKey(0)
                        #plt.imshow(frame[y:y + h, x:x + w])
                        detection_buffer.put((frame[y:y + h, x:x + w], dtn, i))
                        #print("frame {}".format(frame[y:y + h, x:x + w]))
                        self.timing = time.time() - start_time
                        logger.info("Plate Box Confidence: {},  Datetime: {}, Box Index: {}, Timing: {}".format(confidence,dtn,i,self.timing))
                        if not detection_buffer.empty():
                            start_time = time.time()
                            self.frame,dt,ind = detection_buffer.get()
                            plate = self.detector_v2()
                            self.plate.append(plate)
                            timing = time.time() - start_time
                            logger.info("Plate#: {}, Datetime: {}, Timing: {}".format(plate,dt,timing))
                            self.timing += timing
                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
                        #text = "{}: {:.4f}".format("LP ", confidences[i])
                        #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        logger.info("Stopped Load Process Image")

# Define the thread that will continuously segment plate
#class PlateSegmentation():
#    def __init__(self, name='plate-segmentation'):
#        self.frame = None
#        self.plate=[]
#        self.timing = 0
#        self.run()


#    def run2(self):
#        logger.info("Started Plate Segmentation Process")
#        start_time = time.time()
#        #while 1:
#        if not detection_buffer.empty():
#            self.frame,dt,ind = detection_buffer.get()
#            plate = self.detector_v2()
#            #a = ''.join([percent[0] for plate, percent in enumerate(plate)])
#            #p = np.mean([percent[1] for plate, percent in enumerate(plate)])
#            self.plate.append(plate)
#            self.timing = time.time() - start_time
#            timing_plate = self.timing
#            logger.info("Plate#: {}, Datetime: {}, Timing: {}".format(plate,dt,self.timing))
#            #logger.info("Write Plate Image: lp-{}-{}.png".format(plate[0],dt))
#            #cv2.imwrite("./img/lp-"+plate[0]+"-"+str(dt).replace(" ", "-").replace(":", ".")+".png", self.frame)
#            #print(plate[0],plate[1],dt,ind)
#        logger.info("Stopped Plate Segmentation Process")

    def to_json(self):
        js = {'date':str(datetime.now()),'timing':self.timing, 'plates':[]}
        if len(self.plate) > 0:
            for i in range(len(self.plate)):
                plate = self.plate[i]
                a = ''.join([percent[0] for plate, percent in enumerate(plate)])
                p = str(np.mean([percent[1] for plate, percent in enumerate(plate)]))
                js['plates'].append({'plate':a,'confidence':p})
        return json.dumps(js,indent=4)

    def get_key(self, val):
        #for key, value in mapping_character.items():
        #     if val == value:
        #         return key
        return [key for key, value in mapping_character.items() if val == value][0]
        #return '?'

    def get_percentage(self, val):
        #if val[0] < 0.80:
        #    return 0
        #else:
        #    return val[0]
        return val[0]

    def reco_char_cnn(self, img):
        image_data = img
        image_data = cv2.resize(image_data, (55, 80), interpolation = cv2.INTER_CUBIC)
        image_data = np.array([image_data])
        #image_data = np.stack(image_data, axis = 0)
        char_recognized = char_classifier.predict(image_data)
        return self.get_key(char_recognized.argmax(axis = 1)),char_recognized.max(axis = 1)[0]
        #return self.get_key(char_recognized.argmax(axis = 1)),self.get_percentage(char_recognized.max(axis = 1))


    def plate_segmentation(self):
        try:
            box = self.frame
            #plt.imshow(box)
            #plt.show()
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
                height, width = im2.shape

                # if height of box is not tall enough relative to total height then skip
                #if height / float(h) > 6: continue
                if height / float(h) > 3: continue
                #print(h, height / float(h))

                ratio = h / float(w)
                # if height to width ratio is less than 1.2 skip
                if ratio < 1.2: continue

                # if width is not wide enough relative to total width then skip
                if width / float(w) > 17: continue

                area = h * w
                # if area is less than 100 pixels skip
                if area < 100: continue

                # draw the rectangle
                #rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
                #cv2.imshow("Rectangle1",im2)
                #cv2.waitKey(0)

                # grab character region of image
                delta = 6
                roi = thresh[y-delta:y+h+delta, x-delta:x+w+delta]
                #cv2.imshow("Rectangle",roi)
                #cv2.waitKey(0)

                # perform bitwise not to flip image to black text on white background
                roi = cv2.bitwise_not(roi)
                #cv2.imshow("Rectangle",roi)
                #cv2.waitKey(0)

                # perform another blur on character region
                roi = cv2.medianBlur(roi, 1)
                #cv2.imshow("Rectangle",roi)
                #cv2.waitKey(0)

                try:
                    #plate_num.append(cv2.cvtColor(im2[y-delta:y+h+delta,x-delta:x+w+delta],cv2.COLOR_GRAY2RGB))
                    #cv2.imshow("Rectangle",roi)
                    #cv2.waitKey(0)
                    plate_num.append(cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB))
                except:
                    plate_num.append("?")
        except:
            return []
        return plate_num

    def detector_v2(self):
        ch = 0
        plates = self.plate_segmentation()
        if len(plates) > 0:
            ch = [self.reco_char_cnn(plates[i]) for i in range(len(plates))]
        return ch

    def detector(self):
        #start_time = time.time()
        pc, ch = 0, ''
        plates = self.plate_segmentation()
        if len(plates) > 0:
            for i in range(len(plates)):
                plate = self.reco_char_cnn(plates[i])
                ch += plate[0]
                pc += plate[1]
            pc = pc/len(plates)
        #print("recno char = {}".format(time.time() - start_time))
        return ch,pc
