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
import threading
#import imutils
from datetime import datetime
#import concurrent.futures
import logging
import logging.config
import queue

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

detection_buffer = queue.Queue()

logging.config.fileConfig(fname='./cfg/filelog.conf', disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger(__name__)

#logging.debug("A Debug Logging Message")
#logging.info("A Info Logging Message")
#logging.warning("A Warning Logging Message")
#logging.error("An Error Logging Message")
#logging.critical("A Critical Logging Message")

# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.daemon = True
        self.start()

    def run(self):
        logger.info("Started Camera Thread")
        while 1:
            ret, self.last_frame = self.camera.read()

# Define the thread that will continuously process frame
class LoadProcessImage(threading.Thread):
    def __init__(self, frame, cfg, weights, classes, conf, name='load-process-image'):
        self.frame = frame
        self.conf = conf
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        super(LoadProcessImage, self).__init__(name=name)
        self.daemon = True
        self.start()

    def run(self):
        logger.info("Started Process Image Thread")
        ln = self.net.getLayerNames()
        output_layer_names = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        while 1:
            frame = self.frame.last_frame
            if frame is not None:
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outputs = self.net.forward(output_layer_names)
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
                        if confidence >= 0.9:
                            dtn = datetime.now()
                            detection_buffer.put((frame[y:y + h, x:x + w], dtn, i))
                            logger.info("Plate Box Confidence: {},  Datetime: {}, Box Index: {}".format(confidence,dtn,i))

                            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
                            #text = "{}: {:.4f}".format("LP ", confidences[i])
                            #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        logger.info("Stopped Camera Thread")

# Define the thread that will continuously segment plate
class PlateSegmentation(threading.Thread):
    def __init__(self, pth_model, name='plate-segmentation'):

        self.char_classifier = load_model(pth_model)
        self.frame = None

        self.LETTERS = list(string.ascii_uppercase)
        self.NUMBERS = [str(i) for i in range(10)]
        self.mapping_character = {letter : idx+10 for idx,letter in enumerate(self.LETTERS)}
        self.mapping_character.update({number : idx for idx,number in enumerate(self.NUMBERS)})
        self.NUM_CLASSES = len(self.mapping_character)

        super(PlateSegmentation, self).__init__(name=name)
        self.daemon = True
        self.start()

    def run(self):
        logger.info("Started Plate Segmentation Thread")
        while 1:
            if not detection_buffer.empty():
                self.frame,dt,ind = detection_buffer.get()
                plate = self.detector()
                logger.info("Plate#: {}, Confidence: {}, Datetime: {}".format(plate[0],plate[1],dt))
                logger.info("Write Plate Image: lp-{}-{}.png".format(plate[0],dt))
                cv2.imwrite("./img/lp-"+plate[0]+"-"+str(dt).replace(" ", "-").replace(":", ".")+".png", self.frame)
                #print(plate[0],plate[1],dt,ind)

    def get_key(self, val):
        for key, value in self.mapping_character.items():
             if val == value:
                 return key
        #return [key for key, value in self.mapping_character.items() if val == value][0]

        return '?'

    def get_percentage(self, val):
        if val[0] < 0.80:
            return 0
        else:
            return val[0]

    def reco_char_cnn(self, img):
        image_data = img
        image_data = cv2.resize(image_data, (55, 80), cv2.INTER_AREA)
        image_data = np.array([image_data])
        image_data = np.stack(image_data, axis = 0)
        char_recognized = self.char_classifier.predict(image_data)
        return self.get_key(char_recognized.argmax(axis = 1)),self.get_percentage(char_recognized.max(axis = 1))

    def whatershed(self):
        box = self.frame
        gray = cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
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
        box[markers == -1] = [255,0,0]
        return box

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
            #dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
            dilation = cv2.bitwise_not(opening)
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
                if height / float(h) > 6: continue

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
                #cv2.imshow("Rectangle",im2)
                #cv2.waitKey(0)

                # grab character region of image
                delta = 6
                roi = thresh[y-delta:y+h+delta, x-delta:x+w+delta]

                # perform bitwise not to flip image to black text on white background
                roi = cv2.bitwise_not(roi)

                # perform another blur on character region
                roi = cv2.medianBlur(roi, 1)
                try:
                    plate_num.append(cv2.cvtColor(im2[y-delta:y+h+delta,x-delta:x+w+delta],cv2.COLOR_GRAY2RGB))
                except:
                    plate_num.append("?")
        except:
            return []
        return plate_num

    def detector(self):
        pc, ch = 0, ''
        plates = self.plate_segmentation()
        if len(plates) > 0:
            for i in range(len(plates)):
                plate = self.reco_char_cnn(plates[i])
                ch += plate[0]
                pc += plate[1]
            pc = pc/len(plates)
        return ch,pc
