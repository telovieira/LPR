#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:15:18 2021

@author: telo
"""

## pip install opencv-contrib-python==3.4.13.47 --force-reinstall

##https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/S
## tensorflow==2.6.0

#import argparse
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import string
import os
import threading
import imutils
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()
            
class ClearQueue(threading.Thread):
    def __init__(self, queue, timeout=5, name='clean-queue'):
        self.queue = queue
        self.timeout = timeout
        super(ClearQueue, self).__init__(name=name)
        self.start()

    def run(self):
        while 1:
            time.sleep(self.timeout)
            if len(self.queue) > 0:
                print(self.queue)
                self.queue = []
  
class LicensePlateDetector:

    def __init__(self, pth_weights, pth_cfg, pth_model, pth_classes):
        self.net = cv2.dnn.readNet(pth_cfg, pth_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.char_classifier = load_model(pth_model)
        self.classes = []
        with open(pth_classes, 'r') as f:
            self.classes = f.read().splitlines()
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 0, 0)
        self.coordinates = None
        self.img = None
        #self.fig_image = None
        self.roi_image = None
        
        
        self.LETTERS = list(string.ascii_uppercase)
        self.NUMBERS = [str(i) for i in range(10)]
        self.mapping_character = {letter : idx+10 for idx,letter in enumerate(self.LETTERS)}
        self.mapping_character.update({number : idx for idx,number in enumerate(self.NUMBERS)})
        self.NUM_CLASSES = len(self.mapping_character)
    
    def show_image(self,title, img):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        return
    
    def show_frame(self, title, frame):
        cv2.imshow('The last frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        return
    
    def zoom(self, cv2Object, zoomSize):
        # Resizes the image/video frame to the specified amount of "zoomSize".
        # A zoomSize of "2", for example, will double the canvas size
        cv2Object = imutils.resize(cv2Object, width=(zoomSize * cv2Object.shape[1]))
        # center is simply half of the height & width (y/2,x/2)
        center = (int(cv2Object.shape[0]/2),int(cv2Object.shape[1]/2))
        # cropScale represents the top left corner of the cropped frame (y/x)
        cropScale = (int(center[0]/zoomSize), int(center[1]/zoomSize))
        # The image/video frame is cropped to the center with a size of the original picture
        # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
        # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
        cv2Object = cv2Object[cropScale[0]:(center[0] + cropScale[0]), cropScale[1]:(center[1] + cropScale[1])]
        return cv2Object
    
    def load_image(self, frame, ln):
        global img, outputs
    
        img = frame.copy()
        
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    
        self.net.setInput(blob)
        t0 = time.time()
        outputs = self.net.forward(ln)
        t = time.time() - t0
    
        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)
        
        #cv2.imshow('window',  img)
        #cv2.displayOverlay('window', f'forward propagation time={t:.3}')
        #print(f'forward propagation time={t:.3}')
        #cv2.waitKey(0)
        return outputs
        
    def post_process(self, frame, outputs, conf, queue):
        
        WHITE = (255, 255, 255)
        H, W = frame.shape[:2]
    
        boxes = []
        confidences = []
        classIDs = []
    
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                cv2.rectangle(img, p0, p1, WHITE, 1)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                self.img = frame
                self.coordinates = (x, y, w, h)
                
                confidence = round(confidences[i],2)
                if confidence > 0.90:
                    label = self.detector_webcam()  
                    #if not any( elem == label for elem in queue):
                        #print(label)
                    if len(queue) == 0:
                        queue.append(label)
                    else:
                        pop = queue.pop(0)
                        if label[1] > pop[1]:
                            queue.append(label)
                        else:
                            queue.append(pop)
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                
                cv2.imshow('The last frame',frame)
    
    def detect_plate_webcam(self, video_url=0):
        ln = self.net.getLayerNames()
        output_layer_names = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        camera = cv2.VideoCapture(video_url)
        cam_cleaner = CameraBufferCleanerThread(camera)
        queue = []
        clear_queue = ClearQueue(queue,2)
        #cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        #cap.open(video_url)
        try:
            
            while True:
                if cam_cleaner.last_frame is not None:
                    frame = cam_cleaner.last_frame
                    
                    outputs = self.load_image(frame, output_layer_names)
                    self.post_process(img, outputs, 0.5, queue)
                    
                    #(height, width) = frame.shape[:2]
                    
                    #cv2.imshow('The last frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
                    #blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                    #self.net.setInput(blob)
                    #output_layer_names = self.net.getUnconnectedOutLayersNames()
                    #layer_outputs = self.net.forward(output_layer_names)
                    #t, _ = self.net.getPerfProfile()
                    #print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))
                    #boxes = []
                    #confidences = []
                    #class_ids = []
    
                    #for output in layer_outputs:
                    #    for detection in output:
                    #        scores = detection[5:]
                    #        class_id = np.argmax(scores) 
                    #        confidence = scores[class_id]
                    #        if confidence > 0.2:
                    #            center_x = int(detection[0] * width)
                    #            center_y = int(detection[1] * height)
                    #            w = int(detection[2] * width)
                    #            h = int(detection[3] * height)
                    #            x = int(center_x - w / 2)
                    #            y = int(center_y - h / 2)
                    #
                    #            boxes.append([x, y, w, h])
                    #            confidences.append((float(confidence)))
                    #            class_ids.append(class_id)
    
                    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

                    #if len(indexes) > 0:
                    #    for i in indexes.flatten():
                    #        x, y, w, h = boxes[i]
                    #        self.img = frame
                    #        self.coordinates = (x, y, w, h)
                    #        #label = str(self.classes[class_ids[i]])
                    #        # Get Character Segmentation
                    #        confidence = round(confidences[i],2)
                    #        if confidence > 0.90:
                    #            label = self.detector_webcam()  
                    #            #if not any( elem == label for elem in queue):
                    #                #print(label)
                    #            if len(queue) == 0:
                    #                queue.append(label)
                    #            else:
                    #                pop = queue.pop(0)
                    #                if label[1] > pop[1]:
                    #                    queue.append(label)
                    #                else:
                    #                    queue.append(pop)
                                    #cv2.rectangle(frame, (x,y), (x + w, y + h), self.color, 3)
                                    #cv2.putText(frame, label+ ' ' + str(confidence), (x, y + 20), self.font, 2, (255, 255, 255), 3)
                                       
                            #cv2.imshow('The last frame',frame)
                            #if cv2.waitKey(1) & 0xFF == ord('q'):
                            #    break
                
        finally:
            camera.release()
            cv2.destroyAllWindows()
        return

    def detect_plate(self, img_path: str):
        self.img = cv2.imread(img_path)
        #self.img = orig
        #img = orig.copy()
        height, width, _ = self.img.shape
        blob = cv2.dnn.blobFromImage(self.img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)
        #boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores) 
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    #boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        #if len(indexes) > 0:
        #    for i in indexes.flatten():
        #        x, y, w, h = boxes[i]
        #        label = str(self.classes[class_ids[i]])
        #        confidence = str(round(confidences[i],2))
        #        cv2.rectangle(self.img, (x,y), (x + w, y + h), self.color, 15)
        #        cv2.putText(self.img, label + ' ' + confidence, (x, y + 20), self.font, 3, (255, 255, 255), 3)
        #self.img = img
        self.coordinates = (x, y, w, h)
        return
    
    
    #def crop_plate(self, plate_type=2):
    #    x, y, w, h = self.coordinates
    #    if plate_type == 0:
    #        ## placa antiga
    #        roi = self.img[y+50:y+50 + 65, x:x + w]
    #    elif plate_type == 1:
    #       ## placa mercosul
    #        roi = self.img[y+70:y + h-10, x+90:x + w-70]
    #    elif plate_type == 2:
    #        # regular
    #        roi = self.img[y:y + h, x:x + w]
    #    self.roi_image = roi
    #    return
    
    
    def get_key(self, val):
        for key, value in self.mapping_character.items():
             if val == value:
                 return key
        #return [key for key, value in self.mapping_character.items() if val == value][0]

        return '?'
    
    def get_percentage(self, val):
        if val[0] < 0.8:
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
    
    def plate_segmentation(self):
        try:
            # separate coordinates 
            x, y, w, h = self.coordinates
            box = self.img[y:y + h, x:x + w]
            # grayscale region within bounding box
            gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
            # resize image to three times as large as original for better readability
            scale_percent = 500 # percent of original size
            width = int(box.shape[1] * scale_percent / 100)
            height = int(box.shape[0] * scale_percent / 100)
            dim = (width, height)
            gray = cv2.resize(gray, dim, interpolation = cv2.INTER_CUBIC)
            #self.show_image('GRAY', gray)
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
                #self.show_image('BOX', im2[y:y+h,x:x+w])
    
                # if height of box is not tall enough relative to total height then skip
                if height / float(h) > 6: continue
                
                ratio = h / float(w)
                # if height to width ratio is less than 1.2 skip
                if ratio < 1.2: continue
                #cv2.imshow("width2", im2[y:y+h,x:x+w])
                #cv2.waitKey(0)
    
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
                #cv2.imshow("ROI",roi)
                #cv2.waitKey(0)
                try:
                  plate_num.append(cv2.cvtColor(im2[y-delta:y+h+delta,x-delta:x+w+delta],cv2.COLOR_GRAY2RGB))
                except: 
                  plate_num.append("?")
        except:
            return []
        return plate_num

    def detector(self, img):
        self.detect_plate(img)
        
        p = ''
        plates = self.plate_segmentation()
        p = [self.reco_char_cnn(plates[i]) for i in range(len(plates))]
        print(p)
        #for i in range(len(plates)):
        #  p += self.reco_char_cnn(plates[i])
        return p
    
    def detector_webcam(self):
        
        ch = ''
        pc = 0
        plates = self.plate_segmentation()
        if len(plates) > 0:
            for i in range(len(plates)):
                plate = self.reco_char_cnn(plates[i])
                #p = p + self.reco_char_cnn(plates[i])[0] + '('+str(self.reco_char_cnn(plates[i])[1])+ ')'
                ch += plate[0]
                pc += plate[1]
            pc = pc/len(plates)
        return ch,pc

class CharacterDetection:
    def __init__(self,  pth_model, pth_image):
        self.char_classifier = load_model(pth_model)
        self.img = cv2.imread(pth_image)
        
        self.LETTERS = list(string.ascii_uppercase)
        self.NUMBERS = [str(i) for i in range(10)]
        self.mapping_character = {letter : idx+10 for idx,letter in enumerate(self.LETTERS)}
        self.mapping_character.update({number : idx for idx,number in enumerate(self.NUMBERS)})
        self.NUM_CLASSES = len(self.mapping_character)
        
    def get_key(self, val):
        #for key, value in self.mapping_character.items():
        #     if val == value:
        #         return key
        return [key for key, value in self.mapping_character.items() if val == value][0]

        #return key

    def reco_char_cnn(self, img):
        image_data = img
        image_data = cv2.resize(image_data, (55, 80), cv2.INTER_AREA)
        image_data = np.array([image_data])
        image_data = np.stack(image_data, axis = 0)
        char_recognized = self.char_classifier.predict(image_data)
        return self.get_key(char_recognized.argmax(axis = 1))
    
    def plate_segmentation(self):
        # separate coordinates 
        box = self.img
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # resize image to three times as large as original for better readability
        gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        # perform gaussian blur to smoothen image
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        #cv2.imshow("Gray", gray)
        #cv2.waitKey(0)
        # threshold the image using Otsus method to preprocess for tesseract
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #cv2.imshow("Otsu Threshold", thresh)
        #cv2.waitKey(0)
        # create rectangular kernel for dilation
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rect_kern)
        #cv2.imshow("opening", opening)
        #cv2.waitKey(0)
        # apply dilation to make regions more clear
        #dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
        dilation = cv2.bitwise_not(opening)
        #cv2.imshow("dilationS", dilation)
        #cv2.waitKey(0) 
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
            ##23.558282208588956
            ##1.4553571428571428
            ##25.714285714285715
            ##18256
            #
            #print(height / float(h))
            #print(h / float(w))
            #print(width / float(w))
            #print(h * w)
            #print('')
            #cv2.imshow("oi", im2[y:y+h,x:x+w])
            #cv2.waitKey(0)
            

            # if height of box is not tall enough relative to total height then skip
            if height / float(h) < 23 or  height / float(h) > 25: continue
            
            ratio = h / float(w)
            # if height to width ratio is less than 1.2 skip
            if ratio < 1.2: continue
            #cv2.imshow("width2", im2[y:y+h,x:x+w])
            #cv2.waitKey(0)

            # if width is not wide enough relative to total width then skip
            if width / float(w) < 17 or width / float(w) > 30: continue 

            area = h * w
            # if area is less than 100 pixels skip
            if area < 100: continue
            #cv2.imshow("oi", im2[y:y+h,x:x+w])
            #cv2.waitKey(0)
            #print(height / float(h))
            #print(h / float(w))
            #print(width / float(w))
            #print(h * w)
            #print('')

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
            #cv2.imshow("ROI",roi)
            #cv2.waitKey(0)
            try:
              plate_num.append(cv2.cvtColor(im2[y-delta:y+h+delta,x-delta:x+w+delta],cv2.COLOR_GRAY2RGB))
            except: 
              plate_num.append("?")
        return plate_num

    def detector(self):   
        p = ''
        plates = self.plate_segmentation()
        p = [self.reco_char_cnn(plates[i]) for i in range(len(plates))]
        #for i in range(len(plates)):
        #  p += self.reco_char_cnn(plates[i])
        return p
    
#lpd = LicensePlateDetector(
#    pth_weights='/content/gdrive/My Drive/license_plates/cfg/yolov4_custom_best.weights', 
#    pth_cfg='/content/gdrive/My Drive/license_plates/cfg/yolov4_custom.cfg', 
#    pth_classes='/content/gdrive/My Drive/license_plates/cfg/classes.txt',
#    pth_model='/content/gdrive/My Drive/license_plates/cfg/letters_numbers_model_aug.h5'
#)
