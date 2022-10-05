from flask import Flask, request
from detector.plate_detection_v1 import LoadProcessImageThread, PlateSegmentationThread, LoadProcessImage, PlateSegmentation
from detector.plate_detection_v4 import LoadProcessImage as LPImage
from werkzeug.utils import secure_filename
from detector.plate_detection import LicensePlateDetector
import json
from flask import jsonify
import time
import os
import configparser
import queue
from datetime import datetime

UPLOAD_FOLDER = '/root/license_plates/img'

configParser = configparser.RawConfigParser()
configFilePath = './cfg/config.conf'
configParser.read(configFilePath)

pth_weights = configParser.get('config', 'pth_weights')
pth_cfg = configParser.get('config', 'pth_cfg')
pth_model = configParser.get('config', 'pth_model')
pth_classes = configParser.get('config', 'pth_classes')
url_webcam = configParser.get('config', 'url_webcam')
if url_webcam == "0":
    url_webcam = int(url_webcam)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


## Uncomment to use Thread
#lpi = LoadProcessImageThread(pth_cfg, pth_weights, pth_classes, 0.5)
#lps = PlateSegmentationThread(pth_model)


lpd = LicensePlateDetector(
    pth_weights=pth_weights,
    pth_cfg=pth_cfg,
    pth_model=pth_model,
    pth_classes=pth_classes
)



@app.route('/')
def home():
    return "License Detector Works!!!"


@app.route('/api/lpr/thread',methods=['POST','GET'])
def get_lpr_thread():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        #lpi.frame = file_path

        return lps.plate
    return "LPR-No Image"

@app.route('/api/lpr/v2',methods=['POST','GET'])
def get_lpr_v2():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        lpi_ = LoadProcessImage(file_path, pth_cfg, pth_weights, pth_classes, 0.5)
        lpsi_ = PlateSegmentation(pth_model)
        os.remove(file_path)
        return lpsi_.to_json() #jsonify (date=str(datetime.now()), plate=lpsi_.plate, confidende=lpsi_.perc)
    if request.method == 'GET':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        lpi_ = LoadProcessImage(file_path, pth_cfg, pth_weights, pth_classes, 0.5)
        lpsi_ = PlateSegmentation(pth_model)
        os.remove(file_path)
        return lpsi_.to_json() #jsonify (date=str(datetime.now()), plate=lpsi_.plate, confidende=lpsi_.perc)
    return "LPR-No Image"

@app.route('/api/lpr/v4',methods=['POST','GET'])
def get_lpr_v4():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        lpimage = LPImage(file_path, 0.5)
        os.remove(file_path)
        return lpimage.to_json() #jsonify (date=str(datetime.now()), plate=lpsi_.plate, confidende=lpsi_.perc)
    if request.method == 'GET':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        lpimage = LPImage(file_path, 0.5)
        os.remove(file_path)
        return lpimage.to_json() #jsonify (date=str(datetime.now()), plate=lpsi_.plate, confidende=lpsi_.perc)
    return "LPR-No Image"


@app.route('/api/lpr',methods=['POST','GET'])
def get_lpr():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        out = lpd.detector(file_path)
        os.remove(file_path)
        return out+'\r\n'

    return "LPR-No Image"
