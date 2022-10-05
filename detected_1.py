from detector.plate_detection_v1 import LoadProcessImage
import argparse
#import configparser
import cv2
import queue
import json
from datetime import datetime
import numpy as np

# argumentos de entrada
ap = argparse.ArgumentParser()
#ap.add_argument('-c', '--config', required=True, help='config', default='./cfg/config.conf')
ap.add_argument('-i', '--image', required=True, help='image')
args = vars(ap.parse_args())

#configParser = configparser.RawConfigParser()
#configFilePath = args["config"]
#configParser.read(configFilePath)

#url = args["url"] ##"rtsp://nathan:SENHA_RTSP@192.168.0.224:554/Streaming/channels/1/"

if __name__ == "__main__":
    #detection_buffer = queue.Queue()
    #cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)

    #pth_weights = configParser.get('config', 'pth_weights')
    #pth_cfg = configParser.get('config', 'pth_cfg')
    #pth_model = configParser.get('config', 'pth_model')
    #pth_classes = configParser.get('config', 'pth_classes')
    #url_webcam = configParser.get('config', 'url_webcam')
    #if url_webcam == "0":
    #    url_webcam = int(url_webcam)

    #pth_weights='/Users/telo/license_plates/cfg/yolov4_custom_best.weights'
    #pth_cfg='/Users/telo/license_plates/cfg/yolov4_custom.cfg'
    #pth_model='/Users/telo/license_plates/cfg/letters_numbers_model_aug_v2.h5'
    #pth_classes='/Users/telo/license_plates/cfg/classes.txt'

    #camera = cv2.VideoCapture(url_webcam)

    #width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = camera.get(cv2.CAP_PROP_FPS)

    #print(width, height, fps)

    #cam_cleaner = CameraBufferCleanerThread(camera)
    img = args["image"]
    #lpi = LoadProcessImage(img, pth_cfg, pth_weights, pth_classes, 0.5)
    #ps = PlateSegmentation(pth_model)
    lpi = LoadProcessImage(img, 0.5)
    #ps = PlateSegmentation()
    #js = {'date':str(datetime.now()),'plates':[]}
    #for i in range(len(ps.plate)):
    #    plate = ps.plate[i]
    #    a = ''.join([percent[0] for plate, percent in enumerate(plate)])
    #    p = str(np.mean([percent[1] for plate, percent in enumerate(plate)]))
    #    js['plates'].append({'plate':a,'confidence':p})
    print(lpi.to_json())
    #for i in range(len(plate)):
    #    print(json.dumps({'date':str(datetime.now()), 'plate':a, 'confidence':p},indent=4))
    #show(img)

    #a = ''.join([percent[0] for plate, percent in enumerate(plate)])
    #p = np.mean([percent[1] for plate, percent in enumerate(plate)])
    #print(a,p)
    #print("program ended")
    #camera.release()
    #cv2.destroyAllWindows()
