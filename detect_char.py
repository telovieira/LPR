from detector.plate_detection import CharacterDetection
import time
import argparse

start_time = time.time()

# argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='image')
args = vars(ap.parse_args())


lpd = CharacterDetection(
    pth_model='/Users/telo/license_plates/cfg/letters_numbers_model_aug.h5',
    pth_image=args["image"]
)


p=lpd.detector()
print(p)
print('Time %s seconds : '% (time.time() - start_time))
