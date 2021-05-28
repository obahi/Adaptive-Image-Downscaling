import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_image

detector = cv2.CascadeClassifier('haarcascade_number.xml')
img = cv2.imread('Data/img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = gray

detection = detector.detectMultiScale(img,scaleFactor=1.2,minNeighbors = 5, minSize=(25,25))

for (x,y,w,h) in detection:
    carplate_img = img[y+15:y+h-10 ,x+15:x+w-20]

carplate_extract_img = enlarge_img(carplate_img, 150)

carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img,3)

print(pytesseract.image_to_string(carplate_extract_img_gray_blur,
                                  config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))