import numpy as np
from PIL import ImageGrab
import cv2
import time
 
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked
 
 
def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
 
    vertices =  np.array([[10,500], [10,300], [300,200],[500,200], [800,300], [800,500]])
    processed_img = roi(processed_img, [vertices])
 
    return processed_img
 
 
while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
 
    new_screen = process_img(screen)
 
    cv2.imshow('Masked_data', new_screen)
 
    # 'q'키를 누르면 종료합니다
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
