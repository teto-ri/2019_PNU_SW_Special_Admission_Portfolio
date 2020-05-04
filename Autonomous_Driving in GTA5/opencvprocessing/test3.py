
import numpy as np
from PIL import ImageGrab
import cv2
import time
 

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
 
    return processed_img
 
 
while(True):
    
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
 
    
    new_screen = process_img(screen)
 
    
    cv2.imshow('pygta5-2', new_screen)
 
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
