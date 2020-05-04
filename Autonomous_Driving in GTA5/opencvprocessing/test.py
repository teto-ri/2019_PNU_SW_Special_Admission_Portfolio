import numpy as np
from PIL import ImageGrab
import cv2
 
# 무한루프를 돌면서 
while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen,(80,60))
    cv2.imshow('pygta5-1', screen)
 
    # 'q'키를 누르면 종료합니다
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
