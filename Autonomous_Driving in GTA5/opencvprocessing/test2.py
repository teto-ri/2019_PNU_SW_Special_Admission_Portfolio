import numpy as np
from PIL import ImageGrab
import cv2
 
while(True):
    # (0,40)부터 (800,600)좌표까지 데이터를 저장하고 screen에 저장
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
 
    # 창을 생성하고 이 창에 screen이미지를 띄움
    cv2.imshow('gtaVtestbot', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
 
    # q키를 누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
