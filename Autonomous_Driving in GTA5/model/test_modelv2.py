import numpy as np
import cv2, time, os
from grabscreen import grab_screen
from getkeys import key_check
from PIL import ImageGrab
from alexnet import alexnet
from directkeys import PressKey, ReleaseKey, W,A,S,D

WIDTH = 80
HEIGHT = 60
LR = 1e-3      # Learning Rate
EPOCHS = 8     
 
MODEL_NAME = 'ML-Based-AI-{}-{}-{}-epoch.model'.format(LR, 'alexnetv2', EPOCHS)

# 전진키
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
 
# 좌회전키
def left():
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)
 
# 우회전키
def right():
    PressKey(D)
    ReleaseKey(A)
    PressKey(W)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    
    while(True):

        if not paused:
            screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen,(80,60))

            prediction = mode.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            moves = list(np.around(prediction))
            print(moves, prediction)

            if moves == [1,0,0]:
                left()
            elif moves == [0,1,0]:
                straight()
            elif moves == [0,0,1]:
                right()

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

        

main()

