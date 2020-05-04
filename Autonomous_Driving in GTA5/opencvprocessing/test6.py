'''Ruled_based Self driving AI
Author: Shin_PC, Shin Byenog Geun
reference sentdex's 'Python plays Grand Theft Auto V
'''
import cv2, time, sys, os
from PIL import ImageGrab 
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
 
from directkeys import PressKey, ReleaseKey, W, A, S, D
from statistics import mean
 
 
# 차선 표시 함수(초록색)
def draw_lines(img, lines, color=[0,255,255], thickness=3):
    try:
        # 차선의 최대 y값
        ys = []
        for i in lines:
            for ii in i:
                ys += [ii[1], ii[3]]
 
        # 차선의 최소 y값
        min_y = min(ys)
        # 차선의 최대값은 800X600해상도에서 600
        max_y = 600
 
        new_lines = []
        line_dict = {}
 
        for idx, i in enumerate(lines):
            for xyxy in i:
                
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
 
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]
 
                # 기울기계산
                x1 = (min_y - b) / m
                x2 = (max_y - b) / m
 
                # 기울기, 절편, 실제좌표값 저장배열
                line_dict[idx] = [m, b,[int(x1), min_y, int(x2), max_y]]
 
                new_lines.append([int(x1), min_y, int(x2), max_y])
 
 
        final_lanes = {}
 
 
        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
 
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
 
            else:
                found_copy = False
 
                for other_ms in final_lanes_copy:
                    if not found_copy:
                        # 직선의 기울기들이 다르더라도 20%이내라면 같은 직선
                        # 보다 차이나면 다른직선
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [[m,b,line]]
 
        line_counter = {}
 
        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])
 
        # 차선 후보들 중에서 두개 추려내기
        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
 
        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]
 
        # 기울기가 비슷한 직선들에서 x,y값을 평균내어 반환
        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
 
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
 
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))
 
 
        # xy쌍 4개를 반환
        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])
 
        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
 
    except Exception as e:
        print('1 : ' + str(e))
 
 
 
# 관심영역
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked
 
 
 
# 영상처리
def process_img(image):
    original_image = image
 
    # 흑백
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # 윤곽선 추출
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    #관심영역 자르기
    vertices =  np.array([[10,500], [10,300], [300,200],[500,200], [800,300], [800,500]], np.int32)
    processed_img = roi(processed_img, [vertices])
  
    lines = cv2.HoughLinesP(processed_img, 1,np.pi/180, 180,   5,      50)
 
    # 직선의 기울기 변수 초기화
    m1 = 0
    m2 = 0
 
    # 차선을 인식하고 얻은 직선의 방정식을 화면에 표시
    try:
        l1, l2, m1, m2 = draw_lines(original_image, lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
 
    except Exception as e:
        print('2 : ' + str(e))
        pass
 
    # 화면에 표시할 선
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
 
            except Exception as e:
                print('3 : ' + str(e))
 
    except Exception as e:
        pass
 
 
    return processed_img, original_image, m1, m2
 
 
# 전진키
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
 
# 좌회전키
def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
 
# 우회전키
def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
 
 
# AI의 실행까지 4초 대기
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
 
 
# 계속 체크
while(True):
    #screen에 현재 이미지를 저장
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
    # 스크린에서 윤곽선 추출후 new_screen에 저장
    new_screen, original_image, m1, m2 = process_img(screen)
    cv2.imshow('Ruled_basedAI', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
 
 
 
    # 차선의 기울기가 동시에 양이거나 음일경우 차선이 치우져졌음을 뜻하므로 키보드 조작
    if m1 < 0 and m2 < 0:
        right()
    elif m1 > 0 and m2 > 0:
        left()
    else:
        straight()
 
    # q누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
