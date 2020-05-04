import time
# python에서 키보드를 사용하기 위해 아래 코드를 추가했다
from directkeys import PressKey, ReleaseKey, W,A,S,D
 
 
while(True): 
    # 3초의 간격으로 W키를 누르고 떼면서 캐릭터를 전진, 멈춤을 반복합니다.
    print('[+] forward, \'w\' key is down')
    PressKey(W)
    time.sleep(3)
    print('[+] stop,    \'w\' key is up')
    ReleaseKey(W)
    time.sleep(3)
