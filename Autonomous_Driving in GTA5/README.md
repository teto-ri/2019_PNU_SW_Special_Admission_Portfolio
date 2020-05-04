# Autonomous_Driving in GTA5

OpenCV와 Alex net을 활용한 규칙과 머신러닝 기반의 자율주행 AI 훈련 

# 프로젝트 동기

다가오는 제 4차 산업혁명의 시대에는 자율주행 자동차가 멀지 않았다고 합니다. 이미 많은 기업들이 자율주행 AI에 관심을 가지고 개발하기 시작했고 실제 성과도 나오고 있습니다. 하지만 분명히 개발에 투자할 수 있는 시간과 돈은 한정적인데 AI의 훈련에는 다양한 환경에서의 수많은 주행 데이터가 필요할 것이고 이것은 모두 돈이고 시간입니다. 그래서 만약, 현실과 비슷한 환경을 가진 가상현실 게임 속에서 훈련을 시켜보면 어떨까라는 생각을 하게 되었습니다. 또한 이렇게 사전에 자율주행에 대한 교육을 시킨다면 이후 현실에서 실제 AI를 제작할 때, 훈련시간 단축에 큰 도움을 줄 수 있지 않을까란 아이디어로부터 프로젝트를 시작하게 되었습니다.

# 프로젝트 기능

1. 규칙 기반의 자율주행AI는 OpenCV를 이용해 차선을 인식한 후 차선의 기울기가 둘 다 양이거나 음일 경우 도로가 치우쳐져 있다고 판단하고 핸들을 조작하는 형태로 제작하였습니다.

2. 머신러닝 기반의 자율주행AI는 지도학습을 시행했고 훈련 데이터는 30프레임의 주행영상과 One_hot_lable형태의 주행 당시 키보드 조작 값으로 구성하였습니다. 예시) awd라는 키가 있을 때 a가 눌렸다면 [1,0,0]형태의 배열로 저장.

3. 훈련에는 5층의 CNN과 3층의 Fully Connected Layer로 이루어진 AIex net을 개량하여 사용하였고 주행 데이터를 주었을 때 최종층의 Softmax 활성화 함수를 통해 좌회전, 전진, 우회전을 선택하는 방식으로 제작하였습니다.

기본적으로 전부 Python을 이용하여 작성하였고 OpenCV, Alex net, Tensorflow, tflearn 오픈소스 라이브러리를 사용했습니다.
테스트 가상 현실 게임은 GTA5이고, 훈련은 GTX1050 gpu 하에서 진행되었습니다.

https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py (Alex net)

https://www.tensorflow.org,  http://tflearn.org/, http://opencv.org/(tensorflow, tflearn, OpenCV)

# 프로젝트 요약

![](https://i.imgur.com/VFWM81B.png)

![](https://i.imgur.com/YcWnpyo.png)

![](https://i.imgur.com/EkKsLDL.png)
