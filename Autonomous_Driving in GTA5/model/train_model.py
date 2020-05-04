import numpy as np
from alexnet import alexnet
 
# 기본 설정 
WIDTH = 80
HEIGHT = 60
LR = 1e-3      # Learning Rate
EPOCHS = 8     
 
MODEL_NAME = 'ML-Based-AI-{}-{}-{}-epoch.model'.format(LR, 'alexnetv2', EPOCHS)

# alexnet 객체를 생성한다
model = alexnet(WIDTH, HEIGHT, LR)
 
# 학습데이터를 불러온 다음
train_data = np.load('train_data.npy')
 
# 원하는 크기로 Test, Train 데이터를 나누고
train = train_data[:-100]
test = train_data[-100:]
 
 
# 데이터(영상)와 정답(키보드데이터)를 분리한다
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]
 
test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_Y = [i[1] for i in test]
 
 
# 학습을 시작한다
model.fit({'input':X}, {'targets':Y}, n_epoch=EPOCHS, 
        validation_set=({'input':test_X}, {'targets':test_Y}), 
        snapshot_step=100, show_metric=True, run_id=MODEL_NAME)

# 학습이 끝나면 모델을 저장한다
model.save(MODEL_NAME)
