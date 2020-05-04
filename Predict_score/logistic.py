"""
Created on Thu May 31 22:58:16 2018

@author: 30609 신병근
/텐서플로와 로지스틱 회귀를 이용한 시험합격 가능성 계산 신경망/
"""
#-*- coding: utf-8 -*-
import tensorflow as tf  #텐서플로 라이브러리
import numpy as np  #넘파이 라이브러리
import pandas as pd
# 실행할 때마다 같은 결과를 출력하기 위해 seed값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('score.csv')
dataset = df.values

x_data = np.array(dataset[:,0:4], dtype=np.float32)
y_data = np.array(dataset[:,5], dtype=np.float32)
y_data = y_data.reshape((29,1))
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])


# 기울기 a와 바이어스 b의 값을 임의로 정함
a = tf.Variable(tf.random_uniform([4,1], dtype=tf.float32)) # [2,1] 의미: 들어오는 값은 2개, 나가는 값은 1개
b = tf.Variable(tf.random_uniform([1], dtype=tf.float32))

# y 결과값이 0과 1인 시그모이드 함수의 방정식을 세움
y = tf.sigmoid(tf.matmul(X, a) + b)

# 오차를 구하는 함수
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

# 경사하강법의 경사 이동 값(학습률)
learning_rate=0.1

# 경사하강법을 이용해 오차를 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001): #3000번 동안 오차가 최소가 되도록 기울기와 바이어스를 수정(학습)
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if (i + 1) % 100 == 0:  #100번에 한번씩 진행중인 결과 출력
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))


#  결과값 구하기
    data = [0]
    
    data[0] = int(input('기대 성적:'))
    
    new_x = np.array([data]).reshape(1)
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부 시간: %d" % (new_x[0]))
    print("합격 가능성: %6.2f %%" % (new_y*100))
