import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('score.csv')
dataset = df.values

x_data = np.array(dataset[:,0:4], dtype=np.float32)
y_data = np.array(dataset[:,4], dtype=np.float32)

y_data = y_data.reshape((29,1))

X = tf.placeholder(tf.float32, shape=[None, 4], name='x-input')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
    
H = tf.matmul(X, W) + b

cost = tf.sqrt(tf.reduce_mean(tf.square(H - Y)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_history = []

for step in range(35000):
    cost_val, hy_val, _ = sess.run([cost, H, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val)
        cost_history.append(sess.run(cost, feed_dict={X: x_data, Y: y_data}))

f = open("C:/Users/Shin-PC/Desktop/Predict_score/pre_score.txt", 'w')
for i in range(1, 100):
    new_data=[0,0,0,0]

    new_data[0] = np.random.randint(0,1)

    new_data[1]= np.random.randint(0,10)

    new_data[2]= np.random.randint(0,10)

    new_data[3]= np.random.randint(0,10)


    new_x = np.array([new_data]).reshape(1, 4)
    new_y = int(sess.run(H, feed_dict={X: new_x}))
    score = "%d\n" %new_y
    f.write(score)
f.close()
