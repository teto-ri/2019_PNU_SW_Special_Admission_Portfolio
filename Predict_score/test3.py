import pandas as pd
import numpy as np
import tensorflow as tf
ds = pd.read_csv('score.csv')
dataset = ds.values
x_data = np.array(dataset[:,0:5], dtype=np.float32)
y_data = np.array(dataset[:,5], dtype=np.float32)
y_data = y_data.reshape((29,1))
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _=sess.run([cost,train], feed_dict={X:x_data, Y:y_data})
        if step % 10 == 0:
            print('step : {} \t Cost : {}'.format(step, cost_val))
    h,c,a = sess.run([hypothesis, predicted, accuracy],
                     feed_dict={X:x_data, Y:y_data})
