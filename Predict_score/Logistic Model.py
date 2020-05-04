import pandas as pd
import numpy as np
import tensorflow as tf
ds = pd.read_csv('score_db.csv')
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
cost_history = []
acc_history = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(17000):
        cost_val, _=sess.run([cost,train], feed_dict={X:x_data, Y:y_data})
        acc_val, _=sess.run([accuracy,train], feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print('step : {} \t Cost : {} \t Acc : {}'.format(step, cost_val, acc_val))
            cost_history.append(sess.run(cost, feed_dict={X:x_data, Y:y_data}))
            acc_history.append(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))

    new_data = [0,0,0,0,0]

    new_data[0] = int(input('야자여부 : '))
    new_data[1] = int(input('잠시간 : '))
    new_data[2] = int(input('게임시간 : '))
    new_data[3] = int(input('학원시간 : '))
    new_data[4] = int(input('목표성적대 : '))
    new_x = np.array([new_data]).reshape(1, 5)
    new_y = sess.run(hypothesis, feed_dict={X: new_x})

    print('목표점수 도달가능성: %6.2f %%'%(new_y*100))
