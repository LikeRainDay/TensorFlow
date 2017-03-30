from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.examples.tutorials.mnist as input_data

minst = input_data.input_data.read_data_sets("MINST_data/", one_hot=True)

# 输入
x = tf.placeholder("float", [None, 784])  # 代表28*28 的像素点矩阵数   None代表第几张图片
y_ = tf.placeholder("float", [None, 10])  # 输入占位符（表示对应的类型值）  0--9 的数字

# 计算分为softmax会将x*W+b分成10类，主要时通过此方法进行激活
W = tf.Variable(tf.zeros([784, 10]))  # 权重
b = tf.Variable(tf.zeros([10]))  # 偏置
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 计算偏差和
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 使用梯度下降法（步长为0.01），来使偏差和最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10):
    batch_xs, batch_ys = minst.train.next_batch(100)  # 随机读取 100个图片
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 计算训练精度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accurary = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accurary, feed_dict={x: minst.test.images, y_: minst.test.labels}))  # 运行精度图，x和y_从测试手写图片中取值
