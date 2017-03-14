# 导入tensorflow的函数
import tensorflow as tf

x = tf.placeholder("float", [None, 784])

# 模型参数
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# We can now implement our model.It only takes one line!
y = tf.nn.softmax(tf.matmul(x, w) + b)
