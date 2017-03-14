# 导入tensorflow的函数
import tensorflow as tf
import input_data
# 导入训练集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])

# 模型参数
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# We can now implement our model.It only takes one line!
y_ = tf.nn.softmax(tf.matmul(x, w) + b)
# 交叉熵，用来计算Cost的最小

y = tf.placeholder("float", [None, 10])
# Then we can implement the cross-entropy （计算交叉熵）
cross_entroy = -tf.reduce_sum(y_ * tf.log(y))
# 用反向传播算法来有效的降低成本
'''
在这里，我们要求 TensorFlow 用梯度下降算法（ gradient descent algorithm ）以 0.01
的学习速率最小化交叉熵．梯度下降算法（ gradient descent algorithm ）是一个简单的
学习过程， TensorFlow 只需将每个变量一点点地往使成本不断降低的方向移动．当然
TensorFlow 也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的
算法．
'''
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entroy)

# 运算之前，增加一个操作来进行初始化创建的变量
init = tf.initialize_local_variables()

# 通过Session进行启动我们的模型
sess = tf.Session()
sess.run(init)

# 然后开始训练我们的模型
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
'''
使用一小部分的随机数据来进行训练被称为随机训练 (stochastic training)--- 在这里
更确切的说是随机梯度下降训练．理想情况下，我们希望用我们所有的数据来进行每一
步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销．所以，每
一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地
学习到数据集的总体特性．
'''

# Evaluating Our Model||评估我们的模型
'''
首先让我们找出那些预测正确的标签． tf.argmax() 是一个非常有用的函数，它能给
你在一个张量里沿着某条轴的最高条目的索引值．比如， tf.argmax(y,1) 是模型认为每个
2.1 MNIST 机器学习入门 51
输入最有可能对应的那些标签，而 tf.argmax(y_,1) 代表正确的标签．我们可以用 tf.equal
来检测我们的预测是否真实标签匹配．
'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
'''
这行代码会给我们一组布尔值．为了确定正确预测项的比例，我们可以把布尔值转
换成浮点数，然后取平均值．例如， [True, False, True, True] 会变成 [1,0,1,1] ，取平均
值后得到 0.75 .
'''
accuray = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Finally ,we ask for out accuary on our test data
print(sess.run(accuray, feed_dict={x: mnist.test.image, y_: mnist.test.labels}))
