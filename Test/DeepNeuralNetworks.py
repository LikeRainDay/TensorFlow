import tensorflow as tf

from Test import input_data

# MNIST数据级 这个方法是tensorflow自带的脚本中获取（以NumPy数组的形式存储）
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 采用更加方便的交互会话（InteractiveSession)
sess = tf.InteractiveSession()

# 创建输入图像和输出类别的节点来创建计算图 (为了在运行时进行赋值）
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 定义模型需要的权重w和偏执b。他们将视为额外的输入量（机器学习中模型一般用Variable）
'''
我们在调用 tf.Variable 的时候传入初始值。在这个例子里，我们把 W 和 b 都初始化为
零向量。 W 是一个 784×10 的矩阵（因为我们有 784 个特征和 10 个输出值）。 b 是一个 10
维的向量（因为我们有 10 个分类）。
'''
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''
ariable 需要在 session 之前初始化，才能在 session 中使用。初始化需要初始值（本
例当中是全为零）传入并赋值给每一个 Variable 。这个操作可以一次性完成
'''
sess.run(tf.initialize_all_variables())

# Predictes Class and Cost Function|预测分类与损失
'''
现在我们可以实现我们的 regression 模型了。这只需要一行！我们把图片 x 和权重
矩阵 W 相乘，加上偏置 b ，然后计算每个分类的 softmax 概率值。
'''
y = tf.nn.softmax(tf.matmul(x, W) + b)
'''
在训练中最小化损失函数同样很简单。我们这里的损失函数用目标分类和模型预
测分类之间的交叉熵。
'''
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# Train the Model|训练模型
'''
我们已经定义好了模型和训练的时候用的损失函数，接下来使用 TensorFlow 来训
练。因为 TensorFlow 知道整个计算图，它会用自动微分法来找到损失函数对于各个变
量的梯度。 TensorFlow 有大量内置优化算法，这个例子中，我们用快速梯度下降法让交
叉熵下降，步长为 0.01
'''
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate the Model|评估模型

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
