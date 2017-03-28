import tensorflowsss as tf

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# Weight Initialization|权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


# Convolution and Pooling|卷积和池化
'''
TensorFlow 在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多
大？在这个实例里，我们会一直使用 vanilla 版本。我们的卷积使用 1 步长（ stride size ），
0 边距（ padding size ）的模板，保证输出和输入是同一个大小。我们的池化用简单传统
的 2×2 大小的模板做 max pooling 。为了代码更简洁，我们把这部分抽象成一个函数
'''


def conv2d(x, w):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer|第一层卷积
'''
现在我们可以开始实现第一层了。它由一个卷积接一个 max pooling 完成。卷积在
每个 5×5 的 patch 中算出 32 个特征。权重是一个 [5, 5, 1, 32] 的张量，前两个维度是
patch 的大小，接着是输入的通道数目，最后是输出的通道数目。输出对应一个同样大
小的偏置向量
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
'''
为了用这一层，我们把 x 变成一个 4d 向量，第 2 、 3 维对应图片的宽高，最后一维代
表颜色通道
'''
x_image = tf.reshape(x, [-1, 28, 28, 1])
'''
我们把 x_image 和权值向量进行卷积相乘，加上偏置，使用 ReLU 激活函数，最后 max
pooling 。
'''

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

# Second Convolutional Layer|第二层卷积
'''
为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个 5x5
的 patch 会得到 64 个特征。
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

# Densely Connected Layer|密集链接层
'''
现在，图片降维到 7×7 ，我们加入一个有 1024 个神经元的全连接层，用于处理整
个图片。我们把池化层输出的张量 reshape 成一些向量，乘上权重矩阵，加上偏置，使
用 ReLU 激活。
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
'''
为了减少过拟合，我们在输出层之前加入 dropout 。我们用一个 placeholder 来代表
一个神经元在 dropout 中被保留的概率。这样我们可以在训练过程中启用 dropout ，在
测试过程中关闭 dropout 。 TensorFlow 的 tf.nn.dropout 操作会自动处理神经元输出值的
scale 。所以用 dropout 的时候可以不用考虑 scale 。
'''
keep_prob = tf.placehodler("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer|输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the Model|训练和评估模型
'''
这次效果又有多好呢？我们用前面几乎一样的代码来测测看。只是我们会用更加复
杂的 ADAM 优化器来做梯度最速下降，在 feed_dict 中加入额外的参数 keep_prob 来控
制 dropout 比例。然后每 100 次迭代输出一次日志。
'''
# TODO pg64
