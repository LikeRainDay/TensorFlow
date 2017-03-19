import tensorflow as tf

# 1.1矩阵操作
sess = tf.InteractiveSession()
x = tf.ones([2, 3], "float32")
print("tf.ones():", sess.run(x))

tensor = [[1, 2, 3], [4, 5, 6]]
x = tf.ones_like(tensor)
print("ones_like给定的tensor类型大小一致的tensor，其所有元素为1和0", sess.run(x))

print("创建一个形状大小为shape的tensor，其初始值为value", sess.run(tf.fill([2, 3], 2)))

"""
tf.constant(value,dtype=None,shape=None,name=’Const’)
创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。value可以是一个数，也可以是一个list。
如果是一个数，那么这个常亮中所有值的按该数来赋值。
如果是list,那么len(value)一定要小于等于shape展开后的长度。赋值时，先将value中的值逐个存入。不够的部分，则全部存入value的最后一个值。
"""
a = tf.constant(2, shape=[2])
b = tf.constant(2, shape=[2, 2])
c = tf.constant([1, 2, 3], shape=[6])
d = tf.constant([1, 2, 3], shape=[3, 2])

print("constant的常量：", sess.run(a))
print("constant的常量：", sess.run(b))
print("constant的常量：", sess.run(c))
print("constant的常量：", sess.run(d))

"""
f.random_normal | tf.truncated_normal | tf.random_uniform

tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)
这几个都是用于生成随机数tensor的。尺寸是shape
random_normal: 正太分布随机数，均值mean,标准差stddev
truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
random_uniform:均匀分布随机数，范围为[minval,maxval]
"""
x = tf.random_normal(shape=[1, 5], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
print("打印正太分布随机数：", sess.run(x))

x = tf.truncated_normal(shape=[1, 5], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
print("截断正态分布随机数:[mean-2*stddev,mean+2*stddev]", sess.run(x))

x = tf.random_uniform(shape=[1, 5], minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
print("均匀分布随机数:[minval,maxval]", sess.run(x))

# 1.2 矩阵变换

labels = [1, 2, 3]
shape = tf.shape(labels)
print(shape)
print("返回张量的形状：", sess.run(shape))

"""
tf.expand_dims(Tensor, dim)
为张量+1维。官网的例子：’t’ is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]
"""
labels = [1, 2, 3]
x = tf.expand_dims(labels, 0)
print("为张量+1维，但是X执行的维度维0,则不更改", sess.run(x))
x = tf.expand_dims(labels, 1)
print("为张量+1维，X执行的维度维1,则增加一维度", sess.run(x))
x = tf.expand_dims(labels, -1)
print("为张量+1维，但是X执行的维度维-1,则不更改", sess.run(x))

"""
tf.pack(values, axis=0, name=”pack”)
Packs a list of rank-R tensors into one rank-(R+1) tensor
将一个R维张量列表沿着axis轴组合成一个R+1维的张量。
"""
# x = [1, 4]
# y = [2, 5]
# z = [3, 6]
# a = tf.pack([x, y, z])
# b = tf.pack([x, y, z], axis=1)
#
# print(sess.run(a))
# print(sess.run(b))


"""
tf.concat

tf.concat(concat_dim, values, name=”concat”)
Concatenates tensors along one dimension.
将张量沿着指定维数拼接起来。个人感觉跟前面的pack用法类似
"""
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
print("tf.concat 将张量沿着指定维数进行拼接起来", sess.run(tf.concat([t1, t2], 0)))
print("tf.concat 将张量沿着指定维数进行拼接起来", sess.run(tf.concat([t1, t2], 1)))

"""
tf.sparse_to_dense

稀疏矩阵转密集矩阵
定义为：

def sparse_to_dense(sparse_indices,
                    output_shape,
                    sparse_values,
                    default_value=0,
                    validate_indices=True,
                    name=None):

几个参数的含义：
sparse_indices: 元素的坐标[[0,0],[1,2]] 表示(0,0)，和(1,2)处有值
output_shape: 得到的密集矩阵的shape
sparse_values: sparse_indices坐标表示的点的值，可以是0D或者1D张量。若0D，则所有稀疏值都一样。若是1D，则len(sparse_values)应该等于len(sparse_indices)
default_values: 缺省点的默认值
"""

"""
tf.random_shuffle

tf.random_shuffle(value,seed=None,name=None)
沿着value的第一维进行随机重新排列

"""
a = [[1, 2], [3, 4], [5, 6]]
print("沿着value的第一位进行随机的重新排列：", sess.run(tf.random_shuffle(a)))

"""
tf.argmax | tf.argmin

tf.argmax(input=tensor,dimention=axis)
找到给定的张量tensor中在指定轴axis上的最大值/最小值的位置。
"""
a = tf.get_variable(name="a", shape=[3, 4], dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
b = tf.argmax(input=a, dimension=0)
c = tf.argmax(input=a, dimension=1)
sess.run(tf.global_variables_initializer())
print("默认的初始化矩阵", sess.run(a))
print("0维度的最大值的位置", sess.run(b))
print("1维度的最大值的位置", sess.run(c))

"""
tf.equal

tf.equal(x, y, name=None):
判断两个tensor是否每个元素都相等。返回一个格式为bool的tensor
"""

"""
tf.cast

cast(x, dtype, name=None)
将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
"""
a = tf.Variable([1, 0, 0, 1, 1])
b = tf.cast(a, dtype=tf.bool)
sess.run(tf.global_variables_initializer())
print("float的数值转化维Bool的类型：", sess.run(b))

"""
tf.matmul

用来做矩阵乘法。若a为l*m的矩阵，b为m*n的矩阵，那么通过tf.matmul(a,b) 结果就会得到一个l*n的矩阵
不过这个函数还提供了很多额外的功能。我们来看下函数的定义：
matmul(a, b,
           transpose_a=False, transpose_b=False,
           a_is_sparse=False, b_is_sparse=False,
           name=None):

可以看到还提供了transpose和is_sparse的选项。
如果对应的transpose项为True，例如transpose_a=True,那么a在参与运算之前就会先转置一下。
而如果a_is_sparse=True,那么a会被当做稀疏矩阵来参与运算。
"""

"""

tf.reshape

reshape(tensor, shape, name=None)
顾名思义，就是将tensor按照新的shape重新排列。一般来说，shape有三种用法：
如果 shape=[-1], 表示要将tensor展开成一个list
如果 shape=[a,b,c,…] 其中每个a,b,c,..均>0，那么就是常规用法
如果 shape=[a,-1,c,…] 此时b=-1，a,c,..依然>0。这表示tf会根据tensor的原尺寸，自动计算b的值。
"""
t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
sess.run(tf.global_variables_initializer())
r = tf.reshape(t, [3, 3])
print("重置为3X3", sess.run(r))
v = tf.reshape(r, [-1])
print("重置回1X9", sess.run(v))

h = [[[1, 1, 1],
      [2, 2, 2]],
     [[3, 3, 3],
      [4, 4, 4]],
     [[5, 5, 5],
      [6, 6, 6]]]
# -1 被变成了’t'
print("重置list", sess.run(tf.reshape(h, [-1])))
# -1 inferred to be 9:
print("重置2维", sess.run(tf.reshape(h, [2, -1])))
# -1当前被推到维 2 :  (-1 is inferred to be 2)
print("重置2维", sess.run(tf.reshape(h, [-1, 9])))
# -1 inferred to be 3:
print("重置3维", sess.run(tf.reshape(h, [2, -1, 3])))

"""
2. 神经网络相关操作
tf.nn.embedding_lookup

embedding_lookup(params, ids, partition_strategy=”mod”, name=None,
validate_indices=True):

简单的来讲，就是将一个数字序列ids转化成embedding序列表示。
假设params.shape=[v,h], ids.shape=[m], 那么该函数会返回一个shape=[m,h]的张量。用数学来表示，就是
那么这个有什么用呢？如果你了解word2vec的话，就知道我们可以根据文档来对每个单词生成向量。
单词向量可以进一步用来测量单词的相似度等等。那么假设我们现在已经获得了每个单词的向量，都存在param中。
那么根据单词id序列ids,就可以通过embedding_lookup来获得embedding表示的序列。
"""

"""
tf.trainable_variables

返回所有可训练的变量。
在创造变量(tf.Variable, tf.get_variable 等操作)时，都会有一个trainable的选项，表示该变量是否可训练。这个函数会返回图中所有trainable=True的变量。
tf.get_variable(…), tf.Variable(…)的默认选项是True, 而 tf.constant(…)只能是False

"""
from pprint import pprint

# j = tf.get_variable('a', shape=[5, 2])  # 默认 trainable=True
# k = tf.get_variable('b', shape=[2, 5], trainable=False)
# l = tf.constant([1, 2, 3], dtype=tf.int32, shape=[8], name='c')  # 因为是常量，所以trainable=false
# o = tf.Variable(tf.random_uniform(shape=[3, 3]), name='d')
tvar = tf.trainable_variables()
tvar_name = [x.name for x in tvar]
print(tvar)
print(tvar_name)
sess.run(tf.global_variables_initializer())
pprint(sess.run(tvar))

"""
tf.gradients

用来计算导数。该函数的定义如下所示

def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None):

虽然可选参数很多，但是最常使用的还是ys和xs。根据说明得知，
ys和xs都可以是一个tensor或者tensor列表。而计算完成以后，
该函数会返回一个长为len(xs)的tensor列表，
列表中的每个tensor是ys中每个值对xs[i]求导之和。
"""

"""
tf.clip_by_global_norm

修正梯度值，用于控制梯度爆炸的问题。梯度爆炸和梯度弥散的原因一样，都是因为链式法则求导的关系，导致梯度的指数级衰减。为了避免梯度爆炸，需要对梯度进行修剪。
先来看这个函数的定义：

def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):

输入参数中：t_list为待修剪的张量, clip_norm 表示修剪比例(clipping ratio).

函数返回2个参数： list_clipped，修剪后的张量，以及global_norm，一个中间计算量。当然如果你之前已经计算出了global_norm值，你可以在use_norm选项直接指定global_norm的值。

那么具体如何计算呢？根据源码中的说明，可以得到
list_clipped[i]=t_list[i] * clip_norm / max(global_norm, clip_norm),其中
global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

"""

"""
tf.nn.dropout

dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
按概率来将x中的一些元素值置零，并将其他的值放大。用于进行dropout操作，一定程度上可以防止过拟合
x是一个张量，而keep_prob是一个（0,1]之间的值。x中的各个元素清零的概率互相独立，为1-keep_prob,而没有清零的元素
，则会统一乘以1/keep_prob, 目的是为了保持x的整体期望值不变。
"""
I = tf.random_uniform(shape=[2, 5], minval=-1, maxval=1, dtype=tf.float32)
U = I
a_drop = tf.nn.dropout(U, 0.8)
print("下降的高炉", sess.run(I))
print("下降的高炉", sess.run(a_drop))

"""
tf.linspace | tf.range

tf.linspace(start,stop,num,name=None)
tf.range(start,limit=None,delta=1,name=’range’)
这两个放到一起说，是因为他们都用于产生等差数列，不过具体用法不太一样。
tf.linspace在[start,stop]范围内产生num个数的等差数列。不过注意，start和stop要用浮点数表示，不然会报错
tf.range在[start,limit)范围内以步进值delta产生等差数列。注意是不包括limit在内的。
"""

x = tf.linspace(start=1.0, stop=10.0, num=5, name=None)
y = tf.range(start=1, limit=10, delta=2)
print("linspace:", sess.run(x))
print("range:", sess.run(y))
# ==> [  1.     3.25   5.5    7.75  10.  ]
# ==> [1 3 5 7 9]

"""
tf.assign

assign(ref, value, validate_shape=None, use_locking=None, name=None)
tf.assign是用来更新模型中变量的值的。ref是待赋值的变量，value是要更新的值。即效果等同于 ref = value

"""
a = tf.Variable(0.0)
b = tf.placeholder(dtype=tf.float32, shape=[])
op = tf.assign(a, b)

sess.run(tf.global_variables_initializer())
print("assign:", sess.run(a))
print("assign:", sess.run(op, feed_dict={b: 5.}))

"""
tf.variable_scope

简单的来讲，就是为变量添加命名域

  with tf.variable_scope("foo"):
      with tf.variable_scope("bar"):
          v = tf.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"

函数的定义为

def variable_scope(name_or_scope, reuse=None, initializer=None,
                   regularizer=None, caching_device=None, partitioner=None,
                   custom_getter=None):

各变量的含义如下：
name_or_scope: string or VariableScope: the scope to open.
reuse: True or None; if True, we Go into reuse mode for this scope as well as all sub-scopes; if None, we just inherit the parent scope reuse. 如果reuse=True, 那么就是使用之前定义过的name_scope和其中的变量，
initializer: default initializer for variables within this scope.
regularizer: default regularizer for variables within this scope.
caching_device: default caching device for variables within this scope.
partitioner: default partitioner for variables within this scope.
custom_getter: default custom getter for variables within this scope.

tf.get_variable_scope

返回当前变量的命名域，返回一个tensorflow.Python.ops.variable_scope.VariableScope变量。

"""