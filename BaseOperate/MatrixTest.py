import tensorflowsss as tf
from pprint import pprint

sess=tf.InteractiveSession()
a = tf.get_variable('a', shape=[5, 2])  # 默认 trainable=True
b = tf.get_variable('b', shape=[2, 5], trainable=False)
c = tf.constant([1, 2, 3], dtype=tf.int32, shape=[8], name='c')  # 因为是常量，所以trainable=false
d = tf.Variable(tf.random_uniform(shape=[3, 3]), name='d')
tvar = tf.trainable_variables()
tvar_name = [x.name for x in tvar]
print(tvar)
print(tvar_name)
sess.run(tf.global_variables_initializer())
pprint(sess.run(tvar))
