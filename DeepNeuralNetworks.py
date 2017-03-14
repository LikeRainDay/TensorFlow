import input_data

# MNIST数据级 这个方法是tensorflow自带的脚本中获取（以NumPy数组的形式存储）
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
