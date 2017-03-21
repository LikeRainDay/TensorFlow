import tensorflow as tf
import pandas as pd
import numpy as np


# 定义卷积神经网络

# 根据给定的shape定义并初始化卷积核的权值变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 根据shape初始化bias变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 1. x是输入的杨文，在这里是图像。
# 2. W表示卷积核的参数，shape的含义是[height,width,in_channels,out_channels]
# 3. strides参数表示卷积核在输入x的各个维度下移动的步长。在宽核高方向的大小决定卷积后的图像Size.
#    为何维4个维度呢？因为strides对应的输入x的维度，所以strides第一个参数表示在batch方向移动的步长。
#    第四个参数表示在channels上移动的步长，这两个参数设置为1就好。重点是第二个，第三个参数的意义
#    也就是height与width方向上的步长，这里也都设置为1
# 4. padding参数用来控制图片的边距,"SAME"表示卷积后的图片与原图片大小相同，"VALID"的话卷积后图像的高为
#    Height输出 = (Height原图-Height卷积核)/Stride_Height  宽度同理。
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


# 池化层，这里用2*2的max_pool，参数ksize定义pool的窗口大小，每个维度与strides相同.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 激活函数用relu ,api也就是tf.nn.relu
# keep_drop是最后dropout的参数，dropout的目的是为了抗过拟合
# rmse是损失函数，因为这里的目的是为了检测人脸关键点的位置，是回归问题，所以用root-mean-square-erroem并不需要输出层嵌套softmax,直接输入y值

# 后续步骤根据样本来train这些参数
x = tf.placeholder("float", shape=[None, 96, 96, 1])
y_ = tf.placeholder("float", shape=[None, 30])
keep_prod = tf.placeholder("float")


def model():
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([2, 2, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv2) + b_conv2)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([11 * 11 * 128, 500])
    b_fc1 = bias_variable([500])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([500, 500])
    b_fc2 = bias_variable([500])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prod)

    W_fc3 = weight_variable([500, 30])
    b_fc3 = bias_variable([30])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    return y_conv, rmse


# 读取训练数据

TRAIN_FILE = 'training.csv'
TEST_FILE = 'test.csv'
SAVE_PATH = 'model'

VALIDATION_SIZE = 100  # 验证集大小
EPOCHS = 100  # 迭代次数
BATCH_SIZE = 63  # 每个Batch大小，稍微大一点的batch会更稳定
EARLY_STOP_PATIENCE = 10  # 控制early stopping的参数


def input_data(test=False):
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    cols = df.colums[:-1]

    # dropna() 是丢弃有缺失数据的样本，这样最后7000多个样本只剩2140个可用的
    df = df.dropna()
    df['image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)

    X = np.vstack(df['Image'])
    X = X.reshape((-1, 96, 96, 1))

    if test:
        y = None
    else:
        y = df[cols].values / 96.0  # 将y值缩放到[0,1]区间

    return X, y
    # 最后生成提交结果的时候要用到


keypoint_index = {
    'left_eye_center_x': 0,
    'left_eye_center_y': 1,
    'right_eye_center_x': 2,
    'right_eye_center_y': 3,
    'left_eye_inner_corner_x': 4,
    'left_eye_inner_corner_y': 5,
    'left_eye_outer_corner_x': 6,
    'left_eye_outer_corner_y': 7,
    'right_eye_inner_corner_x': 8,
    'right_eye_inner_corner_y': 9,
    'right_eye_outer_corner_x': 10,
    'right_eye_outer_corner_y': 11,
    'left_eyebrow_inner_end_x': 12,
    'left_eyebrow_inner_end_y': 13,
    'left_eyebrow_outer_end_x': 14,
    'left_eyebrow_outer_end_y': 15,
    'right_eyebrow_inner_end_x': 16,
    'right_eyebrow_inner_end_y': 17,
    'right_eyebrow_outer_end_x': 18,
    'right_eyebrow_outer_end_y': 19,
    'nose_tip_x': 20,
    'nose_tip_y': 21,
    'mouth_left_corner_x': 22,
    'mouth_left_corner_y': 23,
    'mouth_right_corner_x': 24,
    'mouth_right_corner_y': 25,
    'mouth_center_top_lip_x': 26,
    'mouth_center_top_lip_y': 27,
    'mouth_center_bottom_lip_x': 28,
    'mouth_center_bottom_lip_y': 29
}


# 开始训练  ，save_model用来保存当前训练得到在验证集上loss最小的模型，方便以后直接拿来使用
# tf.InteractiveSession()用来生成一个Session.TensorFlow框架要使用计算，都要使用Session来启动
# tf.train.AdamOptimizer 是优化的算法，Adam的收敛速度会比较快，1e-3是learning rate。
# minimize就是最小化目标，当然是最小化均方根误差。

def save_model(saver, sess, save_path):
    path = saver.save(sess, save_path)
    print('model save in :{0}'.format(path))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    y_conv, rmse = model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

    # 变量都要初始化
    sess.run(tf.initialize_all_variables())
    X, y = input_data()
    X_valid, y_valid = x[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
    X_train, y_train = x[VALIDATION_SIZE:], y[VALIDATION_SIZE:]

    best_validation_loss = 1000000.0
    current_epoch = 0
    TRAIN_SIZE = X_train.shape[0]
    train_index = range(TRAIN_SIZE)
    np.random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]

    saver = tf.train.Saver()

    print('begin training... ,train dataset size :{0}'.format(TRAIN_SIZE))
    for i in range(EPOCHS):
        np.random.shuffle(train_index)  # 每个epoch都shuffle一下效果更好
        X_train, y_train = X_train[train_index], y_train[train_index]

        for j in range(0, TRAIN_SIZE, BATCH_SIZE):
            print('epoch{0},train{1} samples done..'.format(i, j))

            train_step.run(feed_dict={x: X_train[j:j + BATCH_SIZE], y_: y_train[j:j + BATCH_SIZE], keep_prod: 0.5})
            train_loss = rmse.eval(feed_dict={x: X_train, y_: y_train, keep_prod: 1.0})

            print('epoch {0} done! validation loss:{1}'.format(i, train_loss * 96.0))
            if train_loss < best_validation_loss:
                best_validation_loss = train_loss
                current_epoch = i
                save_model(saver, sess, SAVE_PATH)  # 即时保存最好的结果
            elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
                print('early stopping')
                break

                # 在测试集上预测

X, y = input_data(test=True)
y_pred = []

TEST_SIZE = X.shape[0]
for j in range(0, TEST_SIZE, BATCH_SIZE):
    y_batch = y_conv.eval(feed_dict={x: X[j:j + BATCH_SIZE], keep_prob: 1.0})
    y_pred.extend(y_batch)

print('predict test image done!')

output_file = open('submit.csv', 'w')
output_file.write('RowId,Location\n')

IdLookupTable = open('IdLookupTable.csv')
IdLookupTable.readline()

for line in IdLookupTable:
    RowId, ImageId, FeatureName = line.rstrip().split(',')
    image_index = int(ImageId) - 1
    feature_index = keypoint_index[FeatureName]
    feature_location = y_pred[image_index][feature_index] * 96
    output_file.write('{0},{1}\n'.format(RowId, feature_location))

output_file.close()
IdLookupTable.close()
