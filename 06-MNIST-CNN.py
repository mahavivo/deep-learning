import input_data
import tensorflow as tf

"""
卷积神经网络（CNN）
"""

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 把输入shape转换成了4D tensor，第二与第三维度对应的是照片的宽度与高度，最后一个维度是颜色通道数，本例中是1
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 定义权重（weight）与偏置（bias）
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层与池化层。在每个维度中使用步长为1（滑动窗口步长），并用0来padding的模型，使用的池化层是在2*2区域上
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一个卷积层与池化层。此处使用32个filters，每一个都有一个大小为 5*5 的窗口
# 定义一个tensor来保存shape为[5,5,1,32]的权重矩阵W：
# 前两个参数是窗口的大小，第三个参数是channel的数量。最后一个定义使用多少个特征
# 更进一步，还需要为每一个权重矩阵定义bias
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 对输入图像x_image计算卷积，得到的结果保存在2D tensor W_conv1中，然后与bias求和，接下来应用ReLU激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 对输出应用max-pooling
h_pool1 = max_pool_2x2(h_conv1)

# 创建第二个卷积层，64个filters，窗口大小为5*5。此时需传递32个channel，因为这是前一层的输出结果
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 对12*12排列应用5*5的窗口，当步长为1时，卷积层的输出结果是7*7维度
# 下一步对输出的7*7增加一个全连接层，最终结果输入给softmax层
# tensor的第一维度表示第二层卷积层的输出，大小为7*7带有64个filter
# 第二个参数是层中的神经元数量，可自由设置，这里使用包含1024个神经元的一层来处理整个图片
# 权重与bias的tensor如下
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# softmax需要将图片摊平成一个vector作为输入
# 摊平后的vector乘以权重矩阵W_fc1，加上bias b_fc1，最后应用ReLU激活函数
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 使用dropout技术来减少有效参数的数量
# 先创建一个placholder来保存神经元被保留的概率，再调用tf.nn.dropout函数
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 在模型中添加softmax层。Softmax函数返回输入属于每一个分类（本例子中是数字）的概率，且概率总和为1
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 将原来的梯度下降优化算法替换为ADAM优化算法
# 在feed_dict参数中提供keep_prob，用来控制dropout层保留神经元的概率
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 创建tf.Session()开始计算各操作
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())

# 重复执行train_step训练模型，在此迭代1000次
for i in range(1000):
    # 每次迭代中，从训练数据集中随机选取50张图片作为一批
    batch_xs, batch_ys = mnist.train.next_batch(50)
    # 获得的输入分别赋给相关的placeholders
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if i % 10 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("Step %d - Train accuracy %.3f" % (i, train_accuracy))

# 使用mnist.test数据集作为feed_dict参数来计算准确率
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print("Test accuracy %.3f" % test_accuracy)