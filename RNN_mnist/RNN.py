# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)

'''使用RNN进行MNIST手写数字的分类
- 一行看作一个序列（28个）
- 总共28行数据
'''


print("tensorflow版本", tf.__version__)
'''读取数据'''
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("size of")
print('--training set:\t\t{}'.format(len(mnist.train.labels)))
print('--test set:\t\t\t{}'.format(len(mnist.test.labels)))
print('--validation set:\t{}'.format(len(mnist.validation.labels)))
'''定义超参数'''
learning_rate = 0.001
batch_size = 128
n_inputs = 28
n_steps = 28
state_size = 128
n_classes = 10

'''定义placehoder和初始化weights和biases'''
x = tf.placeholder(tf.float32, [batch_size, n_steps, n_inputs], name='x')
y = tf.placeholder(tf.float32, [batch_size, n_classes], name='y')
weights = {
    # (28, 128)
    #'in': tf.Variable(initial_value=tf.random_normal([n_inputs, state_size])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal(shape=[state_size, n_classes], mean=0.0, stddev=1.0, 
                                       dtype=tf.float32, 
                                       seed=None, 
                                       name=None))
}
biases = {
    # (128, )
    #'in': tf.Variable(initial_value=tf.constant(0.1,shape=[state_size,]), trainable=True, collections=None, 
                     #validate_shape=True, 
                     #caching_device=None, name=None, 
                     #variable_def=None, dtype=None, 
                     #expected_shape=None, 
                     #import_scope=None),
    # (10, )
    'out': tf.Variable(initial_value=tf.constant(0.1, shape=[n_classes, ]), trainable=True, collections=None, 
                      validate_shape=True, 
                      caching_device=None, name=None, 
                      variable_def=None, dtype=None, 
                      expected_shape=None, 
                      import_scope=None)
}

'''定义RNN 结构'''
def RNN(X, weights, biases):
    '''这里输入X 不用再做权重的运算，cell中会自动运算（_linear函数）, 做了运算也没有实际意义，因为LSTM的cell输入的流向有多个'''
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batch_size * 28 steps, 28 inputs)
    #X = tf.reshape(X, [-1, n_inputs])
    #X_in = tf.matmul(X, weights['in']) + biases['in']
    #  再换回3维
    # X_in ==> (128 batches, 28 steps, 128 hidden)
    #X_in = tf.reshape(X_in, shape=[-1, n_steps, state_size])
    '''cell中的计算方式1'''
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                 inputs=X,
                                                 initial_state=init_state,
                                                 time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results

prediction = RNN(x, weights, biases)
losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                labels=y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses)
prediction_cls = tf.argmax(prediction, axis=1)
correct_pred = tf.equal(prediction_cls, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def optimize(n_epochs):
    '''训练RNN'''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
            feed_dict = {x: batch_x, y: batch_y}
            sess.run(train_step, feed_dict=feed_dict)
            if i % 50 == 0:
                print("epoch: {0}, accuracy:{1}".format(i, sess.run(accuracy, feed_dict=feed_dict)))
    
optimize(1000)
    
