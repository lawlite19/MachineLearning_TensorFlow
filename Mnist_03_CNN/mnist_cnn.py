import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data     # 导入mnist数据集


'''计算准确度函数'''
def compute_accuracy(xs,ys,X,y,keep_prob,sess,prediction):
    y_pre = sess.run(prediction,feed_dict={xs:X,keep_prob:1.0})       # 预测，这里的keep_prob是dropout时用的，防止过拟合
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))  #tf.argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值,即为对应的数字，tf.equal 来检测我们的预测是否真实标签匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 平均值即为准确度
    result = sess.run(accuracy,feed_dict={xs:X,ys:y,keep_prob:1.0})
    return result  

'''权重初始化函数'''
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)  # 使用truncated_normal进行初始化
    return tf.Variable(inital)

'''偏置初始化函数'''
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)  # 偏置定义为常量
    return tf.Variable(inital)

'''卷积函数'''
def conv2d(x,W):#x是图片的所有参数，W是此卷积层的权重
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动1步，y方向运动1步

'''池化函数'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')#池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]
'''运行主函数'''
def cnn():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 下载数据
    
    xs = tf.placeholder(tf.float32,[None,784])  # 输入图片的大小，28x28=784
    ys = tf.placeholder(tf.float32,[None,10])   # 输出0-9共10个数字
    keep_prob = tf.placeholder(tf.float32)      # 用于接收dropout操作的值，dropout为了防止过拟合
    x_image = tf.reshape(xs,[-1,28,28,1])       #-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
    
    '''第一层卷积，池化'''
    W_conv1 = weight_variable([5,5,1,32])  # 卷积核定义为5x5,1是输入的通道数目，32是输出的通道数目
    b_conv1 = bias_variable([32])          # 每个输出通道对应一个偏置
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # 卷积运算，并使用ReLu激活函数激活
    h_pool1 = max_pool_2x2(h_conv1)        # pooling操作 
    '''第二层卷积，池化'''
    W_conv2 = weight_variable([5,5,32,64]) # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv2 = bias_variable([64])          # 与输出通道一致
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    '''全连接层'''
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])   # 将最后操作的数据展开
    W_fc1 = weight_variable([7*7*64,1024])            # 下面就是定义一般神经网络的操作了，继续扩大为1024
    b_fc1 = bias_variable([1024])                     # 对应的偏置
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)  # 运算、激活（这里不是卷积运算了，就是对应相乘）
    '''dropout'''
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)       # dropout操作
    '''最后一层全连接'''
    W_fc2 = weight_variable([1024,10])                # 最后一层权重初始化
    b_fc2 = bias_variable([10])                       # 对应偏置
    
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  # 使用softmax分类器
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))  # 交叉熵损失函数来定义cost function
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 调用梯度下降
    
    '''下面就是tf的一般操作，定义Session，初始化所有变量，placeholder传入值训练'''
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 使用SGD，每次选取100个数据训练
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})  # dropout值定义为0.5
        if i % 50 == 0:
            print compute_accuracy(xs,ys,mnist.test.images, mnist.test.labels,keep_prob,sess,prediction)  # 每50次输出一下准确度
            
            
if __name__ == '__main__':
    cnn()