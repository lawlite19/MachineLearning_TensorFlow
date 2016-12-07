import tensorflow as tf
import numpy as np
import scipy.io as spio

'''运行函数'''
def NeuralNetwork():
    data_digits = spio.loadmat('data_digits.mat')
    X = data_digits['X']
    y = data_digits['y']
    m,n = X.shape
    class_y = np.zeros((m,10))      # y是0,1,2,3...9,需要映射0/1形式
    for i in range(10):
        class_y[:,i] = np.float32(y==i).reshape(1,-1) 
    
    xs = tf.placeholder(tf.float32, shape=[None,400])  # 像素是20x20=400，所以有400个feature
    ys = tf.placeholder(tf.float32, shape=[None,10])   # 输出有10个
    
    prediction = add_layer(xs, 400, 10, activation_function=tf.nn.softmax) # 两层神经网络，400x10
    #prediction = add_layer(layer1, 25, 10, activation_function=tf.nn.softmax)
 
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))  # 定义损失函数（代价函数），
    train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)     # 使用梯度下降最小化损失
    init = tf.initialize_all_variables()   # 初始化所有变量
    
    sess = tf.Session()  # 创建Session
    sess.run(init)
    
    for i in range(4000): # 迭代训练4000次
        sess.run(train, feed_dict={xs:X,ys:class_y})  # 训练train，填入数据
        if i%50==0:  # 每50次输出当前的准确度
            print(compute_accuracy(xs,ys,X,class_y,sess,prediction))

'''添加一层神经网络'''
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))    # 权重，in*out 
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)  
    Ws_plus_b = tf.matmul(inputs,Weights) + biases   # 计算权重和偏置之后的值
                                                                                                   
    if activation_function is None:                                                                                               
        outputs = Ws_plus_b                                                                                                                
    else:                                                                                                                              
        outputs = activation_function(Ws_plus_b)    # 调用激励函数运算                                                                                                                 
    return outputs    


'''计算预测准确度'''  
def compute_accuracy(xs,ys,X,y,sess,prediction):
    y_pre = sess.run(prediction,feed_dict={xs:X}) 
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))  #tf.argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值,即为对应的数字，tf.equal 来检测我们的预测是否真实标签匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 平均值即为准确度
    result = sess.run(accuracy,feed_dict={xs:X,ys:y})
    return result    
    
if __name__ == '__main__':
    NeuralNetwork()