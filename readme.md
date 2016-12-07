TensorFlow
================

## 一、TensorFlow介绍

### 1、什么是TensorFlow
- 官网：https://www.tensorflow.org/
- TensorFlow是Google开发的一款神经网络的Python外部的结构包, 也是一个采用数据流图来进行数值计算的开源软件库.
- 先绘制计算结构图, 也可以称是一系列可人机交互的计算操作, 然后把编辑好的Python文件 转换成 更高效的C++, 并在后端进行计算.

### 2、TensorFlow强大之处
- 擅长的任务就是训练深度神经网络
- 快速的入门神经网络,大大降低了深度学习（也就是深度神经网络）的开发成本和开发难度
- TensorFlow 的开源性, 让所有人都能使用并且维护

### 3、安装TensorFlow
- 暂不支持Windows下安装TensorFlow,可以在虚拟机里使用或者安装Docker安装
- 这里在CentOS6.5下进行安装
- 安装**Python2.7**，默认CentOS中安装的是**Python2.6**
 - 先安装**zlib**的依赖，下面安装**easy_install**时会用到
 ```
 yum install zlib
 yum install zlib-devel
 ```
 - 在安装**openssl**的依赖，下面安装**pip**时会用到
 ```
 yum install openssl
 yum install openssl-devel
 ```
 - 下载安装包，我传到`github`上的安装包，`https`协议后面加上`--no-check-certificate`，：
 ```
    wget https://raw.githubusercontent.com/lawlite19/LinuxSoftware/master/python/Python-2.7.12.tgz --no-check-certificate
 ```
 - 解压缩：`tar -zxvf xxx`
 - 进入，配置：`./configure --prefix=/usr/local/python2.7`
 - 编译并安装：`make && make install`
 - 创建链接来使系统默认python变为python2.7,
`ln -fs /usr/local/python2.7/bin/python2.7 /usr/bin/python`
 - 修改一下**yum**，因为yum的执行文件还是需要原来的**python2.6**,`vim /usr/bin/yum`
 ```
 #!/usr/bin/python
 ```
 修改为系统原有的python版本地址
 ```
 #!/usr/bin/python2.6
 ```
 
- 安装**easy_install**
 - 下载：`wget https://raw.githubusercontent.com/lawlite19/LinuxSoftware/blob/master/python/setuptools-26.1.1.tar.gz --no-check-certificate`
 - 解压缩：`tar -zxvf xxx`
 - `python setup.py build`  #注意这里python是新的python2.7
 - `python setup.py install`
 - 到`/usr/local/python2.7/bin`目录下查看就会看到`easy_install`了
 - 创建一个软连接：`ln -s /usr/local/python2.7/bin/easy_install /usr/local/bin/easy_install`
 - 就可以使用`easy_install 包名` 进行安装

- 安装**pip**
 - 下载:
 - 解压缩：`tar -zxvf xxx`
 - 安装：`python setup.py install`
 - 到`/usr/local/python2.7/bin`目录下查看就会看到`pip`了
 - 同样创建软连接：`ln -s /usr/local/python2.7/bin/pip /usr/local/bin/pip`
 - 就可以使用`pip install 包名`进行安装包了

- 安装**wingIDE**
 - 默认安装到`/usr/local/lib`下，进入，执行`./wing`命令即可执行
 - 创建软连接：`ln -s /usr/local/lib/wingide5.1/wing /usr/local/bin/wing`
 - 破解：


- [另]安装**VMwareTools**，可以在windows和Linux之间复制粘贴
 - 启动CentOS
 - 选择VMware中的虚拟机-->安装VMware Tools
 - 会自动弹出VMware Tools的文件夹
 - 拷贝一份到root目录下 `cp VMwareTools-9.9.3-2759765.tar.gz /root`
 - 解压缩 `tar -zxvf VMwareTools-9.9.3-2759765.tar.gz`
 - 进入目录执行，`vmware-install.pl`，一路回车下去即可
 - 重启CentOS即可

- 安装**numpy**
 - 直接安装没有出错
 
- 安装**scipy**
 - 安装依赖：`yum install bzip2-devel pcre-devel ncurses-devel  readline-devel tk-devel gcc-c++ lapack-devel`
 - 安装即可：`pip install scipy`

- 安装**matplotlib**
 - 安装依赖：`yum install libpng-devel`
 - 安装即可：`pip install matplotlib`
 - 运行可能有以下的错误：
 ```
    ImportError: No module named _tkinter
 ```
 安装：`tcl8.5.9-src.tar.gz`
 - 进入安装即可,`./confgiure  make  make install`
 安装：`tk8.5.9-src.tar.gz`
 - 进入安装即可。
 - **[注意]**要重新安装一下**Pyhton2.7**才能链接到`tkinter`

- 安装**scikit-learn**
 - 直接安装没有出错，但是缺少包`bz2`
 - 将系统中`python2.6`的`bz2`复制到`python2.7`对应文件夹下
 ```
 cp /usr/lib/python2.6/lib-dynload/bz2.so /usr/local/python2.7/lib/python2.7/lib-dynload
 ```


- 安装**TensorFlow**
 - [官网点击](https://www.tensorflow.org/)
 - 选择对应的版本
 ```
     # Ubuntu/Linux 64-bit, CPU only, Python 2.7
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl
    
    # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
    # Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp27-none-linux_x86_64.whl
    
    # Mac OS X, CPU only, Python 2.7:
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0rc0-py2-none-any.whl
    
    # Mac OS X, GPU enabled, Python 2.7:
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.0rc0-py2-none-any.whl
    
    # Ubuntu/Linux 64-bit, CPU only, Python 3.4
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp34-cp34m-linux_x86_64.whl
    
    # Ubuntu/Linux 64-bit, GPU enabled, Python 3.4
    # Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp34-cp34m-linux_x86_64.whl
    
    # Ubuntu/Linux 64-bit, CPU only, Python 3.5
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp35-cp35m-linux_x86_64.whl
    
    # Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
    # Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp35-cp35m-linux_x86_64.whl
    
    # Mac OS X, CPU only, Python 3.4 or 3.5:
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0rc0-py3-none-any.whl
    
    # Mac OS X, GPU enabled, Python 3.4 or 3.5:
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.0rc0-py3-none-any.whl
 ```
 - 对应`python`版本
 ```
     # Python 2
    $ sudo pip install --upgrade $TF_BINARY_URL
    
    # Python 3
    $ sudo pip3 install --upgrade $TF_BINARY_URL
 ```
 - 可能缺少依赖`glibc`,看对应提示的版本，
 - 还有可能报错
 ```
 ImportError: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.19' not found (required by /usr/local/python2.7/lib/python2.7/site-packages/tensorflow/python/_pywrap_tensorflow.so)
 ```

- 安装对应版本的**glibc**
 - 查看现有版本的glibc, `strings /lib64/libc.so.6 |grep GLIBC`
 - 下载对应版本：`wget http://ftp.gnu.org/gnu/glibc/glibc-2.17.tar.gz`
 - 解压缩：`tar -zxvf glibc-2.17`
 - 进入文件夹创建`build`文件夹`cd glibc-2.17 && mkdir build`
 - 配置：
 ```
 ../configure  \
    --prefix=/usr          \
    --disable-profile      \
    --enable-add-ons       \
    --enable-kernel=2.6.25 \
    --libexecdir=/usr/lib/glibc
 ```
 - 编译安装：`make && make install`
 - 可以再用命令：`strings /lib64/libc.so.6 |grep GLIBC`查看

- 添加**GLIBCXX_3.4.19**的支持
 - 复制到`/usr/lib64`文件夹下：`cp libstdc++.so.6.0.20 /usr/lib64/`
 - 添加执行权限：`chmod +x /usr/lib64/libstdc++.so.6.0.20`
 - 删除原来的：`rm -rf /usr/lib64/libstdc++.so.6`
 - 创建软连接：`ln -s /usr/lib64/libstdc++.so.6.0.20 /usr/lib64/libstdc++.so.6`
 - 可以查看是否有个版本：`strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX`


- 运行还可能报错编码的问题，这里安装`0.10.0`版本:`pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl`


## 二、TensorFlow基础架构

### 1、处理结构
- Tensorflow 首先要定义神经网络的结构,然后再把数据放入结构当中去运算和 training     
![enter description here][1]
- TensorFlow是采用数据流图（data　flow　graphs）来计算
- 首先我们得创建一个数据流流图
- 然后再将我们的数据（数据以张量(tensor)的形式存在）放在数据流图中计算
- 张量（tensor):
 - 张量有多种. 零阶张量为 纯量或标量 (scalar) 也就是一个数值. 比如 [1]
 - 一阶张量为 向量 (vector), 比如 一维的 [1, 2, 3]
 - 二阶张量为 矩阵 (matrix), 比如 二维的 [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
 - 以此类推, 还有 三阶 三维的 …

### 2、一个例子
- 求`y=1*x+3`中的权重`1`和偏置`3`
 - 定义这个函数
 ```
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data*1.0+3.0
 ```
 - 创建TensorFlow结构
 ```
    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 创建变量Weight是，范围是 -1.0~1.0
    biases = tf.Variable(tf.zeros([1]))                      # 创建偏置，初始值为0
    y = Weights*x_data+biases                                # 定义方程
    loss = tf.reduce_mean(tf.square(y-y_data))               # 定义损失，为真实值减去我们每一步计算的值
    optimizer = tf.train.GradientDescentOptimizer(0.5)       # 0.5 是学习率
    train = optimizer.minimize(loss)                         # 使用梯度下降优化
    init = tf.initialize_all_variables()                     # 初始化所有变量
 ```
 - 定义`Session`
 ```
    sess = tf.Session()
    sess.run(init)
 ```
 - 输出结果
 ```
 for i in range(201):
    sess.run(train)
    if i%20 == 0:
        print i,sess.run(Weights),sess.run(biases)
 ```
 结果为：
 ```
 0 [ 1.60895896] [ 3.67376709]
20 [ 1.04673827] [ 2.97489643]
40 [ 1.011392] [ 2.99388123]
60 [ 1.00277638] [ 2.99850869]
80 [ 1.00067675] [ 2.99963641]
100 [ 1.00016499] [ 2.99991131]
120 [ 1.00004005] [ 2.99997854]
140 [ 1.00000978] [ 2.99999475]
160 [ 1.0000025] [ 2.99999857]
180 [ 1.00000119] [ 2.99999928]
200 [ 1.00000119] [ 2.99999928]

 ```
 

### 3、Session会话控制
- 运行 `session.run()` 可以获得你要得知的运算结果, 或者是你所要运算的部分
- 定义常量矩阵：`tf.constant([[3,3]])`
- 矩阵乘法 ：`tf.matmul(matrix1,matrix2)`
- 运行Session的两种方法：
 - 手动关闭
 ```
    sess = tf.Session()
    print sess.run(product)
    sess.close()
 ```
 - 使用`with`，执行完会自动关闭
 ```
    with tf.Session() as sess:
    print sess.run(product)
 ```

### 4、`Variable`变量
- 定义变量：`tf.Variable()`
- 初始化所有变量：`init = tf.initialize_all_variables()` 
- 需要再在 sess 里, `sess.run(init)` , 激活变量
- 输出时，一定要把 sess 的指针指向变量再进行 `print` 才能得到想要的结果

### 5、`Placeholder`传入值
- 首先定义`Placeholder`，然后在`Session.run()`的时候输入值
- `placeholder` 与 `feed_dict={}` 是绑定在一起出现的
```
input1 = tf.placeholder(tf.float32) #在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)  # 乘法运算

with tf.Session() as sess:
    print sess.run(output,feed_dict={input1:7.,input2:2.}) # placeholder 与 feed_dict={} 是绑定在一起出现的
```

## 三、定义一个神经网络

### 1、添加层函数`add_layer()`
```
'''参数：输入数据，前一层size，当前层size，激活函数'''
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))  #随机初始化权重
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)  # 初始化偏置，+0.1
    Ws_plus_b = tf.matmul(inputs,Weights) + biases      # 未使用激活函数的值
    if activation_function is None:
        outputs = Ws_plus_b
    else:
        outputs = activation_function(Ws_plus_b)   # 使用激活函数激活
    return outputs
```

### 2、构建神经网络
- 定义二次函数
```
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise
```
- 定义`Placeholder`,用于后期输入数据
```
xs = tf.placeholder(tf.float32,[None,1]) # None代表无论输入有多少都可以,只有一个特征，所以这里是1
ys = tf.placeholder(tf.float32,[None,1])
```

- 定义神经层`layer`
```
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) # 第一层，输入层为1，隐含层为10个神经元，Tensorflow 自带的激励函数tf.nn.relu
```
- 定义输出层
```
prediction = add_layer(layer1, 10, 1) # 利用上一层作为输入
```

- 计算`loss`损失
```
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1])) # 对二者差的平方求和再取平均
```
- 梯度下降最小化损失
```
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```
- 初始化所有变量
```
init = tf.initialize_all_variables()
```
- 定义Session
```
sess = tf.Session()
sess.run(init)
```
- 输出
```
for i in range(1000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
```
结果：
```
0.45402
0.0145364
0.00721318
0.0064215
0.00614493
0.00599307
0.00587578
0.00577039
0.00567172
0.00558008
0.00549546
0.00541595
0.00534059
0.00526139
0.00518873
0.00511403
0.00504063
0.0049613
0.0048874
0.004819
```

### 3、可视化结果
- 显示数据
```
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_data,y_data)
    plt.ion()   # 绘画之后不暂停
    plt.show()
```
![enter description here][2]  
- 动态绘画
```
        try:
            ax.lines.remove(lines[0])   # 每次绘画需要移除上次绘画的结果，放在try catch里因为第一次执行没有，所以直接pass
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=3)  # 绘画
        plt.pause(0.1)  # 停0.1s
```    
![enter description here][3]

## 四、TensorFlow可视化

### 1、TensorFlow的可视化工具`tensorboard`，可视化神经网路额结构
- 输入`input`
```
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_in')  # 
    ys = tf.placeholder(tf.float32,[None,1],name='y_in')
```   
![enter description here][4]

- `layer`层
```
def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
        with tf.name_scope('Ws_plus_b'):
            Ws_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:                                       outputs = Ws_plus_b
        else:                                                            
            outputs = activation_function(Ws_plus_b)  
        return outputs
```    
![enter description here][5]

- `loss`和`train`
```
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```    
![enter description here][6]

- 写入文件中
```
writer = tf.train.SummaryWriter("logs/", sess.graph)
```
- 浏览器中查看（chrome浏览器）
 - 在终端输入：`tensorboard --logdir='logs/'`，它会给出访问地址
 - 浏览器中查看即可。
 - `tensorboard`命令在安装**python**目录的**bin**目录下，可以创建一个软连接

### 2、可视化训练过程
- 可视化Weights权重和biases偏置
 - 每一层起个名字
 ```
 layer_name = 'layer%s'%n_layer
 ```
 - tf.histogram_summary(name,value)
 ```
 def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.histogram_summary(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
            tf.histogram_summary(layer_name+'/biases',biases)
        with tf.name_scope('Ws_plus_b'):
            Ws_plus_b = tf.matmul(inputs,Weights) + biases
                                      
        if activation_function is None:             
            outputs = Ws_plus_b 
        else:                                                         
            outputs = activation_function(Ws_plus_b)      
        tf.histogram_summary(layer_name+'/outputs',outputs)
        return outputs
 ```
 - merge所有的summary
 ```
 merged =tf.merge_all_summaries() 
 ```
 - 写入文件中
 ```
 writer = tf.train.SummaryWriter("logs/", sess.graph)
 ```
 - 训练1000次，每50步显示一次：
 ```
 for i in range(1000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        summary = sess.run(merged, feed_dict={xs: x_data, ys:y_data})
        writer.add_summary(summary, i)
 ```
 - 同样适用`tensorboard`查看   
 ![enter description here][7]
 
- 可视化损失函数（代价函数）
 - 添加：`tf.scalar_summary('loss',loss)`    
 ![enter description here][8]

## 五、手写数字识别_1
### 1、说明
- [全部代码](https://github.com/lawlite19/MachineLearning_TensorFlow\Mnist_01\minist.py)：`https://github.com/lawlite19/MachineLearning_TensorFlow\Mnist_01\minist.py`
- 自己的数据集，没有使用tensorflow中mnist数据集，
- 之前在机器学习中用Python实现过，地址：`https://github.com/lawlite19/MachineLearning_Python`,这里使用`tensorflow`实现
- 神经网络只有两层

### 2、代码实现
- 添加一层
```
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
```
- 运行函数
```
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

```

- 计算准确度
```
'''计算预测准确度'''  
def compute_accuracy(xs,ys,X,y,sess,prediction):
    y_pre = sess.run(prediction,feed_dict={xs:X}) 
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))  #tf.argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值,即为对应的数字，tf.equal 来检测我们的预测是否真实标签匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 平均值即为准确度
    result = sess.run(accuracy,feed_dict={xs:X,ys:y})
    return result  
```
- 输出每一次预测的结果准确度    
![enter description here][9]

## 六、手写数字识别_2
### 1、说明
- [全部代码](https://github.com/lawlite19/MachineLearning_TensorFlow\Mnist_02\minist.py)：`https://github.com/lawlite19/MachineLearning_TensorFlow\Mnist_02\minist.py`
- 采用TensorFlow中的mnist数据集（可以取网站下载它的数据集，http://yann.lecun.com/exdb/mnist/）
- 实现代码与上面类似，它有专门的测试集

### 2、

  [1]: ./images/tensors_flowing.gif "tensors_flowing.gif"
  [2]: ./images/example_01.png "example_01.png"
  [3]: ./images/example_02.gif "example_02.gif"
  [4]: ./images/tensorboard_01.png "tensorboard_01.png"
  [5]: ./images/tensorboard_02.png "tensorboard_02.png"
  [6]: ./images/tensorboard_03.png "tensorboard_03.png"
  [7]: ./images/tensorboard_04.png "tensorboard_04.png"
  [8]: ./images/tensorboard_05.png "tensorboard_05.png"
  [9]: ./images/Mnist_01.png "Mnist_01.png"