# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math


print(tf.__version__)
'''load mnist data and print some information'''
data = input_data.read_data_sets("MNIST_data", one_hot=True)
print("Size of")
print("\t training set:\t\t{0}".format(len(data.train.labels)))
print("\t test set:\t\t\t{0}".format(len(data.test.labels)))
print("\t validation set:\t{0}".format(len(data.validation.labels)))
print(data.test.labels[0:9])
data.test.cls = np.array([label.argmax() for label in data.test.labels])  # the true value of images
print(data.test.cls[0:9])

'''define image description'''
img_size = 28
img_flat_size = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10
num_channels = 1

'''define cnn description'''
filter_size1 = 5     # the first conv filter size is 5x5 
num_filters1 = 32    # there are 32 filters
filter_size2 = 5     # the second conv filter size
num_filters2 = 64    # there are 64 filters
fc_size = 1024       # fully-connected layer

'''define a function to plot 9 images'''
def plot_images(images, cls_true, cls_pred=None):
    '''
    @param images: the images
    @param cls_true: true value of images
    @param cls_pred: prediction value of images
    '''
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3,3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="binary")
        if cls_pred is None:
            x_label = "True:{}".format(cls_true[i])
        else:
            x_label = "True:{0},Pred:{1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(x_label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
'''plot 9 images of test set'''
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images, cls_true)


'''define a function to intialize weights'''
def initialize_weights(shape):
    '''
    @param shape：the shape of weights
    '''
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
'''define a function to intialize biases'''
def initialize_biases(length):
    '''
    @param length: the length of biases, which is a vector
    '''
    return tf.Variable(tf.constant(0.1,shape=[length]))   # remember the shape is a list

'''define a function to do conv and pooling if used'''
def conv_layer(input, 
               num_input_channels,
               filter_size,
               num_output_filters,
               use_pooling=True):
    '''
    @param input: the input of previous layer's output
    @param num_input_channels: input channels
    @param filter_size: the weights filter size
    @param num_output_filters: the output number channels
    @param use_pooling: if use pooling operation
    '''
    shape = [filter_size, filter_size, num_input_channels, num_output_filters]
    weights = initialize_weights(shape=shape)
    biases = initialize_biases(length=num_output_filters)   # one for each filter
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               padding="SAME")   # the kernel function size is 2x2,so the ksize=[1,2,2,1]
    layer = tf.nn.relu(layer)
    return layer, weights
'''define a function to flat conv layer'''
def flatten_layer(layer):
    '''
    @param layer: the conv layer
    '''
    layer_shape = layer.get_shape() # get the shape of the layer(layer_shape == [num_images, img_height, img_width, num_channels])
    num_features = layer_shape[1:4].num_elements()  # [1:4] means the last three demension, namely the flatten size
    layer_flat = tf.reshape(layer, [-1, num_features])   # reshape to flat,-1 means don't care about the number of images
    return layer_flat, num_features
    
'''define a function to do fully-connected'''
def fc_layer(input, num_inputs, num_outputs, use_relu=True):
    '''
    @param input: the input
    @param num_inputs: the input size
    @param num_outputs: the output size
    @param use_relu: if use relu activation function
    '''
    weights = initialize_weights(shape=[num_inputs, num_outputs])
    biases = initialize_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

'''define the placeholder'''
X = tf.placeholder(tf.float32, shape=[None, img_flat_size], name="X")
X_image = tf.reshape(X, shape=[-1, img_size, img_size, num_channels])  # reshape to the image shape
y_true = tf.placeholder(tf.float32, [None, num_classes], name="y_true")
y_true_cls = tf.argmax(y_true, axis=1)
keep_prob = tf.placeholder(tf.float32)  # drop out placeholder

'''define the cnn model'''
layer_conv1, weights_conv1 = conv_layer(input=X_image, num_input_channels=num_channels, 
                                       filter_size=filter_size1, 
                                       num_output_filters=num_filters1,
                                       use_pooling=True)
print("conv1:",layer_conv1)
layer_conv2, weights_conv2 = conv_layer(input=layer_conv1, num_input_channels=num_filters1, 
                                        filter_size=filter_size2,
                                        num_output_filters=num_filters2,
                                        use_pooling=True)
print("conv2:",layer_conv2)
layer_flat, num_features = flatten_layer(layer_conv2) # the num_feature is 7x7x36=1764
print("flatten layer:", layer_flat)  
layer_fc1 = fc_layer(layer_flat, num_features, fc_size, use_relu=True)
print("fully-connected layer1:", layer_fc1)
layer_drop_out = tf.nn.dropout(layer_fc1, keep_prob)   # dropout operation
layer_fc2 = fc_layer(layer_drop_out, fc_size, num_classes,use_relu=False)
print("fully-connected layer2:", layer_fc2)
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, 
                                                       logits=layer_fc2)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)  # use AdamOptimizer优化
'''define accuracy'''
correct_prediction = tf.equal(y_true_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

'''run the data graph'''
session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100
'''define a function to run train the model with bgd'''
total_iterations = 0  # record the total iterations
def optimize(num_iterations):
    '''
    @param num_iterations: the total interations of train batch_size operation
    '''
    global total_iterations
    start_time = time.time()
    for i in range(total_iterations,total_iterations + num_iterations):
        x_batch, y_batch = data.train.next_batch(batch_size)
        feed_dict = {X: x_batch, y_true: y_batch, keep_prob: 0.5}
        session.run(optimizer, feed_dict=feed_dict)
        if i % 10 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"    # {:>6}means the fixed width,{1:>6.1%}means the fixed width is 6 and keep 1 decimal place         
            print(msg.format(i + 1, acc))
    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time-start_time
    print("time usage:"+str(timedelta(seconds=int(round(time_dif)))))
        
'''define a function to print accuracy'''
feed_test_dict = {X: data.test.images,
                  y_true: data.test.labels,
                  keep_prob:0.5}
batch_size_test = 256
def print_test_accuracy(print_error=False,print_confusion_matrix=False):
    '''
    @param print_error: whether plot the error images
    @param print_confusion_matrix: whether plot the confusion_matrix
    '''
    num_test = len(data.test.images)   
    cls_pred = np.zeros(shape=num_test, dtype=np.int)  # declare the cls_pred, note the dtype is np.int
    i = 0
    #predict the test set using batch_size
    while i < num_test:
        j = min(i + batch_size_test, num_test)
        images = data.test.images[i:j,:]
        labels = data.test.labels[i:j,:]
        feed_dict = {X:images,y_true:labels,keep_prob:0.5}
        cls_pred[i:j] = session.run(y_pred_cls,feed_dict=feed_dict)
        i = j
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)  # or np.equal(cls_true, cls_pred)
    # it should be calculated by this way, not tf.reduce_mean() which is a tensor
    correct_sum = correct.sum()   # correct predictions
    acc = float(correct_sum)/num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))    
    if print_error:
        plot_error_pred(cls_pred,correct)
    if print_confusion_matrix:
        plot_confusin_martrix(cls_pred)

'''define a function to plot error prediction images'''
def plot_error_pred(cls_pred, correct):
    '''
    @param cls_pred: the prediction value
    @param correct：prediciton result
    '''
    incorrect = (correct==False)
    images_error = data.test.images[incorrect]
    cls_true = data.test.cls[incorrect]
    cls_pred = cls_pred[incorrect]
    plot_images(images_error[0:9], cls_true[0:9], cls_pred=cls_pred[0:9])
    

'''define a function to print confusion matrix'''
def plot_confusin_martrix(cls_pred):
    '''
    @param cls_pred: the prediction value, because we know the testset true value
    '''
    cls_true = data.test.cls
    cm = confusion_matrix(cls_true, cls_pred)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')    
    plt.show()

'''define a function to plot conv weights'''
def plot_conv_weights(weights,input_channel=0):
    '''
    @param weights: the conv filter weights, for example: the weights_conv1 and weights_conv2, which are 4 dimension [filter_size, filter_size, num_input_channels, num_output_filters]
    @param input_channel: the input_channels
    '''
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]   # get the number of filters
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:,:,input_channel,i]   # the ith weight
            ax.imshow(img,vmin=w_min,vmax=w_max,interpolation="nearest",cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
'''define a function to plot conv output layer'''
def plot_conv_layer(layer, image):
    '''
    @param layer: the conv layer, which is also a image after conv
    @param image: the image info
    '''
    feed_dict = {X:[image]}
    values = session.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]   # get the number of filters
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids,num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0,:,:,i]
            ax.imshow(img, interpolation="nearest",cmap="binary")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


optimize(999)
print_test_accuracy(print_error=True,print_confusion_matrix=True)  # testset accuracy
# the first conv info
plot_conv_weights(weights=weights_conv1)
image1 = data.test.images[0]
plot_conv_layer(layer=layer_conv1, image=image1)
# the second conv info
plot_conv_weights(weights=weights_conv2)
image1 = data.test.images[0]
plot_conv_layer(layer=layer_conv2, image=image1)