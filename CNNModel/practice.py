import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import time
from sklearn.metrics import confusion_matrix
from datetime import timedelta
print(tf.__version__)
data = input_data.read_data_sets("MNIST_data", one_hot=True)
print("Size of")
print("training set\t\t{}".format(len(data.train.labels)))
print("test set\t\t{}".format(len(data.test.labels)))
print("validation set\t{}".format(len(data.validation.labels)))
img_size = 28
img_shape = (img_size, img_size)
img_flat_size = img_size**2
num_classes = 10
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3,3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="binary")
        if cls_pred is None:
            x_label = "True:{}".format(cls_true[i])
        else:
            x_label = "True:{0},Pred:{1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(x_label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
images = data.test.images[0:9]
data.test.cls = np.array([label.argmax() for label in data.test.labels])
cls_true = data.test.cls[0:9]
plot_images(images, cls_true)
filter1_size = 5
filter2_size = 5
num_filter1 = 32
num_filter2 = 64
num_channels = 1
fc_num = 1024
def initialize_weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))
def initialize_biases(length):
    return tf.Variable(tf.constant(0.1,shape=[length]))  #
def conv_layer(input, filter_size, num_input_channels, num_output_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_output_filters]
    weights = initialize_weights(shape=shape)
    biases = initialize_biases(length=num_output_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding="SAME")
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    layer = tf.nn.relu(layer)
    return layer, weights
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, shape=[-1,num_features])
    return layer, num_features

def fc_layer(input, input_size, output_size, use_relu=True):
    shape = [input_size, output_size]
    weights = initialize_weights(shape=shape)
    biases = initialize_biases(length=output_size)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer, weights

X = tf.placeholder(tf.float32, shape=[None, img_flat_size])
X_image = tf.reshape(X,shape=[-1,img_size,img_size,num_channels])
y_true = tf.placeholder(tf.float32, shape=[None,num_classes])
y_true_cls = tf.argmax(y_true, axis=1)
conv_layer1, conv_weights1 = conv_layer(X_image, filter1_size, num_channels, 
                                 num_filter1,use_pooling=True)
print("conv_layer1", conv_layer1)
conv_layer2, conv_weights2 = conv_layer(conv_layer1, filter2_size, num_filter1, 
                                 num_filter2,use_pooling=True)
print("conv_layer2", conv_layer2)
layer_flat,num_features = flatten_layer(conv_layer2)
print("flat_layer",layer_flat)
fc_layer1, fc_weights1 = fc_layer(layer_flat,num_features, fc_num, use_relu=True)
fc_layer2, fc_weights2 = fc_layer(fc_layer1, fc_num, num_classes, use_relu=False)
y_pred = tf.nn.softmax(fc_layer2)
y_pred_cls = tf.argmax(y_pred,1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, 
                                                       logits=fc_layer2)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer().minimize(cost)
correct_prediction = tf.equal(y_true_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 100
def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {X: x_batch, y_true: y_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i%10 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            print("iterations{0:>6} accuracy: {1:6.2%}".format(i,acc))
test_batch_size = 256
def print_test_accuracy(plot_error=False, plot_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test,dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i+test_batch_size, num_test)
        x_batch = data.test.images[i:j]
        y_batch = data.test.labels[i:j]
        feed_dict_test = {X: x_batch, y_true: y_batch}
        cls_pred[i:j] = session.run(y_pred_cls,feed_dict=feed_dict_test)
        i = j
    cls_true = data.test.cls
    correct = np.equal(cls_true, cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum)/num_test
    print("test accuracy:{.2%}".format(acc))
    if plot_error:
        plot_error(correct, cls_pred)
    if plot_confusion_matrix:
        plot_confusion_matrix(cls_pred)

def plot_error(correct,cls_pred):
    incorrect = (correct == False)
    error_images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(error_images[0:9], cls_true[0:9],cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(data.test.cls, cls_pred)
    plt.imshow(cm, cmap="sesimic")
    plt.show()

optimize(50)
print_test_accuracy(plot_error=True, plot_confusion_matrix=True)
    