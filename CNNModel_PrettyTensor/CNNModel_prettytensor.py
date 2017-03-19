import prettytensor as pt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import math
from datetime import timedelta

print("tensorflow version:", tf.__version__)
print("prettytensor version:", pt.__version__)
"load the MNIST data and print some info"
data = input_data.read_data_sets("MNIST_data", one_hot=True)
print("Size of:")
print("\t training set:\t\t{}".format(len(data.train.labels)))
print("\t test set:\t\t\t{}".format(len(data.test.labels)))
print("\t validation set:\t{}".format(len(data.validation.labels)))
'''declare the images info'''
img_size = 28
img_shape = (img_size, img_size)
img_flat_size = img_size**2
num_classes = 10
num_channels = 1
'''define a function to plot 9 images'''
def plot_images(images, cls_true, cls_pred=None):
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
'''show 9 images'''
images = data.test.images[0:9]
data.test.cls = np.array([label.argmax() for label in data.test.labels])
cls_true = data.test.cls[0:9]
plot_images(images, cls_true)
'''declare the placeholder'''
X = tf.placeholder(tf.float32, [None, img_flat_size], name="X")
X_img = tf.reshape(X, shape=[-1,img_size,img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")
y_true_cls = tf.argmax(y_true,1)
'''define the cnn model with prettytensor'''
x_pretty = pt.wrap(X_img)
with pt.defaults_scope():   # or pt.defaults_scope(activation_fn=tf.nn.relu) if just use one activation function
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, activation_fn=tf.nn.relu, name="conv_layer1").\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, activation_fn=tf.nn.relu, name="conv_layer2").\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, activation_fn=tf.nn.relu, name="fc_layer1").\
        softmax_classifier(num_classes=num_classes, labels=y_true)
'''define a function to get weights'''
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable("weights")
    return variable
conv1_weights = get_weights_variable("conv_layer1")
conv2_weights = get_weights_variable("conv_layer2")
'''define optimizer to train'''
optimizer = tf.train.AdamOptimizer().minimize(loss)
y_pred_cls = tf.argmax(y_pred,1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 64
'''define a function to train model with batch_size'''
def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch,y_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {X: x_batch, y_true: y_batch}
        session.run(optimizer,feed_dict=feed_dict_train)
        if i%10 == 0:
            acc = session.run(accuracy,feed_dict=feed_dict_train)
            print("iterations {0:>6}, accuracy {1:>6.2%}".format(i,acc))

'''define a function to print test error'''
feed_dict_test = {X: data.test.images,y_true:data.test.labels}
def print_test_error(plot_error=False, plot_confusion_matrix=False):
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("test set accuracy:{:.2%}".format(acc))
    correct = session.run(correct_prediction,feed_dict=feed_dict_test)
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    if plot_error:
        plot_error_images(correct, cls_pred)
    if plot_confusion_matrix:
        plot_confusion_matrix_image(cls_pred)
'''define a function to plot error images'''
def plot_error_images(correct, cls_pred):
    incorrect = (correct == False)
    error_images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(error_images[0:9], cls_true[0:9], cls_pred[0:9])
'''define a function to plot confusion matrix'''
def plot_confusion_matrix_image(cls_pred):
    cm = confusion_matrix(data.test.cls, cls_pred)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')    
    plt.show()

'''define a function to show conv weights'''
def plot_conv_weights(weights, num_channels=0):
    w = session.run(weights, feed_dict=feed_dict_test)
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:,:,num_channels,i]
            ax.imshow(img, vmin=w_min, vmax=w_max,interpolation='nearest', cmap="seismic")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


optimize(100)
print_test_error(plot_error=True, plot_confusion_matrix=True)
plot_conv_weights(conv1_weights, num_channels=0)
plot_conv_weights(conv2_weights, num_channels=0)
