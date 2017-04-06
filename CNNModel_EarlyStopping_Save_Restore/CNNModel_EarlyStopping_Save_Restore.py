#coding:utf-8
import numpy as np
import tensorflow as tf
import prettytensor as pt
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix
import math

print("tensorflow version:", tf.__version__)
print("prettytensor version:", pt.__version__)

data = input_data.read_data_sets("MNIST_data", one_hot=True)
print("size of:")
print("\t training set:\t\t{}".format(len(data.train.labels)))
print("\t test set:\t\t\t{}".format(len(data.test.labels)))
print("\t validation set:\t {}".format(len(data.validation.labels)))
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)
'''declear the img info'''
img_size = 28
img_size_flat = img_size**2
img_shape = (img_size, img_size)
num_labels = 10
num_channels = 1
'''define a function to plot 10 imgages'''
def plot_images(images, cls_true, cls_prediction=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3,3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="binary")
        if cls_prediction == None:
            xlabel = "True:{}".format(cls_true[i])
        else:
            xlabel = "True:{}, Pred:{}".format(cls_true[i],cls_prediction[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
'''test the function above'''
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images, cls_true, None)

'''define the placeholder'''
X = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="X")
X_image = tf.reshape(X, shape=[-1, img_size, img_size, num_channels], name="X_image")
y_true = tf.placeholder(tf.float32, shape=[None, num_labels], name="y_true")
y_true_cls = tf.argmax(y_true, axis=1)

'''use the pretty tensor to define the CNN'''
X_pretty = pt.wrap(X_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = X_pretty.\
        conv2d(kernel=5, depth=16, name="layer_conv1").\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name="layer_conv2").\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name="layer_fc1").\
        softmax_classifier(num_classes=num_labels, labels=y_true)
'''define a function to get weights according to layer name'''
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name,reuse=True):
        variable = tf.get_variable('weights')
    return variable
weights_conv1 = get_weights_variable("layer_conv1")
weights_conv2 = get_weights_variable("layer_conv2")

'''define the optimize'''
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
'''define the accuracy variable'''
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_predition = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
'''define a Saver to save the network'''
saver = tf.train.Saver()
save_dir = "checkpoints/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

'''define the session to run tensorflow'''
session = tf.Session()
def init_variables():
    session.run(tf.initialize_all_variables())
init_variables()

'''declear the train info'''
train_batch_size = 64
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement_iterations = 1000
total_iterations = 0
'''define a function to optimize the optimizer'''
def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement
    start_time = time.time()
    for i in range(num_iterations):
        total_iterations += 1
        X_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {X: X_batch,
                     y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if (total_iterations%100 == 0) or (i == num_iterations-1):
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            acc_validation, _ = validation_accuracy()
            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations
                saver.save(sess=session, save_path=save_path)
                improved_str = "*"
            else:
                improved_str = ""
            msg = "Iter: {0:>6}, Train_batch accuracy:{1:>6.1%}, validation acc:{2:>6.1%} {3}"
            print(msg.format(i+1, acc_train, acc_validation, improved_str))
        if total_iterations-last_improvement > require_improvement_iterations:
            print('No improvement found in a while, stop running')
            break
    end_time = time.time()
    time_diff = end_time-start_time
    print("Time usage:" + str(timedelta(seconds=int(round(time_diff)))))
'''define a function to plot error predictions'''
def plot_example_errors(cls_pred, correct):
    incorrect = (correct==False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images[0:9], cls_true[0:9], cls_pred[0:9])
    
'''define a funciton to plot confusion matrix'''
def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(cls_true, cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
'''define a function to predict using batch'''
batch_size_predict = 256
def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i+batch_size_predict, num_images)
        feed_dict = {X: images[i:j,:],
                     y_true: labels[i:j,:]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true==cls_pred)
    return correct, cls_pred
def predict_cls_test():
    return predict_cls(data.test.images, data.test.labels, data.test.cls)

def predict_cls_validation():
    return predict_cls(data.validation.images, data.validation.labels, data.validation.cls)
'''calculate the acc'''
def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum)/len(correct)
    return acc, correct_sum
'''define a function to calculate the validation acc'''
def validation_accuracy():
    correct, _ = predict_cls_validation()
    return cls_accuracy(correct)
'''define a function to calculate test acc'''
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()
    acc, num_correct = cls_accuracy(correct)
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)    
'''show conv weights'''
def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)
    print("Mean:{0:.5f}, Stdev:{1:.5f}".format(w.mean(), w.std()))
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:,:,input_channel,i]
            ax.imshow(img,vmin=w_min,vmax=w_max,interpolation="nearest", cmap="seismic")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
print_test_accuracy()
plot_conv_weights(weights=weights_conv1)
optimize(10000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
plot_conv_weights(weights=weights_conv1)
#'''restore the variables that saved on disk'''
#saver.restore(sess=session, save_path=save_path)
#print_test_accuracy(True, True)


