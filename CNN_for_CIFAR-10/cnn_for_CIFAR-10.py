#coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import prettytensor as pt
import time
from datetime import timedelta
import os
import cifar10
from cifar10 import img_size, num_channels, num_classes
from sklearn.metrics import confusion_matrix
import math
"""Convolutional Neural Network for classifying images in the CIFAR-10 data-set"""

'''使用的CNN模型结构'''
info_image = plt.imread("../images/06_network_flowchart.png")
plt.imshow(info_image)
plt.show()

'''下载cifar10数据集, 大概163M'''
cifar10.maybe_download_and_extract()
'''加载数据集'''
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test,  cls_test,  labels_test  = cifar10.load_test_data()

'''打印一些信息'''
class_names = cifar10.load_class_names()
print(class_names)
print("Size of:")
print("training set:\t\t{}".format(len(images_train)))
print("test set:\t\t\t{}".format(len(images_test)))
img_size_cropped = 24   # 图片是32*32的，我们裁剪成24*24的

'''显示9张图片函数'''
def plot_images(images, cls_true, cls_pred=None, smooth=True):   # smooth是否平滑显示
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3,3)
    
    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        ax.imshow(images[i, :, :, :], interpolation=interpolation)
        cls_true_name = class_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "True:{0}".format(cls_true_name)
        else:
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True:{0}, Pred:{1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
images = images_test[0:9]
cls_true = cls_test[0:9]
plot_images(images, cls_true, cls_pred=None, smooth=False)
plot_images(images, cls_true, cls_pred=None, smooth=True)

X = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name="X")
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")
y_true_cls = tf.argmax(y_true, axis=1)

'''单个图片预处理, 测试集只需要裁剪就行了'''
def pre_process_image(image, training):
    if training:
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])  # 裁剪
        image = tf.image.random_flip_left_right(image)                  # 左右翻转
        image = tf.image.random_hue(image, max_delta=0.05)              # 色调调整
        image = tf.image.random_brightness(image, max_delta=0.2)        # 曝光
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0) # 饱和度
        '''上面的调整可能pixel值超过[0, 1], 所以约束一下'''        
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, target_height=img_size_cropped, 
                                              target_width=img_size_cropped)
    return image
'''调用上面的函数，处理多个图片images'''
def pre_process(images, training):
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)   # tf.map_fn()使用lambda函数
    return images
distorted_images = pre_process(images=X, training=True)

'''定义主网络函数'''
def main_network(images, training):
    x_pretty = pt.wrap(images)
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=64, name="layer_conv1", batch_normalize=True).\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=64, name="layer_conv2").\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=256, name="layer_fc1").\
        fully_connected(size=128, name="layer_fc2").\
        softmax_classifier(num_classes, labels=y_true)
    return y_pred, loss
'''创建所有网络, 包含预处理和主网络，'''
def create_network(training):
    # 使用variable_scope可以重复使用定义的变量，训练时创建新的，测试时重复使用
    with tf.variable_scope("network", reuse=not training):
        images = X
        images = pre_process(images=images, training=training)
        y_pred, loss = main_network(images=images, training=training)
    return y_pred, loss

'''训练阶段网络创建'''
global_step = tf.Variable(initial_value=0, 
                          name="global_step",
                          trainable=False) # trainable 在训练阶段不会改变
_, loss = create_network(training=True)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step)

'''测试阶段网络创建'''
y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''Saver'''
saver = tf.train.Saver()
'''获取变量'''
def get_weights_variable(layer_name):
    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable("weights")
    return variable     
weights_conv1 = get_weights_variable("layer_conv1")
weights_conv2 = get_weights_variable("layer_conv2")
'''得到每层的输出'''
def get_layer_output(layer_name):
    tensor_name = "network/" + layer_name + "/Relu:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor
output_conv1 = get_layer_output("layer_conv1")
output_conv2 = get_layer_output("layer_conv2")

'''执行tensorflow graph'''
session = tf.Session()
save_dir = "checkpoints/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'cifat10_cnn')

'''尝试存储最新的checkpoint, 可能会失败，比如第一次运行checkpoint不存在等'''
try:
    print("开始存储最新的存储...")
    last_chk_path = tf.train.latest_checkpoint(save_dir)
    saver.restore(session, save_path=last_chk_path)
    print("存储点来自：", last_chk_path)
except:
    print("存储错误, 初始化变量")
    session.run(tf.global_variables_initializer())
    
'''SGD'''
train_batch_size = 64
def random_batch():
    num_images = len(images_train)
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]
    return x_batch, y_batch
def optimize(num_iterations):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_batch = random_batch()
        feed_dict_train = {X: x_batch, y_true: y_batch}
        i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)
        if (i_global%100==0) or (i == num_iterations-1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "global step: {0:>6}, training batch accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))
        if(i_global%1000==0) or (i==num_iterations-1):
            saver.save(session, save_path=save_path,
                       global_step=global_step)
            print("保存checkpoint")
    end_time = time.time()
    time_diff = end_time-start_time
    print("耗时：", str(timedelta(seconds=int(round(time_diff)))))
'''绘制错误预测的'''
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
'''confusion_matrix'''
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
'''计算分类'''
# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256
def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {X: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    return correct, cls_pred
def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    
    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
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
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print statistics for the weights.
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def plot_layer_output(layer_output, image):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.
    feed_dict = {X: [image]}
    
    # Retrieve the output of the layer after inputting this image.
    values = session.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i<num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def plot_distorted_image(image, cls_true):
    # Repeat the input image 9 times.
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)

    # Create a feed-dict for TensorFlow.
    feed_dict = {X: image_duplicates}

    # Calculate only the pre-processing of the TensorFlow graph
    # which distorts the images in the feed-dict.
    result = session.run(distorted_images, feed_dict=feed_dict)

    # Plot the images.
    plot_images(images=result, cls_true=np.repeat(cls_true, 9))
def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]
img, cls = get_test_image(16)
plot_distorted_image(img, cls)

optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
plot_conv_weights(weights=weights_conv1, input_channel=0)
plot_layer_output(output_conv2, image=img)