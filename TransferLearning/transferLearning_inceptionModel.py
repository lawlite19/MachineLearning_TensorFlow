#coding: utf-8
import numpy as np
import tensorflow as tf
import prettytensor as pt
from matplotlib import pyplot as plt
import time
from datetime import timedelta
import os
import inception   # 第三方下载inception model的代码
from inception import transfer_values_cache  # cache
import cifar10     # 也是第三方的库，下载cifar-10数据集
from cifar10 import num_classes

print("tensorflow version:", tf.__version__)
print("prettytensor version", pt.__version__)

'''下载cifar-10数据集'''
cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
print("所有类别是：",class_names)

'''训练和测试集'''
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test,  cls_test,  labels_test  = cifar10.load_test_data()
print("Size of:")
print("  training set:\t\t{}".format(len(images_train)))
print("  test set:\t\t\t{}".format(len(images_test)))

'''显示9张图片函数'''
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(3, 3)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], interpolation=interpolation)
            if cls_pred is None:
                xlabel = "True:{0}".format(class_names[cls_true[i]])
            else:
                xlabel = "True:{0}, \n Pred:{1}".format(class_names[cls_true[i]], class_names[cls_pred[i]])
            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
#images = images_test[0:9]
#cls_true = cls_test[0:9]
#plot_images(images, cls_true)

'''下载inception model'''
inception.maybe_download()
model = inception.Inception()
'''训练和测试的cache的路径'''
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

print('处理训练集上的transfer-values.......... ')
image_scaled = images_train * 255.0  # cifar-10的pixel是0-1的, shape=(50000, 32, 32, 3)
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=image_scaled, 
                                              model=model)  # shape=(50000, 2048)
print('处理测试集上的transfer-values.......... ')
images_scaled = images_test * 255.0
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             model=model,
                                             images=images_scaled)
print("transfer_values_train: ",transfer_values_train.shape)
print("transfer_values_test: ",transfer_values_test.shape)

'''显示transfer values'''
def plot_transfer_values(i):
    print("输入图片：")
    plt.imshow(images_test[i], interpolation='nearest')
    plt.show()
    print('transfer values --> 此图片在inception model上')
    img = transfer_values_test[i]
    img = img.reshape((32, 64))
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()
plot_transfer_values(16)

'''使用PCA分析transfer values'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
transfer_values = transfer_values_train[0:3000]  # 取3000个，大的话计算量太大
cls = cls_train[0:3000]
print(transfer_values.shape)
transfer_values_reduced = pca.fit_transform(transfer_values)
print(transfer_values_reduced.shape)
## 显示降维后的transfer values
def plot_scatter(values, cls):
    from matplotlib import cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
    colors = cmap[cls]
    x = values[:, 0]
    y = values[:, 1]
    plt.scatter(x, y, color=colors)
    plt.show()
plot_scatter(transfer_values_reduced, cls)
'''使用t-SNE分析transfer values
因为t-SNE运行非常慢，所以这里先用PCA将到50维
'''
from sklearn.manifold import TSNE
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)
tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
print("最终降维后：", transfer_values_reduced.shape)
plot_scatter(transfer_values_reduced, cls)

'''创建网络'''
transfer_len = model.transfer_len   # 获取transfer values的大小，这里是2048
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name="x")
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")
y_true_cls = tf.argmax(y_true, axis=1)
x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(1024, name="layer_fc1").\
        softmax_classifier(num_classes, labels=y_true)
'''优化器'''
global_step = tf.Variable(initial_value=0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step)
'''accuracy'''
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''SGD 训练'''
session = tf.Session()
session.run(tf.initialize_all_variables())
train_batch_size = 64
def random_batch():
    num_images = len(images_train)
    idx = np.random.choice(num_images, 
                           size=train_batch_size,
                           replace=False)
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]
    return x_batch, y_batch
def optimize(num_iterations):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)
        if (i_global % 100 == 0) or (i==num_iterations-1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))            
    end_time = time.time()
    time_diff = end_time - start_time
    print("耗时：", str(timedelta(seconds=int(round(time_diff)))))

'''batch 预测'''
batch_size = 256
def predict_cls(transfer_values, labels, cls_true):
    num_images = len(images_test)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true == cls_pred)
    return correct, cls_pred


'''显示预测错误的'''
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

    n = min(9, len(images))
    
    # Plot the first n images.
    plot_images(images=images[0:n],
                cls_true=cls_true[0:n],
                cls_pred=cls_pred[0:n])
'''显示confusion matrix'''
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix

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

def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
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

print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)
optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)