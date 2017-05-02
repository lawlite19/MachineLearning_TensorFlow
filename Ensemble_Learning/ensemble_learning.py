#coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""using a so-called ensemble of convolutional neural networks. 
Instead of using a single neural network, we use several neural networks and average their outputs."""
print("tensorflow version", tf.__version__)
print("prettytensor version", pt.__version__)

'''打印一下数据集信息'''
data = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Size of:")  # 总共是70000张图片
print("\t training set:\t\t{}".format(len(data.train.labels)))
print("\t test set:\t\t\t{}".format(len(data.test.labels)))
print("\t validation set:\t{}".format(len(data.validation.labels)))
'''test和validation的真实值'''
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)
'''将training set和validation set合并，并重新划分'''
combine_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combine_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)
print("合并后图片：", combine_images.shape)
print("合并后label：", combine_labels.shape)
combined_size = combine_labels.shape[0]
train_size = int(0.8*combined_size)
validation_size = combined_size - train_size
'''函数：将合并后的重新随机划分'''
def random_training_set():
    idx = np.random.permutation(combined_size)   # 将0-combined_size数字随机排列
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]
    x_train = combine_images[idx_train, :]
    y_train = combine_labels[idx_train, :]
    
    x_validation = combine_images[idx_validation, :]
    y_validation = combine_images[idx_validation, :]
    return x_train, y_train, x_validation, y_validation
'''定义图片的相关信息'''
img_size = 28
img_size_flat = img_size ** 2
img_shape  = (img_size, img_size)
num_channels = 1
num_class = 10

'''函数：显示9张图片'''
def plot_images(images, cls_true, ensemble_cls_pred=None, best_cls_pred=None):
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(3, 3)
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape(img_shape), cmap="binary")
            if ensemble_cls_pred is None:
                xlabel = "True:{0}".format(cls_true[i])
            else:
                xlabel = "True:{0}\nEnsenble:{1}\nBest Net:{2}".\
                    format(cls_true[i], ensemble_cls_pred[i], best_cls_pred[i])
            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
#plot_images(images, cls_true)
        
'''定义tensorflow计算图'''
X = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="X")
X_image = tf.reshape(X, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_class], name="y")
y_true_cls = tf.argmax(y_true, axis=1)

X_pretty = pt.wrap(X_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = X_pretty.\
        conv2d(kernel=5, depth=16, name="layer_conv1").\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name="layer_conv2").\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name="layer_fc1").\
        softmax_classifier(num_classes=num_class, labels=y_true)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''保存网络变量'''
saver = tf.train.Saver(max_to_keep=100)  # max_to_keep最多网络的数量
save_dir = "checkpoints/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
'''函数：得到某个网络的保存路径'''
def get_save_path(net_number):
    return save_dir + "network" + str(net_number)

'''执行tesorflow计算图'''
session = tf.Session()
def init_variabels():
    session.run(tf.initialize_all_variables())
'''SGD'''
train_batch_size = 64
def random_batch(x_train, y_train):
    num_images = len(x_train)
    idx = np.random.choice(num_images, size=train_batch_size,replace=False) # 从0-num_images中选择train_batch_size个数字
    x_batch = x_train[idx, :]
    y_batch = y_train[idx, :]
    return x_batch, y_batch

'''函数：执行训练'''
def optimize(num_iterations, x_train, y_train):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch(x_train, y_train)
        feed_dict_train = {X: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "训练次数：{0:>6}, 训练batch上准确率: {1:>6.1%}"
            print(msg.format(i+1, acc))
    end_time = time.time()
    time_diff = end_time-start_time
    print("用时：", str(timedelta(seconds=int(round(time_diff)))))
'''训练5个神经网络 都使用上面定义的计算图，然后保存到硬盘'''
num_networks = 5
num_iterations = 2000
def run_5_networks():
    for i in range(num_networks):
        print("Neural networks:{0}".format(i))
        x_train, y_train, _, _ = random_training_set()
        init_variabels()
        optimize(num_iterations, x_train, y_train)
        saver.save(session, get_save_path(i))
        print()

'''函数：预测test set，使用batch'''
batch_size = 256
def predict_labels(images):
    num_images = len(images)
    pred_labels = np.zeros(shape=[num_images,num_class], dtype=np.float)
    i = 0
    while i < num_images:
        j = min(i+batch_size, num_images)
        feed_dict = {X: images[i:j,:]}
        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)
        i = j
    return pred_labels
'''函数：bool值，判断预测是否正确'''
def correct_prediction(images, labels, cls_true):
    pred_labels = predict_labels(images)
    cls_pred = np.argmax(pred_labels, axis=1)
    correct = (cls_true == cls_pred)
    return correct
'''函数：判断test set'''
def test_correct():
    return correct_prediction(data.test.images, 
                              data.test.labels,
                              data.test.cls)
'''validation set'''
def validation_correct():
    return correct_prediction(data.validation.images,
                              data.validation.labels,
                              data.validation.cls)

'''测试集准确度'''
def test_accuracy():
    correct = test_correct()
    return correct.mean()
'''验证集准确度'''
def validation_accuracy():
    correct = validation_correct()
    return correct.mean()
'''融合预测结果'''
def ensemble_predictions():
    pred_labels = []
    test_accuracies = []
    validation_accuracies = []
    for i in range(num_networks):
        saver.restore(sess=session, save_path=get_save_path(i))
        test_acc = test_accuracy()
        test_accuracies.append(test_acc)
        validation_acc = validation_accuracy()
        validation_accuracies.append(validation_acc)
        msg = "网络：{0}，验证集：{1:.4f}，测试集{2:.4f}"
        print(msg.format(i, validation_acc, test_acc))
        pred = predict_labels(data.test.images)
        pred_labels.append(pred)
    return np.array(pred_labels),\
           np.array(test_accuracies),\
           np.array(validation_accuracies)
#run_5_networks()
pred_labels, test_accuracies, val_accuracies = ensemble_predictions()
print(pred_labels.shape)
ensemble_pred_labels = np.mean(pred_labels, axis=0)
print(ensemble_pred_labels.shape)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
ensemble_correct = (ensemble_cls_pred == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)
print(test_accuracies)
best_net = np.argmax(test_accuracies)
print(best_net)
print(test_accuracies[best_net])
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == data.test.cls)
best_net_incorrect = np.logical_not(best_net_correct)
print("融合后预测对的：", np.sum(ensemble_correct))
print("单个最好模型预测对的", np.sum(best_net_correct))
ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)  # 融合之后好于单个的个数
print(ensemble_better.sum())
best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)  # 单个好于融合之后的个数
print(best_net_better.sum())

'''显示几个融合和最好的额模型预测结果的图片'''
def plot_images_comparison(idx):
    plot_images(data.test.images[idx,:],
                data.test.cls[idx],
                ensemble_cls_pred[idx],
                best_net_cls_pred[idx])
plot_images_comparison(idx=ensemble_better)
