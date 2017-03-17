import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)
'''Load MNIST data and print some information'''
data = input_data.read_data_sets("MNIST_data", one_hot = True)
print("Size of:")
print("\t training-set:\t\t{}".format(len(data.train.labels)))
print("\t test-set:\t\t\t{}".format(len(data.test.labels)))
print("\t validation-set:\t{}".format(len(data.validation.labels)))
print(data.test.labels[0:5])
data.test.cls = np.array([label.argmax() for label in data.test.labels])   # get the actual value
print(data.test.cls[0:5])
'''define the images information'''
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10
'''define a funciton to plot 9 images'''
def plot_images(images, cls_true, cls_pred = None):
    '''
    @parameter images:   the images info
    @parameter cls_true: the true value of image
    @parameter cls_pred: the prediction value, default is None
    '''
    assert len(images) == len(cls_true) == 9  # only show 9 images
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="binary")  # binary means black_white image
        # show the true and pred values
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0},Pred: {1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])  # remove the ticks
        ax.set_yticks([])
    plt.show()
'''show 9 images'''
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images, cls_true)

'''define the placeholder'''
X = tf.placeholder(tf.float32, [None, img_size_flat])    # None means the arbitrary number of labels, the features size is img_size_flat 
y_true = tf.placeholder(tf.float32, [None, num_classes]) # output size is num_classes
y_true_cls = tf.placeholder(tf.int64, [None])

'''define weights and biases'''
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))  # img_size_flat*num_classes
biases = tf.Variable(tf.zeros([num_classes]))

'''define the model'''
logits = tf.matmul(X,weights) + biases 
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, 
                                                       logits=logits)
cost = tf.reduce_mean(cross_entropy)
'''define the optimizer'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
'''define the accuracy'''
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''run the datagraph and use batch gradient descent'''
session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100
'''define a function to run the optimizer'''
def optimize(num_iterations):
    '''
    @parameter num_iterations: the traning times
    '''
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {X: x_batch,y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        
        
feed_dict_test = {X: data.test.images, 
                  y_true: data.test.labels, 
                  y_true_cls: data.test.cls}        

'''define a function to print the accuracy'''    
def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set:{0:.1%}".format(acc))
'''define a function to printand plot the confusion matrix using scikit-learn.'''   
def print_confusion_martix():
    cls_true = data.test.cls  # test set actual value 
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)  # test set predict value
    cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)        # use sklearn confusion_matrix
    print(cm)
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues) # Plot the confusion matrix as an image.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')    
    plt.show()
'''define a function to plot the error prediciton'''    
def plot_example_errors():
    correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test) 
    incorrect = (correct == False)
    images = data.test.images[incorrect]  # get the prediction error images
    cls_pred = cls_pred[incorrect]        # get prediction value
    cls_true = data.test.cls[incorrect]   # get true value
    plot_images(images[0:9], cls_true[0:9], cls_pred[0:9])
'''define a fucntion to plot weights'''
def plot_weights():
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(0.3, 0.3)
    for i, ax in enumerate(axes.flat):
        if i<10:
            image = w[:,i].reshape(img_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min,vmax=w_max,cmap="seismic")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
optimize(num_iterations=100)
print_accuracy()
plot_example_errors()
plot_weights()
print_confusion_martix()