#coding: utf-8
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import inception # 第三方类加载inception model
import os
'''使用训练好的inception model进行预测'''


print("tensorflow version", tf.__version__)

'''下载和加载inception model'''
inception.maybe_download()
model = inception.Inception()

'''预测和显示图片'''
def classify(image_path):
    plt.imshow(plt.imread(image_path))
    plt.show()
    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)
#'''大熊猫图片预测'''
#image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
#classify(image_path)
'''金刚鹦鹉macaw
- inception默认处理的是299*299像素的图片
- 这里这个图片是320*785的，它会自动处理成需要的大小resize
'''
#image_path = os.path.join(inception.data_dir, 'parrot.jpg')
#classify(image_path) 
'''显示处理后图片的样式'''
#def plot_resized_image(image_path):
    #resized_image = model.get_resized_image(image_path)
    #plt.imshow(resized_image, interpolation='nearest')
    #plt.show()
#plot_resized_image(image_path)

'''金刚鹦鹉macaw  裁剪后的图片，看是否能预测正确'''
#image_path = os.path.join(inception.data_dir, 'parrot_cropped1.jpg')
#classify(image_path)

'''随便拿来一张图片预测'''
image_path = os.path.join(inception.data_dir, 'elon_musk.jpg')
classify(image_path)