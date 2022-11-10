#我编写的一部分，用于输出前1000个测试图像中cnn识别错误的图像
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network=DeepConvNet()
network.load_params('deep_convnet_params.pkl')

curr_sub=1                         #代表使用哪块subplot
sub_y,sub_x=4,5                    #代表有多少块subplot
fig = plt.figure()
for i in range(25):                #10代表输出前10*100个测试图像中cnn识别错误的图像，可更改，但需注意该值太大时下面的add_subplot可能会容纳不了
    network.accuracy(x_test[i*100:(i+1)*100],t_test[i*100:(i+1)*100])
    t=network.t
    y=network.y
    sig=(t!=y)
    err_x_test=x_test[i*100:(i+1)*100][sig]
    correct_label=t_test[i*100:(i+1)*100][sig]
    err_lable=y[sig]
    num=len(err_x_test)
    if num>=1:
        for j in range(num):
            ax = fig.add_subplot(sub_y, sub_x, curr_sub, xticks=[], yticks=[])
            ax.set(xlabel=f'correct label{correct_label[j]}',ylabel=f'error label{err_lable[j]}')
            ax.imshow(err_x_test[j].reshape(28,28))
            curr_sub+=1
            if curr_sub > sub_y * sub_x:  # 识别错误图像超过subplot就退出循环，不再检测
                break
    if curr_sub>sub_y*sub_x:       #识别错误图像超过subplot就退出循环，不再检测
        break

plt.tight_layout()
plt.show()

"""
network.accuracy(x_train,t_train):0.9983333333333333
network.accuracy(x_test,t_test):0.9935
"""