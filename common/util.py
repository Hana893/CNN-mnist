# coding: utf-8
import numpy as np


def smooth_curve(x):
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """打乱数据集

    Parameters
    ----------
    x : 训练数据
    t : 监督数据

    Returns
    -------
    x, t : 打乱的训练数据和监督数据
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

"""
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    "

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    "
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

"""

"""col2im不是im2col的逆处理，感觉col2im有错误
input_data=np.array([[[[10,7,6,2],
                       [4,6,9,3],
                       [2,5,1,4]]]])
print('input_data_shape:',input_data.shape)
x=im2col(input_data,2,2,1,0)
print('input_data_im2col:')
print(x)
y=col2im(x,input_data.shape,2,2,1,0)
print('***********')
print(y)

运行结果：
input_data_shape: (1, 1, 3, 4)
input_data_im2col:
[[10.  7.  4.  6.]
 [ 7.  6.  6.  9.]
 [ 6.  2.  9.  3.]
 [ 4.  6.  2.  5.]
 [ 6.  9.  5.  1.]
 [ 9.  3.  1.  4.]]
***********
[[[[10. 14. 12.  2.]
   [ 8. 24. 36.  6.]
   [ 2. 10.  2.  4.]]]]
"""

#这个是我修改后的col2im，经检验是im2col的逆处理


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad , W + 2*pad ))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]



"""
X=[
   [[[1,2,3,0],
    [0,1,2,4],
    [1,0,4,2],
    [3,2,0,1]],

   [[3,0,6,5],
    [4,2,4,3],
    [3,0,1,0],
    [2,3,3,1]],

   [[4,2,1,2],
    [0,1,0,4],
    [3,0,6,2],
    [4,2,4,5]]],


  [[[1,4,3,0],
    [0,6,2,4],
    [1,0,8,2],
    [3,2,10,1]],

   [[3,0,7,5],
    [4,9,4,3],
    [3,0,2,0],
    [2,5,3,1]],

   [[4,0,1,2],
    [0,0,0,4],
    [3,0,5,2],
    [4,2,6,5]]],
   ]
input_data=np.array(X)
print('input_data_shape:',input_data.shape)
x=im2col(input_data,2,2,2,0)
print('input_data_im2col:')
print(x)
y=col2im(x,input_data.shape,2,2,2,0)
print('***********')
print(y)

"""

#我编写的col2im和原来的col2im都能完成CNN的学习推理，原来的col2im最终准确率为0.958，我的为0.953