import sys, os
sys.path.append(os.pardir)
import numpy as np
from layers import Pooling

#我自己编写的用于理解的代码
X=[[[[1,2,3,0],
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
    [4,2,4,5]]]]

x=np.array(X)
print('x.shape:',x.shape)
a=Pooling(2,2,2,0)
print(a.forward(x))
X1=[[
    [[2,4],
     [3,4]],

    [[4,6],
     [3,3]],

    [[4,4],
     [4,6]]
]]
x1=np.array(X1)
print('x1.shape:',x1.shape)
print(a.backward(x1))