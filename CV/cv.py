import cv2 as cv

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt


#  if there are too large images it goes off screen



#  rading videos

# capt = cv.VideoCapture('/dog.mp4')




# capt = cv.absdiff()

# import pytorch

import torch 

print(torch.__version__)
scalar = torch.tensor(7)
print(scalar.ndim)


#  creating tensor
scalar = torch.tensor(8)
scalar2 = torch.tensor(9)
print(scalar.ndim)
print(scalar.item())

#  vector
vector = torch.tensor([8,8])
print(vector.ndim)
print(vector.shape)

#  matrix

matrix = torch.tensor([    [1,2]        ,       [3,4]   ])

matrix[0]
# tensor
TENSOR = torch.tensor([[[1,2,3]     
                        ,[1,2,3], 
                        [3,4,5],
                           ]])


tensor1 = torch.tensor([[[[[1,2,3], [1,2,3] , 
                           [3,45,5],
                           [4,5,6],
                           [3,9,0]]]]])
print(tensor1.ndim)



print(tensor1.shape)
random = torch.rand(3,4)

zeros = torch.arange(start = 0 , end = 32, step = 1)

one_to = torch.arange(start = 1 ,end  = 1089, step = 123 )

torch.matmul(one_to, one_to)
# # find mean
torch.min(zeros, dim = 0)
zeros.min( zeros)
torch.max(zeros)
torch.sum(zeros)

# finf the position of the min value
zeros.argmin()

# max
zeros.argmax()


s = np.zeros((3,2))
#  from numpy to pytorch
tensor3 = torch.from_numpy(s)
#  from pytorch to numpy
tensor4 = zeros.numpy()



#  seed


random_tensor = torch.rand(3,4)
random_tensor_b = torch.rand(3,4)

#  set the seed for the random   the same seed produce the same random number
torch.manual_seed(32)

#  reproduciblity
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# device
tensor = torch.tensor([1,2,3])
print(tensor.device)
#  movev tensor to gpu
tensor_gpu = tensor.to(device)
print(tensor_gpu , tensor_gpu.device)


