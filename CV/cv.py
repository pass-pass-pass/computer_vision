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


zeros = torch.rand()
one_to = torch.arange(start = 1 ,end  = 1089, step = 123 )
