import torch
from torch import nn  
# import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.model_selection  import  train_test_split

 
#  preparing and loading
# create some data
weight = .9
bias = .3



start =0 
end = 1
step = .02
#  unsqueeze to make each data as a new dimension in the bracket
X = torch.arange(start, end ,step).unsqueeze(dim = 1)
y = weight * X + bias
#  splitting the data  into  testing and training
#  training validation test 
train = int(.8 * len(X))
x_train , y_train = X[:40], y[:40]
x_test , y_test = X[40:], y[40:]
def plot_prediciton(train_data = x_train,train_labels = y_train, test_data = x_test, test_label = y_test, predictions = None):
    plt.figure(figsize = (10, 7))
    plt.scatter(train_data, train_labels, c = 'b', s = 4, label = 'traning data' )
    plt.scatter(test_data, test_label, c= 'g', s= 4, label = 'testing data')
    # predictions
    if predictions is not None:
        plt.scatter(test_data, predictions, label = 'prediciton', s =4, c='r')
    plt.legend(prop = {'size':14   })
    plt.show()
plot_prediciton()


# machine learning started 
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.rand(1,  dtype = torch.float ))    #random weights, and find the most ideal one after this
        self.bias = nn.Parameter(torch.rand(1,dtype = torch.float))            # samel, random bias

    def forward(self , x:torch.Tensor) -> torch.Tensor:             # define the computation 
        return self.weights * x + self.bias                        # formula
