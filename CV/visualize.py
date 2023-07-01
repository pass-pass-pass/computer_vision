import numpy as np 
import  pandas as pd    
import matplotlib.pyplot as plt 
from matplotlib import style

import seaborn as sns 

df= pd.read_csv('../CHI_Study_000/000000_210603-191049/ACC.csv')
x= df.values[2:,0]
time = df.values[0, :]
y = df.values[2:, 1]
z = df.values[2:, 2]
plt.figure(figsize=(8,8))
plt.subplot(3,1,1)
plt.plot(x, y)  
plt.show()
