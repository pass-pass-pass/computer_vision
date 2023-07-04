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
# plt.figure(figsize=(8,8))
# plt.subplot(3,1,1)
# plt.plot(x, y)  
# plt.show()

df.sample(19)

df.loc[12:14]
# new line to test thoughts
from urllib.request import urlretrieve
urlretrieve('https://hub.jovian.ml/wp-content/uploads/2020/08/climate.csv')
covid_Df = pd.read_csv('taly-covid-daywise.csv')
print(covid_Df)



