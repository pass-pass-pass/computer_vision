import numpy as np 
import  pandas as pd    
import matplotlib.pyplot as plt 
from matplotlib import style

import seaborn as sns 

df= pd.read_csv('../CHI_Study_001/A0381c_210609-151615/ACC.csv')
df_IBI = pd.read_csv('../CHI_Study_001/A0381c_210609-151615/IBI.csv')
Average_heart_rate = pd.read_csv('../CHI_Study_001/A0381c_210609-151615/HR.csv')
y_average_rate = Average_heart_rate.values[2:,:]
x= df.values[2:,0]
time = df.values[0, :]
y = df.values[2:, 1]
z = df.values[2:, 2]

y_ibi_first = df_IBI.values[2: , 0]
y_ibi_second = df_IBI.values[2:, 1]
plt.figure(figsize=(8,8))
plt.subplot(3,1,1)
vitual_time = np.linspace(1,10, len(x))

plt.scatter(vitual_time,y, color = '#1f77b4', label = 'acceleration in y axis', marker = '*')
plt.scatter(vitual_time,x, color = '#9467bd', label = 'acceleration in x axis', marker = '*')
plt.scatter(vitual_time,z, color = '#bcbd22', label = 'acceleration in z axis', marker = '*')
plt.legend()

plt.title('acceleration performance in three dimensions')

plt.subplot(3, 1,2 )
big_x = np.linspace(1, 10, 238)
plt.scatter(big_x, y_ibi_first, color = 'g', linestyle = '-', label = ' beat interval respect to initial time')
plt.scatter(big_x, y_ibi_second, color = 'r' , linestyle= '-', label = 'heart beat intervals')
print(y_average_rate.shape)
plt.legend() 

plt.title('heart beat interval ')


plt.subplot(3,1,3)

new_x = np.linspace(1,10, 9652)
plt.scatter(new_x , y_average_rate, color = 'b', linestyle='-', label = 'average heart rate')

plt.title('average heart rate')



plt.legend()
plt.show()









# df.sample(19)

# df.loc[12:14]
# # new line to test thoughts
# from urllib.request import urlretrieve
# urlretrieve('https://hub.jovian.ml/wp-content/uploads/2020/08/climate.csv')
# covid_Df = pd.read_csv('taly-covid-daywise.csv')
# print(covid_Df)



