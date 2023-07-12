import numpy as np 
import  pandas as pd    
import matplotlib.pyplot as plt 
from matplotlib import style

import seaborn as sns 

df= pd.read_csv('../CHI_Study_001/A0381c_210609-151615/ACC.csv')
df_IBI = pd.read_csv('../CHI_Study_001/A0381c_210609-151615/IBI.csv')



beat_interval1   =   pd.read_csv('../CHI_Study_001/A0381c_210609-151615/IBI.csv')[2:238]
beat_interval2    =  pd.read_csv('../CHI_Study_001/A0381c_210610-133939/IBI.csv')[2:238]
beat_interval3    =  pd.read_csv('../CHI_Study_001/A0381c_210609-151615/IBI.csv')[2:238]
beat_interval4    =  pd.read_csv('../CHI_Study_001/A0381c_210611-140506/IBI.csv')[2:238]
beat_interval5    =  pd.read_csv('../CHI_Study_001/A0381c_210613-152026/IBI.csv')[2:238]
beat_interval6    =  pd.read_csv('../CHI_Study_001/A0381c_210614-152007/IBI.csv')[2:238]
beat_interval7    =  pd.read_csv('../CHI_Study_001/A0381c_210615-122418/IBI.csv')[2:238]
beat_interval8    =  pd.read_csv('../CHI_Study_001/A0381c_210615-182034/IBI.csv')[2:238]
beat_interval9    =  pd.read_csv('../CHI_Study_001/A0381c_210617-183137/IBI.csv')[2:238]
beat_interval10  =   pd.read_csv('../CHI_Study_001/A0381c_210618-192901/IBI.csv')[2:238]



photoplethysmograph_data1  = pd.read_csv('../CHI_Study_001/A0381c_210609-151615/BVP.csv')[2:240]
photoplethysmograph_data2  = pd.read_csv('../CHI_Study_001/A0381c_210610-133939/BVP.csv')[2:240]
photoplethysmograph_data3  = pd.read_csv('../CHI_Study_001/A0381c_210609-151615/BVP.csv')[2:240]
photoplethysmograph_data4  = pd.read_csv('../CHI_Study_001/A0381c_210611-140506/BVP.csv')[2:240]
photoplethysmograph_data5  = pd.read_csv('../CHI_Study_001/A0381c_210613-152026/BVP.csv')[2:240]
photoplethysmograph_data6  = pd.read_csv('../CHI_Study_001/A0381c_210614-152007/BVP.csv')[2:240]
photoplethysmograph_data7  = pd.read_csv('../CHI_Study_001/A0381c_210615-122418/BVP.csv')[2:240]
photoplethysmograph_data8  = pd.read_csv('../CHI_Study_001/A0381c_210615-182034/BVP.csv')[2:240]
photoplethysmograph_data9  = pd.read_csv('../CHI_Study_001/A0381c_210617-183137/BVP.csv')[2:240]
photoplethysmograph_data10 = pd.read_csv('../CHI_Study_001/A0381c_210618-192901/BVP.csv')[2:240]






beat_interval = np.hstack((
beat_interval1  ,
beat_interval2  ,
beat_interval3  ,
beat_interval4  ,
beat_interval6  ,
beat_interval7  ,
beat_interval8  ,
beat_interval9  ,
beat_interval10 
))











Average_heart_rate1    =  pd.read_csv('../CHI_Study_001/A0381c_210609-151615/HR.csv')[2:8000]
Average_heart_rate2    =  pd.read_csv('../CHI_Study_001/A0381c_210610-133939/HR.csv')[2:8000]
Average_heart_rate3    =  pd.read_csv('../CHI_Study_001/A0381c_210609-151615/HR.csv')[2:8000]
Average_heart_rate4    =  pd.read_csv('../CHI_Study_001/A0381c_210611-140506/HR.csv')[2:8000]
Average_heart_rate5    =  pd.read_csv('../CHI_Study_001/A0381c_210613-152026/HR.csv')[2:8000]
Average_heart_rate6    =  pd.read_csv('../CHI_Study_001/A0381c_210614-152007/HR.csv')[2:8000]
Average_heart_rate7    =  pd.read_csv('../CHI_Study_001/A0381c_210615-122418/HR.csv')[2:8000]
Average_heart_rate8    =  pd.read_csv('../CHI_Study_001/A0381c_210615-182034/HR.csv')[2:8000]
Average_heart_rate9    =  pd.read_csv('../CHI_Study_001/A0381c_210617-183137/HR.csv')[2:8000]
Average_heart_rate10  =   pd.read_csv('../CHI_Study_001/A0381c_210618-192901/HR.csv')[2:8000]



accelerator1    =     pd.read_csv('../CHI_Study_001/A0381c_210609-151615/ACC.csv')[2:8000]
accelerator2    =  pd.read_csv('../CHI_Study_001/A0381c_210610-133939/ACC.csv')[2:8000]
accelerator3    =  pd.read_csv('../CHI_Study_001/A0381c_210609-151615/ACC.csv')[2:8000]
accelerator4    =  pd.read_csv('../CHI_Study_001/A0381c_210611-140506/ACC.csv')[2:8000]
accelerator5    =  pd.read_csv('../CHI_Study_001/A0381c_210613-152026/ACC.csv')[2:8000]
accelerator6    =  pd.read_csv('../CHI_Study_001/A0381c_210614-152007/ACC.csv')[2:8000]
accelerator7    =  pd.read_csv('../CHI_Study_001/A0381c_210615-122418/ACC.csv')[2:8000]
accelerator8    =  pd.read_csv('../CHI_Study_001/A0381c_210615-182034/ACC.csv')[2:8000]
accelerator9    =  pd.read_csv('../CHI_Study_001/A0381c_210617-183137/ACC.csv')[2:8000]
accelerator10  =   pd.read_csv('../CHI_Study_001/A0381c_210618-192901/ACC.csv')[2:8000]




y_accelerator = np.hstack((
accelerator1 ,
accelerator2 ,
accelerator3 ,
accelerator4 ,
accelerator5 ,
accelerator6 ,
accelerator7 ,
accelerator8 ,
accelerator9 ,
accelerator10
))



y_average_rate = np.hstack((Average_heart_rate1,
Average_heart_rate2,
Average_heart_rate3,
Average_heart_rate4,
Average_heart_rate6,
Average_heart_rate7,
Average_heart_rate8,
Average_heart_rate9,
Average_heart_rate10,
))


x= df.values[2:,0]
time = df.values[0, :]
y = df.values[2:, 1]
z = df.values[2:, 2]

y_ibi_first = df_IBI.values[2: , 0]
y_ibi_second = df_IBI.values[2:, 1]
plt.figure(figsize=(14,11))

plt.subplot(2,2,1)
vitual_time = np.linspace(1,10, len(x))

plt.scatter(vitual_time,y, color = '#1f77b4', label = 'acceleration in y axis', marker = '*')
plt.scatter(vitual_time,x, color = '#9467bd', label = 'acceleration in x axis', marker = '*')
plt.scatter(vitual_time,z, color = '#bcbd22', label = 'acceleration in z axis', marker = '*')
plt.legend()

plt.title('acceleration performance in three dimensions')

plt.subplot(2, 2,2 )
big_x = np.linspace(1, 10, 238)
plt.scatter(big_x, y_ibi_first, color = 'g', linestyle = '-', label = ' beat interval respect to initial time')
plt.scatter(big_x, y_ibi_second, color = 'r' , linestyle= '-', label = 'heart beat intervals')
print(y_average_rate.shape)
plt.legend() 

plt.title('heart beat interval ')
plt.xlabel('time ')
plt.ylabel(' the time interval')


plt.subplot(2,2,3)

new_x = np.linspace(1,10, y_average_rate.shape[0])

plt.scatter(new_x , y_average_rate[:, 0], color = 'b', linestyle='-', label = 'average heart rate')

plt.title('average heart rate')
plt.xlabel('patients ')

plt.ylabel(' the average rate')




plt.legend()


plt.subplot(2,2,4)
print(accelerator1.values[:238, 0].shape, '  the shape for    ')
 
ynew = np.vstack((y_average_rate[:238 ,0].flatten() , y_ibi_first  , y_ibi_second,  accelerator1.values[:238, 0].flatten()  ,accelerator1.values[:238, 1].flatten(), accelerator1.values[:238, 2].flatten(),  photoplethysmograph_data1.values.flatten()   ))
print(ynew.T.shape)
df_new = pd.DataFrame(ynew.T, columns = ['average heart rate' ,  'the first heart beat', 'the interval from the previous heart beat', 'accelerator in x axie' , ' accelerator in y axie', ' accelerator in z axie', 'photoplethysmograph'])
heatmap=  sns.heatmap(df_new.corr())
heatmap.set_title('correlation map between average heart rate and other potential factors')

plt.subplots_adjust(left = .1, top = .925, bottom = .187, hspace = .14)

fig = plt.figure(figsize = (10, 7))



plt.boxplot(y_average_rate)
plt.title('the box graph of average heart rate')
# plt.show() 



count = 0
tem = 0
plt.figure(figsize = (13, 9))
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.scatter(   beat_interval[:, i + tem]  ,  y_accelerator[:, i + count - 1][:beat_interval[:, i + tem].shape[0]]   , color = '#9467bd', label = 'heart beat interval against acceleration in x axis', marker = '*')
    plt.scatter(   beat_interval[:, i + tem]  ,  y_accelerator[: , i + count   ][:beat_interval[:, i + tem].shape[0]]          , color = '#1f77b4', label = 'heart beat interval against acceleration in y axis', marker = '*')
    plt.scatter(   beat_interval[:, i + tem]  ,  y_accelerator[:, i + count + 1][:beat_interval[:, i + tem].shape[0]] , color = '#bcbd22', label = 'heart beat interval against acceleration in z axis', marker = '*')
    count +=2
    tem +=1
    plt.xlabel('the heart interval  ')

    plt.ylabel(' acceleration in three dimensions')
    plt.legend()
# plt.show() 

plt.figure(figsize = (13, 9))

for i in range(1,10):
    plt.subplot(3,3,i)
    x_new =  np.linspace(1,30, y_average_rate.shape[0])
    plt.scatter(x_new, y_average_rate[:, i - 1]  , label = 'average heart rate' , color = '#bcbd22')
    plt.xlabel('time ')

    plt.ylabel(' the average heart rate')

    plt.legend()
plt.show()






# df.sample(19)

# df.loc[12:14]
# # new line to test thoughts
# from urllib.request import urlretrieve
# urlretrieve('https://hub.jovian.ml/wp-content/uploads/2020/08/climate.csv')
# covid_Df = pd.read_csv('taly-covid-daywise.csv')
# print(covid_Df)



