import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def distance(x,m):
    return (np.dot(x.T,m)-0.5*np.dot(m,m.T));


print("Ashiqur Rahman, ID : 150204057")

df_train = pd.read_csv('train.txt', sep=" ", header = None,dtype='Int64')
df_train = pd.DataFrame(df_train.values, columns = ['X', 'Y', 'Class'])


df_test= pd.read_csv('test.txt', sep=" " ,  header = None,dtype='Int64')
df_test1 = pd.DataFrame(df_test.values, columns = ['X', 'Y', 'Class'])

df_test2=df_test1.copy();

df_test=df_test1[['X', 'Y']];
df_test=df_test.values


df1=df_train[df_train['Class'] == 1];
df1=df1.iloc[:,0:2]
w1=df1.values

df2=df_train[df_train['Class'] == 2];
df2=df2.iloc[:,0:2]
w2=df2.values

w1_x=w1[:,0]
w1_y=w1[:,1]

w2_x=w2[:,0]
w2_y=w2[:,1]

w1_mean=np.array([np.mean(w1_x),np.mean(w1_y)]);
w2_mean=np.array([np.mean(w2_x),np.mean(w2_y)]);



for i in range(0,df_test.shape[0]):
    if(distance(df_test[i], w1_mean)> distance(df_test[i], w2_mean)):
        df_test2.loc[i,'Class']=1
    else:
        df_test2.loc[i,'Class']=2


w1_class=df_test2[df_test2['Class']==1].values
w2_class=df_test2[df_test2['Class']==2].values


x=np.arange(min(w1_x.min(),w2_x.min()),max(w1_x.max(),w2_x.max())+2)

y=((w1_mean[0]-w2_mean[0])*x - 0.5*(np.dot(w1_mean,w1_mean.T)-np.dot(w2_mean,w2_mean.T)))/ (w2_mean[1]-w1_mean[1])

total_point=df_test.shape[0]

count=0;

for i in range(0,df_test.shape[0]):
    if(df_test2.loc[i,'Class']==df_test1.loc[i,'Class']):
        count=count+1;


print("Total number of Points :",total_point," Perfectly Classified:",total_point-count)

if((total_point-count)==0):
    print("Accuracy: 100%")
else:
    print("Accuracy :",((total_point-count)/total_point)*100,'%')

plt.scatter(w1_x,w1_y,color = 'red', marker = 'o',label="w1")
plt.scatter(w2_x,w2_y,color = 'blue', marker = '+',label="w2")
plt.scatter(w1_mean[0],w1_mean[1],color = 'orange', marker = '<',label="mean of w1")
plt.scatter(w2_mean[0],w2_mean[1],color = 'grey', marker = '>',label="mean of w2")
plt.scatter(w1_class[:,0],w1_class[:,1],color = 'green', marker = 'o',label="class of w1")
plt.scatter(w2_class[:,0],w2_class[:,1],color = 'black', marker = '+',label="class of w2")


plt.plot(x,y,'r--',label='Decision Boundary')
plt.title(" Minimum distance to class mean classifier")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend(loc="best",fontsize="small")

plt.show();



