# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:34:30 2018

@author: Roshan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_all_stocks=pd.read_csv('all_stocks_5yr.csv')
X=dataset_all_stocks.iloc[:,:].values
j1=0
k=0
dict_X={}
l=[]

#extracting companies that have complete weekday data
for i in range(0,X.shape[0]):
    if X[k][6]==X[i][6]:
        j1=j1+1        
    else:
        if (j1==1258):
            dict_X[k]=j1
        j1=0
        k=i
if (j1==1258):
    dict_X[k]=j1

#seperating company wise data
dict_dataset_comp={}
X_dataset_comp=[]
for i in list(dict_X.keys()):
    for j in range(0,dict_X[i]):
        l.append(X[j+i])
    X_dataset_comp.append(l)
    dict_dataset_comp[len(X_dataset_comp)-1]=X[i][X.shape[1]-1]
    l=[]

#extracting oil prices
oil_prices_dataset=pd.read_csv('oil_prices.csv')
Y=oil_prices_dataset.iloc[:,:].values

date_X=list(dataset_all_stocks.iloc[0:len(X_dataset_comp[0]),0].values)
date_X=set(date_X)
date_X=list(date_X)
date_Y=list(oil_prices_dataset.iloc[:,0].values)
date_X.sort()
date_Y.sort()
date_unequal_index=[]

for i in range(0,len(date_X)):
    if (date_X[i]!=date_Y[i]):
        date_unequal_index.append(date_Y.index(date_Y[i]))
        date_Y.remove(date_Y[i])

date_Y.remove(date_Y[len(date_Y)-1])
list1_oil_price_add=[]
for i in range(0,len(date_X)):
    pos=-1
    for j in range(0,Y.shape[0]):
        if (Y[j][0]==date_Y[i]):
            pos=j
            break
    a=[]
    a.append(Y[j][1])
    list1_oil_price_add.append(a)

#extracting exchange rate
exchange_dataset=pd.read_csv('exchange_rate.csv')    
e=exchange_dataset.iloc[:,:].values
j=0
e=np.delete(e,0,axis=0)
e=np.delete(e,0,axis=0)
e=np.delete(e,0,axis=0)
e=np.delete(e,0,axis=0)
rows=e.shape[0]
for i in range(0,rows):
    if i==e.shape[0]:
        break
    f=0
    for j in range(0,len(date_X)):    
        if (date_X[j]==e[i][0]):
            f=1
            break
    if f==0:
        e=np.delete(e,i,axis=0)
avg=0
x=0
list1_exchange_values=[]
for i in e:
    a=[]
    if (i[1]!='.'):
        avg=(avg*x+float(i[1]))/(x+1)    
    else:
        i[1]=avg
    a.append(float(i[1]))
    list1_exchange_values.append(a)

#merging company wise oil prices and exchange rates
for i in range(0,len(X_dataset_comp)):
    X_dataset_comp[i]=np.hstack((X_dataset_comp[i],list1_oil_price_add))
    X_dataset_comp[i]=np.hstack((X_dataset_comp[i],list1_exchange_values))

#convert date to year,month,day
a_year=[]
a_month=[]
a_day=[]

for i in range(0,len(date_X)):
    date_c=date_X[i].split("-")
    for j in range(0,3):
        date_c[j]=float(date_c[j])    
    a=[]
    a.append(date_c[0])
    a_year.append(a)
    a=[]
    a.append(date_c[1])
    a_month.append(a)
    a=[]
    a.append(date_c[2])
    a_day.append(a)
    
    
for i in range(0,len(X_dataset_comp)):
    X_dataset_comp[i]=np.hstack((X_dataset_comp[i],a_year))
    X_dataset_comp[i]=np.hstack((X_dataset_comp[i],a_month))
    X_dataset_comp[i]=np.hstack((X_dataset_comp[i],a_day))

#merging monthly datasets indexing
cpi_d=pd.read_csv('CPI_Index.csv')
cpi=cpi_d.iloc[:,:].values
l_cpi_index=[]
k=0
for i in range(0,6):
    l_year=[]
    for j in range(0,12):
        if (i==5 and j==2):
            break
        if (i==0 and j==0):
            l_year.append(0)
        else:
            l_year.append(cpi[k][6])
            k=k+1
    l_cpi_index.append(l_year)

fedfund_d=pd.read_csv('FEDFUNDS.csv')
fedfund=fedfund_d.iloc[:,:].values
l_fedfund_index=[]
k=0
for i in range(0,6):
    l_year=[]
    for j in range(0,12):
        if (i==5 and j==1):
            break
        if (i==0 and j==0):
            l_year.append(0)
        else:
            l_year.append(fedfund[k][1])
            k=k+1
    l_fedfund_index.append(l_year)
l_fedfund_index[5].append(1.42)

#merging CPI
l_cpi_values=[]
for i in range(0,len(X_dataset_comp)):
    for j in range(0,len(X_dataset_comp[i])):
        r=int(X_dataset_comp[i][j][9])-2013
        c=int(X_dataset_comp[i][j][10])-1
        a=[]
        a.append(l_cpi_index[r][c])
        l_cpi_values.append(a)
    X_dataset_comp[i]=np.hstack((X_dataset_comp[i],l_cpi_values))
    l_cpi_values=[]
    
#merging federal fund rates
l_fedfund_values=[]
for i in range(0,len(X_dataset_comp)):
    for j in range(0,len(X_dataset_comp[i])):
        r=int(X_dataset_comp[i][j][9])-2013
        c=int(X_dataset_comp[i][j][10])-1
        a=[]
        a.append(l_fedfund_index[r][c])
        l_fedfund_values.append(a)
    X_dataset_comp[i]=np.hstack((X_dataset_comp[i],l_fedfund_values))
    l_fedfund_values=[]

#data preprocessing
from sklearn.preprocessing import StandardScaler,Imputer
im=Imputer(strategy='mean',axis=1)
ss=StandardScaler()

dict_comp={}
Y_dataset_comp=[]
for i in range(0,len(X_dataset_comp)):
    dict_comp[X_dataset_comp[i][0][6]]=i
    X_dataset_comp[i]=np.delete(X_dataset_comp[i],6,1)
    X_dataset_comp[i]=np.delete(X_dataset_comp[i],0,1)
    Y=X_dataset_comp[i][:,1]
    Y_dataset_comp.append(Y)

print('Enter the company name - ')
comp=input()
index=dict_comp[comp]      

X=X_dataset_comp[index]
Y=Y_dataset_comp[index]
X=im.fit_transform(X)
Y=Y.reshape(-1,1)
Y=im.fit_transform(Y)

#splitting data into training and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

#classifier part
from sklearn.ensemble import GradientBoostingRegressor
gbregressor=GradientBoostingRegressor(n_estimators=100,random_state=0)
gbregressor.fit(X_train,Y_train)
y_pred_gbr=gbregressor.predict(X_test)
y_pred_train_gbr=gbregressor.predict(X_train)
y_pred_train_gbr=y_pred_train_gbr.tolist()
    
acc_gbr=[]
for i in range(0,len(y_pred_gbr)):
    acc_gbr.append(abs(y_pred_gbr[i]-Y_test[i])/Y_test[i])
final_s_gbr=sum(acc_gbr)/len(acc_gbr)
        
acc_train_gbr=[]
for i in range(0,len(y_pred_train_gbr)):
    acc_train_gbr.append(abs(y_pred_train_gbr[i]-Y_train[i])/Y_train[i])
final_s_train_gbr=sum(acc_train_gbr)/len(acc_train_gbr)

from sklearn.ensemble import RandomForestRegressor
rfregressor=RandomForestRegressor(n_estimators=100,random_state=0)
rfregressor.fit(X_train,Y_train)
y_pred_rfr=rfregressor.predict(X_test)
y_pred_train_rfr=rfregressor.predict(X_train)
y_pred_train_rfr=y_pred_train_rfr.tolist()
    
acc_rfr=[]
for i in range(0,len(y_pred_rfr)):
    acc_rfr.append(abs(y_pred_rfr[i]-Y_test[i])/Y_test[i])
final_s_rfr=sum(acc_rfr)/len(acc_rfr)
        
acc_train_rfr=[]
for i in range(0,len(y_pred_train_rfr)):
    acc_train_rfr.append(abs(y_pred_train_rfr[i]-Y_train[i])/Y_train[i])
final_s_train_rfr=sum(acc_train_rfr)/len(acc_train_rfr)

from sklearn.svm import SVR
svmregressor=SVR(kernel='rbf')
svmregressor.fit(X_train,Y_train)
y_pred_svm=svmregressor.predict(X_test)
y_pred_train_svm=svmregressor.predict(X_train)
y_pred_train_svm=y_pred_train_svm.tolist()

acc_svm=[]
for i in range(0,len(y_pred_svm)):
        acc_svm.append(abs(y_pred_svm[i]-Y_test[i])/Y_test[i])
final_s_svm=sum(acc_svm)/len(acc_svm)

acc_train_svm=[]
for i in range(0,len(y_pred_train_svm)):
        acc_train_svm.append(abs(y_pred_train_svm[i]-Y_train[i])/Y_train[i])
final_s_train_svm=sum(acc_train_svm)/len(acc_train_svm)

#dataset of deviation of training data
l=[]
for i in range(0,len(acc_train_gbr)):
    l1=[]
    l1.append((-1)*acc_train_gbr[i])
    l1.append((-1)*acc_train_rfr[i])
    l1.append((-1)*acc_train_svm[i])
    l.append(l1)

#reinforcement learning
import math
N=1006
d=3
t=0
no_of_selections=[0]*d
score_arr=[0]*d
for i in range(0,N):
    max_bound=0
    ad=0
    for j in range(0,d):
        if no_of_selections[j]>0:
            a=score_arr[j]/no_of_selections[j]
            delta_j=math.sqrt(3/2*(math.log(i+1)/no_of_selections[j]))
            a=a+delta_j
        else:
            a=1e400
        if a>max_bound:
            max_bound=a
            ad=j
    no_of_selections[ad]=no_of_selections[ad]+1
    reward=l[i][ad]
    score_arr[ad]=score_arr[ad]+reward    

    #  ads_selected.append(ad)
print(no_of_selections)
w1=no_of_selections[0]/sum(no_of_selections)
w2=no_of_selections[1]/sum(no_of_selections)
w3=no_of_selections[2]/sum(no_of_selections)

final_pred=w1*rfregressor.predict(X_test)+w2*svmregressor.predict(X_test)+w3*gbregressor.predict(X_test)

acc_test=[]
for i in range(0,len(final_pred)):
    acc_test.append(abs(final_pred[i]-Y_test[i])/Y_test[i])
final_s__test_pred=sum(acc_test)/len(acc_test)

#result of each regressor - 
print("Accuracy of gradient boost regression - ",final_s_gbr)
print("Accuracy of random forest regression - ",final_s_rfr)
print("Accuracy of support vector machine - ",final_s_svm)
print("Accuracy of reinforcement learning - ",final_s__test_pred)

Y_final=Y

print("Correlation of stock opening prices and oil prices - ")
m1=np.mean(Y_final)*1258
m2=np.mean(list1_oil_price_add)*1258
pr=0
for i in range(0,1258):
    n2=Y_final[i].tolist()[0]    
    n3=list1_oil_price_add[i][0]        
    pr=pr+n2*n3
num=pr*1258-m1*m2
varx=0
vary=0
for i in range(0,1258):
    varx=varx+Y_final[i].tolist()[0]*Y_final[i].tolist()[0]
    vary=vary+list1_oil_price_add[i][0]*list1_oil_price_add[i][0]
varx=varx*(1258)
vary=vary*(1258)
varx=varx-m1*m1
vary=vary-m2*m2
x_dev=math.sqrt(varx)
y_dev=math.sqrt(vary)
num=num/(x_dev*y_dev)
print(num)