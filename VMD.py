# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft

path='I:\滑坡位移量\白水河滑坡_ZG118'
# path='C:/Users\dell\Desktop\cyq'
# filename = 'I:\滑坡位移量\白水河滑坡_ZG118\ZG118_累积位移.csv'
filename=path+'/'+'ZG118_累积位移.csv'
# filename='C:/Users\dell\Desktop\cyq\cyq.csv'
df = pd.read_csv(filename,header=0,encoding="gbk")
df=df.values
weiyi=df[:,6:7]
date=df[:,0:1]
alpha = 3 # moderate bandwidth constraint
tau = 0.  # noise-tolerance (no strict fidelity enforcement)
K = 3  # 3 modes
DC = 1  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-6

"""  
alpha、tau、K、DC、init、tol 六个输入参数的无严格要求； 
alpha 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍； 
tau 噪声容限 ；
K 分解模态（IMF）个数； 
DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1； 
init 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数； 
tol 控制误差大小常量，决定精度与迭代次数
"""

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 400  # 图片像素
plt.rcParams['figure.dpi'] = 100  # 分辨率

u, u_hat, omega = VMD(weiyi, alpha, tau, K, DC, init, tol)
plt.figure()
plt.plot(u.T)
plt.plot(weiyi)
plt.xlim(-0.5, date.shape[0]-0.5)
plt.xticks(range(0,date.shape[0]),date.flatten(),rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('日期',fontsize=15)
plt.ylabel('VMD分解后各分量',fontsize=15)
plt.show()

plt.figure(figsize=(7, 7))
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(u[i, :], linewidth=1.5, c='r')
    plt.xticks([])
    plt.xlim(-0.5, date.shape[0] - 0.5)
    plt.ylabel('IMF{}'.format(i + 1),fontsize=15)
    plt.yticks(fontsize=15)
plt.xlim(-0.5, date.shape[0]-0.5)
plt.xticks(range(0,date.shape[0]),date.flatten(),rotation=90,fontsize=15)
plt.xlabel('日期',fontsize=15)
plt.show()

# 保存子序列数据到文件中
for i in range(K):
    a = u[i, :]
    dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
    dataframe.to_csv(path+'/'+"VMD_result_%d.csv" % (i + 1), index=False, sep=',')

zong=u[0, :]+u[1, :]+u[2, :]
plt.figure()
plt.plot(weiyi)
plt.plot(zong)
plt.show()


# plt.figure(figsize=(7, 7), dpi=200)
# # 中心模态
# for i in range(K):
#     plt.subplot(K, 1, i + 1)
#     plt.plot(abs(fft(u[i, :])))
#     plt.ylabel('IMF{}'.format(i + 1))
# plt.show()

from sklearn.cluster import KMeans
# 数据读取
result_path='I:\滑坡位移量\白水河滑坡_ZG118\ZG118_累积位移.csv'
# result_path='I:\滑坡位移量\三峡新浦20210816(黄观文)\XP02\XP02预警数据.csv'
# result_path='C:/Users\dell\Desktop\cyq\cyq.csv'
df = pd.read_csv(result_path,header=0,encoding="gbk")
df=df.values
acc_displacement=df[:,6:7]
date=df[:,0:1]

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 400  # 图片像素
plt.rcParams['figure.dpi'] = 100  # 分辨率
# 计算日位移
acc_displacement1=np.zeros((acc_displacement.shape[0],acc_displacement.shape[1]))
acc_displacement1[0:1]=acc_displacement[0:1]
acc_displacement1[1:acc_displacement.shape[0]]=acc_displacement[0:acc_displacement.shape[0]-1]
daily_displacement=abs(acc_displacement1-acc_displacement)
plt.plot(acc_displacement,label='累积位移量')
plt.plot(daily_displacement,label='逐月位移量变化量')
plt.ylabel('位移量（mm）',fontsize=15)
plt.xlabel('日期',fontsize=15)
plt.legend(fontsize=25)
plt.xlim(-0.5, date.shape[0]-0.5)
plt.xticks(range(0,date.shape[0]),date.flatten(),rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.show()
daily_displacement=daily_displacement[1:,:]
daily_displacement=(daily_displacement-min(daily_displacement))/(max(daily_displacement)-min(daily_displacement))
plt.plot(daily_displacement,label='逐月位移量变化量')
plt.ylabel('位移量（mm）',fontsize=15)
plt.xlabel('日期',fontsize=15)
plt.legend(fontsize=25)
plt.xlim(-0.5, date.shape[0]-0.5)
plt.xticks(range(0,date.shape[0]),date.flatten(),rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.show()

#肘方法看k值
d=[]
for i in range(1,11):    #k取值1~11，做kmeans聚类，看不同k值对应的簇内误差平方和
    km=KMeans(n_clusters=i,init='k-means++',random_state=0)
    km.fit(u[2, :].reshape(-1,1))
    d.append(km.inertia_)  #inertia簇内误差平方和
plt.plot(range(1,11),d,marker='o')
plt.xlabel('聚类个数',fontsize=15)
plt.ylabel('簇内误差平方和',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

clusters_num=3
km=KMeans(n_clusters=clusters_num,init='k-means++',random_state=0)
km.fit(u[2, :].reshape(-1,1))
print('簇内误差平方和:',km.inertia_)  #inertia簇内误差平方和
centers=km.cluster_centers_   #类别中心
print('聚类中心：',centers)
y_pre=km.predict(u[2, :].reshape(-1,1))
print('聚类结果：',y_pre)
colors = ['b', 'r', 'c']
plt.plot(acc_displacement,c='g',label='累积位移量')
plt.plot(u[2, :].reshape(-1,1),c='orange',label='第三分量')
for j in range(clusters_num):
    index_set = np.where(y_pre == j)
    # print(index_set)
    daily_cluster = u[2, :].reshape(-1,1)[index_set]
    # print(daily_cluster)
    acc_cluster = acc_displacement[index_set]
    # print(acc_cluster)
    plt.scatter(index_set, daily_cluster, c=colors[j], marker='.')
    plt.scatter(index_set, acc_cluster, c=colors[j], marker='.')
    plt.vlines(x=index_set, ymin=daily_cluster, ymax=acc_cluster, linestyles='dashed',colors=colors[j])
# for i in range(acc_displacement.shape[0]):
#     plt.vlines(x=i, ymin=daily_displacement[i], ymax=acc_displacement[i], linestyles='dashed')
plt.legend(fontsize=25)
plt.xlim(-0.5, date.shape[0]-0.5)
plt.xticks(range(0,date.shape[0]),date.flatten(),rotation=90,fontsize=15)
plt.xlabel('日期',fontsize=15)
plt.ylabel('位移量（mm）',fontsize=15)
plt.show()


