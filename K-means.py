# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
from matplotlib.ticker import MultipleLocator
'''
不能直接进行聚类，可以尝试转换成速率进行聚类
'''
# 数据读取
# result_path='G:\滑坡位移量\白水河滑坡_ZG118\ZG118_累积位移.csv'
# result_path='G:\滑坡位移量\三峡新浦20210816(黄观文)\XP02\XP02预警数据.csv'
result_path='G:\滑坡位移量\白水河滑坡_ZG118/new\XD01_use.csv'
data = pd.read_csv(result_path,header=0,encoding="gbk")
print(data)
# data=data.values
print(data.shape)
acc_displacement=data['displacement']
#.values.reshape(data.shape[0],1)
date = pd.to_datetime(data["date"], format='%Y/%m')
daily=data['daily']
daily_norm=data['daily_norm']
tanA=data['tanA']
tanA_norm=data['tanA_norm']

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 400  # 图片像素
plt.rcParams['figure.dpi'] = 100  # 分辨率

# 绘制累积位移，逐日位移，tanA
host = host_subplot(111, axes_class=axisartist.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(50, 0))

par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)

p1, = host.plot(date,acc_displacement, label="累积位移D(mm)",marker='o')
p2, = par1.plot(date,daily, label="月位移增量ΔD(mm)")
p3, = par2.plot(date,tanA, label="月位移切线角(°)")

host.set_xlabel("日期")
host.set_ylabel("累积位移D(mm)")
par1.set_ylabel("月位移增量ΔD(mm)")
par2.set_ylabel("月位移切线角(°)")

host.legend()

host.axis["left"].label.set_color(p1.get_color())
host.axis["left"].line.set_color(p1.get_color())
host.axis["left"].major_ticks.set_color(p1.get_color())
host.axis["left"].major_ticklabels.set_color(p1.get_color())

par1.axis["right"].label.set_color(p2.get_color())
par1.axis["right"].line.set_color(p2.get_color())
par1.axis["right"].major_ticks.set_color(p2.get_color())
par1.axis["right"].major_ticklabels.set_color(p2.get_color())

par2.axis["right"].label.set_color(p3.get_color())
par2.axis["right"].line.set_color(p3.get_color())
par2.axis["right"].major_ticks.set_color(p3.get_color())
par2.axis["right"].major_ticklabels.set_color(p3.get_color())

host.set_xlim(min(date), max(date))
host.set_ylim(-100, max(acc_displacement)+200)
par1.set_ylim(min(daily)-150, max(daily)+500)
par2.set_ylim(min(tanA)-0.005, max(tanA)+0.2)

# host.set_xticklabels(labels=date.flatten(), rotation=90)
# plt.setp(host.axis["bottom"].major_ticklabels, rotation=90)
# plt.xlim(-0.5, date.shape[0]-0.5)
# plt.xticks(range(0,date.shape[0]),date.tolist(),rotation=90,fontsize=15)
plt.tight_layout()
tick_spacing = 100  # 通过修改tick_spacing的值可以修改x轴的密度
host.xaxis.set_major_locator(MultipleLocator(tick_spacing))
plt.show()



displacement=data[['daily_norm','tanA_norm']].values
#肘方法看k值
d=[]
for i in range(1,11):    #k取值1~11，做kmeans聚类，看不同k值对应的簇内误差平方和
    km=KMeans(n_clusters=i)
    km.fit(displacement)
    d.append(km.inertia_)  #inertia簇内误差平方和
plt.plot(range(1,11),d,marker='o')
plt.xlabel('聚类个数',fontsize=15)
plt.ylabel('簇内误差平方和',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


clusters_num=2
km=KMeans(n_clusters=clusters_num)
km.fit(displacement)
print('簇内误差平方和:',km.inertia_)  #inertia簇内误差平方和
centers=km.cluster_centers_   #类别中心
print('聚类中心：',centers)
y_pre=km.predict(displacement)
print('聚类结果：',y_pre)
colors = ['b', 'r', 'c']
plt.figure()
plt.plot(acc_displacement,c='g',label='累积位移量')
plt.plot(daily_norm*800,c='orange',label='逐日位移量')
plt.plot(tanA_norm*800,c='black',label='a')
for j in range(clusters_num):
    index_set = np.where(y_pre == j)
    # print(index_set)
    daily_cluster = daily_norm.values[index_set]
    a_cluster = tanA_norm.values[index_set]
    # print(daily_cluster)
    acc_cluster = acc_displacement.values[index_set]
    # print(acc_cluster)
    plt.scatter(index_set, daily_cluster, c=colors[j], marker='.')
    plt.scatter(index_set, a_cluster, c=colors[j], marker='.')
    plt.scatter(index_set, acc_cluster, c=colors[j], marker='.')
    plt.vlines(x=index_set, ymin=daily_cluster, ymax=acc_cluster, linestyles='dashed',colors=colors[j])
    plt.vlines(x=index_set, ymin=a_cluster, ymax=acc_cluster, linestyles='dashed', colors=colors[j])
# for i in range(acc_displacement.shape[0]):
#     plt.vlines(x=i, ymin=daily_displacement[i], ymax=acc_displacement[i], linestyles='dashed')
plt.legend(fontsize=15)
plt.xlim(-0.5, date.shape[0]-0.5)
plt.xticks(range(0,date.shape[0]),date,rotation=90,fontsize=15)
plt.xlabel('日期',fontsize=15)
plt.ylabel('位移量（mm）',fontsize=15)
plt.show()



plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 400  # 图片像素
plt.rcParams['figure.dpi'] = 150  # 分辨率
# 绘制累积位移，逐日位移，tanA
host = host_subplot(111, axes_class=axisartist.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(50, 0))

par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)

p1, = host.plot(date,acc_displacement, label="累积位移D(mm)",marker='o')
p2, = par1.plot(date,daily, label="月位移增量ΔD(mm)")
p3, = par2.plot(date,tanA, label="月位移切线角(°)")

host.set_xlabel("日期",fontsize=15)
host.set_ylabel("累积位移D(mm)",fontsize=15)
par1.set_ylabel("月位移增量ΔD(mm)",fontsize=15)
par2.set_ylabel("月位移切线角(°)",fontsize=15)

host.legend()

host.axis["left"].label.set_color(p1.get_color())
host.axis["left"].line.set_color(p1.get_color())
host.axis["left"].major_ticks.set_color(p1.get_color())
host.axis["left"].major_ticklabels.set_color(p1.get_color())

par1.axis["right"].label.set_color(p2.get_color())
par1.axis["right"].line.set_color(p2.get_color())
par1.axis["right"].major_ticks.set_color(p2.get_color())
par1.axis["right"].major_ticklabels.set_color(p2.get_color())

par2.axis["right"].label.set_color(p3.get_color())
par2.axis["right"].line.set_color(p3.get_color())
par2.axis["right"].major_ticks.set_color(p3.get_color())
par2.axis["right"].major_ticklabels.set_color(p3.get_color())

host.set_xlim(min(date), max(date))
host.set_ylim(-100, max(acc_displacement)+200)
par1.set_ylim(min(daily)-150, max(daily)+500)
par2.set_ylim(min(tanA)-0.005, max(tanA)+0.2)

span_start=[3,18,28,40,52,64]
span_end=[7,20,30,44,54,66]
for k in range(len(span_start)):
    host.axvspan(date[span_start[k]], date[span_end[k]], facecolor='gray', alpha=0.25)

# host.set_xticklabels(labels=date.flatten(), rotation=90)
# plt.setp(host.axis["bottom"].major_ticklabels, rotation=90)
# plt.xlim(-0.5, date.shape[0]-0.5)
# plt.xticks(range(0,date.shape[0]),date.tolist(),rotation=90,fontsize=15)
plt.tight_layout()
tick_spacing = 150  # 通过修改tick_spacing的值可以修改x轴的密度
host.xaxis.set_major_locator(MultipleLocator(tick_spacing))
plt.show()








