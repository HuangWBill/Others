# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from PyEMD import EMD, Visualisation  # 可视化

##载入时间序列数据
data = pd.read_csv(r'46086xiabannian.csv', usecols=[0])
S1 = data.values
S = S1[:, 0]

# print(len(S))
t = np.arange(0, len(S), 1)  # t 表示横轴的取值范围
# Extract imfs and residue
# In case of EMD
emd = EMD()
emd.emd(S)

# 获得分量+残余分量
imfs, res = emd.get_imfs_and_residue()

# 分量的个数
# print(len(imfs))

vis = Visualisation()
# 分量可视化
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
# 频率可视化
vis.plot_instant_freq(t=t, imfs=imfs)
vis.show()

# 保存分量+残余分量
for i in range(len(imfs)):
    a = imfs[i]
    dataframe = pd.DataFrame({'imf{}'.format(i + 1): a})
    dataframe.to_csv(r"imf-%d.csv" % (i + 1), index=False, sep=',')

# 保存残余分量
dataframe = pd.DataFrame(res)
dataframe.to_csv(r"res.csv", index=False, sep=',')
