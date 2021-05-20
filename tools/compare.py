import _init_paths
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率
from utils.misc import plot_performance
import numpy as np
import os
result_dir = './'
prefix = 'eval2D_results_MHP_HRNet_'

result_lst = os.listdir(result_dir)

marker_lst = ['o','^','s','*','x','+','d','X','D','.']
linestyle_lst = ['solid','dashed','dashdot','dotted']
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}
for result in result_lst:
    if prefix in result:
        if 'softmax_hmloss' in result or 'w32_softmax_pose2dloss' in result:continue
        print(result)
        mse2d_each_joint = np.loadtxt(os.path.join(result, 'mse2d_each_joint.txt'))
        PCK2d_lst = np.loadtxt(os.path.join(result, 'PCK2d.txt'))
        start,end = 0,30
        th2d_lst,PCK2d = PCK2d_lst[0,start:end], PCK2d_lst[1,start:end]
        # 2D pose mse
        print('2D pose MSE: {:.4f}'.format(mse2d_each_joint.mean()))

        # 2/3D pose PCK
        fig = plt.figure(1)
        label = os.path.basename(result).replace(prefix,'').replace('_v1','')
        plt.plot(th2d_lst, PCK2d, marker=marker_lst.pop(), linestyle=random.choice(linestyle_lst),label=label)
        plt.xlabel('Threshold [px]')
        plt.ylabel('2D PCK')

        # Area under the curve
        area = (PCK2d[0] + 2 * PCK2d[1:-1].sum() + PCK2d[-1])  * (th2d_lst[1] - th2d_lst[0]) / 2 / (th2d_lst[-1] - th2d_lst[0])
        print('2D PCK: {:.4f}'.format(area))
        plt.tight_layout()
    
    plt.legend(loc=4, ncol=1, borderaxespad=0., prop=font)
    plt.ylim([0.94,1])
    plt.title(prefix.split('_')[2])


# Stereo Hand比较：《2017 zimmermann》
PSO_thlst = np.array([20,25,30,35,40,45,50])
PSOPCK = np.array([0.32,0.56,0.68,0.74,0.79,0.85,0.89])
X26_thlst = np.linspace(21,50,12)
X26 = np.array([0.47, 0.54, 0.6,0.66,0.73,0.78,0.83,0.86,0.89,0.9,0.91,0.92])
CHPR_thlst = PSO_thlst
CHPR = np.array([0.57,0.71,0.81,0.88,0.9,0.92,0.95])
ICPPSO_thlst=PSO_thlst
ICPPSO = np.array([0.52,0.63,0.72,0.75,0.79,0.82,0.83])
Zimmer_thlst = X26_thlst
ZImmer = np.array([0.86,0.9,0.906,0.91,0.92,0.93,0.935,0.939,0.943,0.948,0.955,0.98])

# RHD比较：《2017 zimmermann》
PCK2d_lst_best = np.loadtxt(os.path.join('eval2D_results_RHD_HRNet_w32_trainable_softmax_pose2dloss_v1', 'PCK2d.txt'))
PCK2d_lst_worst = np.loadtxt(os.path.join('eval2D_results_RHD_HRNet_w32_max_hmloss_v1', 'PCK2d.txt'))

start,end = 0,30
th2d_lst, PCK2d_lst_best = PCK2d_lst_best[0,start:end], PCK2d_lst_best[1,start:end]
PCK2d_lst_worst = PCK2d_lst_worst[1,start:end]
th_lst = np.linspace(0,30,20)
PoseNet = np.array([0, 0.01,0.04,0.08,0.12,0.18,0.24,0.3,0.39,0.48,\
    0.54,0.59,0.62,0.65,0.7,0.72,0.75,0.77,0.79,0.80])
HandSegNet = np.array([0,0.005,0.02,0.06,0.1,0.15,0.21,0.26,0.32,0.39,\
    0.47,0.54,0.62,0.66,0.7,0.73,0.76,0.79,0.82,0.85])

plt.figure(4)
plt.subplot(1,2,1)
plt.plot(th2d_lst, PCK2d_lst_best, label='MyBest (AUC = 0.871)',linestyle='solid',marker='*')
plt.plot(th2d_lst, PCK2d_lst_worst, label='MyWorst (AUC = 0.845)',linestyle='solid',marker='.')
plt.plot(th_lst,PoseNet,label='PoseNet[30] (AUC = 0.724)',color='blue',linestyle='dashed',marker='s')
plt.plot(th_lst,HandSegNet,label='HandSegNet[30] (AUC = 0.663)',color='red',linestyle='dashdot',marker='o')
plt.xlabel('Threshold [px]')
plt.ylabel('2D PCK')
plt.legend(loc=4, ncol=1, borderaxespad=0., prop=font)
plt.title('RHD')

# MHP 2D PCKAUC 《WACV2021 Temporal Aware》 Not valid
PCK2d_lst_best = np.loadtxt(os.path.join('eval2D_results_MHP_HRNet_w32_trainable_softmax_pose2dloss_v1', 'PCK2d.txt'))
PCK2d_lst_worst = np.loadtxt(os.path.join('eval2D_results_MHP_HRNet_w32_max_hmloss_v1', 'PCK2d.txt'))
CPM_lst = np.loadtxt(os.path.join('eval2D_results_MHP_CPM_v1', 'PCK2d.txt'))
start,end = 0,30
th2d_lst, PCK2d_lst_best = PCK2d_lst_best[0,start:end], PCK2d_lst_best[1,start:end]
PCK2d_lst_worst = PCK2d_lst_worst[1,start:end]
CPM = CPM_lst[1,start:end]
plt.subplot(1,2,2)
plt.plot(th2d_lst, PCK2d_lst_best, label='MyBest (AUC=0.804)',linestyle='solid',marker='*')
plt.plot(th2d_lst, PCK2d_lst_worst, label='MyWorst (AUC=0.713)',linestyle='solid',marker='.')
plt.plot(th2d_lst,CPM,label='CPM (AUC=0.513)',color='blue',linestyle='dashed',marker='o')
plt.xlabel('Threshold [px]')
plt.ylabel('2D PCK')
plt.legend(loc=4, ncol=1, borderaxespad=0., prop=font)
plt.title('MHP')

# MHP 3D PCKAUC 《WACV2021 Temporal Aware》
PCK3d_lst = np.loadtxt(os.path.join('evaluation3D_results\eval3D_resultsVolTriangulation_MHP_v2_4views', 'PCK3d.txt')) # 20-50

start,end = 0, PCK3d_lst.shape[1]
th3d_lst,PCK3d = PCK3d_lst[0,start:end], PCK3d_lst[1,start:end]
area = (PCK3d[0] + 2 * PCK3d[1:-1].sum() + PCK3d[-1])  * (th3d_lst[1] - th3d_lst[0]) / 2 / (th3d_lst[-1] - th3d_lst[0])
# start,end = 0,30
# th3d_lst,PCK3d = PCK3d_lst[0,start:end], PCK3d_lst[1,start:end]
# area = (PCK3d[0] + 2 * PCK3d[1:-1].sum() + PCK3d[-1])  * (th3d_lst[1] - th3d_lst[0]) / 2 / (th3d_lst[-1] - th3d_lst[0])
# input(area)
th_lst = np.linspace(20,50,7)
Chen = np.array([0.825,0.89,0.93,0.95,0.975,0.98,0.985])
Cai = np.array([0.798,0.86,0.91,0.945,0.966,0.98,0.985])
TASSN = np.array([0.72, 0.813,0.87,0.91,0.94,0.96,0.975])

plt.figure(6)
plt.plot(th_lst,Chen,label='Chen et al.(AUC = 0.939)',color='green',linestyle='dashed',marker='s')
plt.plot(th_lst,Cai,label='Cai et al. (AUC = 0.928)',color='blue',linestyle='dashed',marker='v')
plt.plot(th_lst,TASSN,label='TASSN (AUC = 0.892)',color='red',linestyle='dashed',marker='o')
plt.plot(th3d_lst, PCK3d, label='Mine (AUC = {:.3f})'.format(area),linestyle='solid',marker='d',linewidth =2.0)

plt.xlabel('threshold [mm]')
plt.ylabel('3D PCK')
plt.legend(loc=4, ncol=1, borderaxespad=0., prop=font)
plt.title('在MHP数据集上与现有方法的对比')
plt.ylim([0.4,1.0])

# number of views
DLT_2views = {(1,2):(22.3264,0.8345), (1,3):(38.1175,0.7000), (1,4):(37.7532,0.5651), (2,3):(40.1288,0.6448), (2,4):(39.5667,0.5255), (3,4):(35.6219, 0.6516)}
DLT_3views = {(1,2,3): (21.6417,0.8736), (1,2,4):(30.2314,0.6845), (1,3,4):(28.2788,0.7544), (2,3,4):(32.2947,0.7006)}
DLT_4views = {(1,2,3,4): (26.1526,0.8036)}
l2, l3, l4 = len(DLT_2views), len(DLT_3views), len(DLT_4views)
DLT_EPE = np.array([sum([i[0] for i in DLT_2views.values()]) / l2, sum([i[0] for i in DLT_3views.values()]) / l3, sum([i[0] for i in DLT_4views.values()]) / l4])
DLT_PCK = np.array([sum([i[1] for i in DLT_2views.values()]) / l2, sum([i[1] for i in DLT_3views.values()]) / l3, sum([i[1] for i in DLT_4views.values()]) / l4])

RANSAC_2views = {(1,2):(21.4033, 0.8380), (1,3):(28.8372,0.7089), (1,4):(32.2637,0.5631), (2,3):(32.3255,0.6518), (2,4):(38.5398,0.5223), (3,4):(30.2430,0.6598)}
RANSAC_3views = {(1,2,3):(19.9156,0.8759), (1,2,4):(29.4164,0.6787), (1,3,4):(26.2652,0.7386), (2,3,4):(28.8931,0.6917)}
RANSAC_4views = {(1,2,3,4):(22.9570,0.8070)}
RANSAC_EPE = np.array([sum([i[0] for i in RANSAC_2views.values()]) / l2, sum([i[0] for i in RANSAC_3views.values()]) / l3, sum([i[0] for i in RANSAC_4views.values()]) / l4])
RANSAC_PCK = np.array([sum([i[1] for i in RANSAC_2views.values()]) / l2, sum([i[1] for i in RANSAC_3views.values()]) / l3, sum([i[1] for i in RANSAC_4views.values()]) / l4])

Vol_2views = {(1,2):(15.9557,0.9256), (1,3):(25.1398,0.7504), (1,4):(16.5884,0.9108), (2,3):(27.6085,0.7164), (2,4):(18.7685,0.8808), (3,4):(20.4065,0.8364)}
Vol_3views = {(1,2,3):(13.3625,0.9481), (1,2,4):(13.8154,0.9426), (1,3,4):(11.8769,0.9507), (2,3,4):(15.0871,0.9114)}
Vol_4views = {(1,2,3,4):(11.3530,0.9578)}
Vol_EPE = np.array([sum([i[0] for i in Vol_2views.values()]) / l2, sum([i[0] for i in Vol_3views.values()]) / l3, sum([i[0] for i in Vol_4views.values()]) / l4])
Vol_PCK = np.array([sum([i[1] for i in Vol_2views.values()]) / l2, sum([i[1] for i in Vol_3views.values()]) / l3, sum([i[1] for i in Vol_4views.values()]) / l4])

view_lst = np.array([2,3,4])

plt.figure()
print(DLT_EPE,DLT_PCK)
plt.plot(view_lst, DLT_EPE, label='DLT', color='green', marker='o')
print(RANSAC_EPE,RANSAC_PCK)
plt.plot(view_lst, RANSAC_EPE, label='RANSAC', color='blue', marker='^')
print(Vol_EPE,Vol_PCK)
plt.plot(view_lst, Vol_EPE, label='Volumetric', color='red', marker='D')
plt.xlabel('视角数')
plt.ylabel('末端误差EPE [mm]')
plt.xticks([2,3,4])
plt.ylim([0,40])
plt.title('末端误差随视角数的变化情况')
plt.legend(loc = 'best', prop=font)

plt.figure()
plt.plot(view_lst, DLT_PCK, label='DLT', color='green', marker='o')
plt.plot(view_lst, RANSAC_PCK, label='RANSAC', color='blue', marker='^')
plt.plot(view_lst, Vol_PCK, label='Volumetric', color='red', marker='D')
plt.xlabel('视角数')
plt.ylabel('3D PCK')
plt.xticks([2,3,4])
plt.ylim([0.5,1.0])
plt.title('PCK曲线下面积随视角数的变化情况')
plt.legend(loc = 'best', prop=font)

# random occlusion
radius_lst = np.array([0,10,30,50,70])
RANSAC_EPE = np.array([22.9570, 23.5161, 27.7543, 37.9286, 44.4879])
RANSAC_PCK = np.array([0.8070,0.7967,0.7222,0.5785,0.4525])
DLT_EPE = np.array([26.1526, 26.5264, 30.2176, 46.3534, 50.8730])
DLT_PCK = np.array([0.8036,0.7969,0.7266,0.5925,0.4989])
Vol_EPE = np.array([11.3530,11.4860,14.2956,18.4491,24.4509])
Vol_PCK = np.array([0.9578,0.9560,0.9256,0.8790,0.7868])
plt.figure()
plt.plot(radius_lst, DLT_EPE, label='DLT', color='green', marker='o')
plt.plot(radius_lst, RANSAC_EPE, label='RANSAC', color='blue', marker='^')
plt.plot(radius_lst, Vol_EPE, label='Volumetric', color='red', marker='D')
plt.xlabel('遮挡圆的半径 [px]')
plt.ylabel('末端误差EPE [mm]')
plt.ylim([0,60])
plt.legend(loc = 'best', prop=font)
plt.title('末端误差随遮挡程度的变化情况')

plt.figure()
plt.plot(radius_lst, DLT_PCK, label='DLT', color='green', marker='o')
plt.plot(radius_lst, RANSAC_PCK, label='RANSAC', color='blue', marker='^')
plt.plot(radius_lst, Vol_PCK, label='Volumetric', color='red', marker='D')
plt.xlabel('遮挡圆的半径 [px]')
plt.ylabel('3D PCK')
plt.legend(loc = 'best', prop=font)
plt.ylim([0,1.0])
plt.title('PCK曲线下面积随遮挡程度的变化情况')
plt.show()