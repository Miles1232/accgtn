import time
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class plot_h36m(object):

    def __init__(self, prediction_data, GT_data):
        print ('prediction_data',type(prediction_data))
        print ('GT_data',type(GT_data))
        self.joint_xyz = GT_data
        self.nframes = 25

        self.joint_xyz_f = prediction_data

        # set up the axes
        xmin = -800
        xmax = 800
        ymin = -800
        ymax = 800
        zmin = -800
        zmax = 800
        # xmax = np.max(GT_data[:, :, 0]) + 50
        # xmin = np.min(GT_data[:, :, 0]) - 50
        # ymax = np.max(GT_data[:, :, 1]) + 100
        # ymin = np.min(GT_data[:, :, 1]) - 100
        # zmax = np.max(GT_data[:, :, 2])
        # zmin = np.min(GT_data[:, :, 2])
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.view_init(elev=100, azim=-90)
        # plt.axis('off')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        # self.ax.set_title(actions,loc='left')
        # self.ax.set_title("frame:{}".format(i))
        self.chain = [np.array([0, 1, 2, 3, ]),
                      np.array([0, 4, 5, 6]),
                      np.array([0, 7, 8, 9, 10]),
                      np.array([8, 11, 12, 13]),
                      np.array([8, 14, 15, 16])]
        # self.chain = [np.array([0, 1, 2, 3, 4,5]),
        #               np.array([0, 6,7,8,9,10]),
        #               np.array([0, 11,12,13,14]),
        #               np.array([12,15,16,17,19,17,18]),
        #               np.array([12,20,21,22,23,22,24])]
        print (type(self.chain))
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)
        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])
        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='blue')) # red: prediction
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='k')) # blue: ground truth
        # print(self.lns)
    def plot(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=True)
        # ani.save(f'{base_path}/vis/{actions}/{actions}.gif',writer='pillow')
        ani.save(f'{base_path}/{actions}.gif', writer='pillow')
        plt.show()
        
if __name__ == '__main__':
    config = yaml.load(open('config.yml'),Loader=yaml.FullLoader)
    use_node = np.array([0,1,2,3,6,7,8,11,12,13,14,15,16,17,20,21,22])
    # print(len(use_node))
    dic = dict(
        directions=np.array([1426, 374, 1087, 156, 1145, 332, 1438, 660]) - 2,
        eating=np.array([1426, 374, 1087, 156, 1329, 955, 1145, 332]) - 2,
        greeting=np.array([402, 1398, 63, 1180, 305, 955, 121, 332]) - 2,
        sitting=np.array([1426, 1398, 1087, 1180, 1329, 955, 1145, 332]) - 2,
        sittingdown=np.array([1426, 1398, 1087, 1180, 1145, 332, 1438, 1689]) - 2,
        phoning=np.array([1426, 374, 1087, 156, 1329, 121, 332, 374]) - 2,
        takingphoto=np.array([1426, 1087, 1180, 955, 1145, 332, 1438, 660]) - 2,
        posing=np.array([402, 374, 63, 156, 835, 305, 955, 402]) - 2,
        purchases=np.array([1087, 1180, 955, 1145, 332, 660, 304, 201]) - 2,
        waiting=np.array([1426, 2068, 1398, 1087, 1180, 1145, 332, 1428]) - 2,
        walkingdog=np.array([402, 374, 63, 156, 305, 121, 332, 414]) - 2,
        walkingtogether=np.array([1087, 1180, 1329, 955, 1145, 337, 660, 304]) - 2,
        discussion=np.array([1426, 2063, 1398, 1087, 1180, 1145, 332, 1438]) - 2,
        walking=np.array([1087, 955, 1145, 332, 660, 304, 201, 142]) - 2,
        smoking=np.array([1426, 1398, 1087, 1180, 1329, 955, 1145, 332]) - 2)
    #load GT_data
    base_path = config['base_dir_human36']
    # for actions in ["walking", "eating", "smoking", "discussion", "directions",
    #            "greeting", "phoning", "posing", "purchases", "sitting",
    #            "sittingdown", "takingphoto", "waiting", "walkingdog","walkingtogether"]:
    for actions in ["eating"]:
        test_save_path = os.path.join(f'{base_path}{actions}', f'{actions}.npy')
        GT_data = np.load(test_save_path)

        # load prediction_data
        prediction_data_path = os.path.join(f'{base_path}{actions}', 'vis.npy')
        # prediction_data_path = os.path.join(f'./data/coordinate', 'vis.npy')
        prediction_data = np.load(prediction_data_path)


        print('prediction_data:\n',prediction_data.shape)
        print('GT_data:\n',GT_data.shape)

        # nframes = prediction_data.shape[0]
        prediction_data = prediction_data[:,use_node,:]
        GT_data = GT_data[dic[actions][-1]+20:dic[actions][-1]+45,use_node,:]
        # GT_data = GT_data[77:,use_node,:]

        # predict_plot = plot_h36m(GT_data ,GT_data)
        predict_plot = plot_h36m(prediction_data, GT_data)
        predict_plot.plot()
        # s = 0
        # for i in range(1, 26):
        #     predict_plot = plot_h36m(prediction_data[s:i], GT_data[s:i])
        #     predict_plot.plot()
        #     s += 1
    
    
    


























# fig = plt.figure()
# ax = plt.axes(xlim=(-800, 800), ylim=(-800,800), zlim=(-800,800), projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
#
# chain = [np.array([0, 1, 2, 3]),
#          np.array([0, 4, 5, 6]),
#          np.array([0, 7, 8, 9, 10]),
#          np.array([7, 11, 12, 13]),
#          np.array([7, 14, 15, 16])]
#
# scats = []
# lns = []
# filename = filename
#
# def update(frame):
#     for scat in scats:
#         scat.remove()
#     for ln in lns:
#         ax.lines.pop(0)
#
#     scats = []
#     lns = []
#
#     xdata = np.squeeze(GT_data[frame, :, 0])
#     ydata = np.squeeze(GT_data[frame, :, 1])
#     zdata = np.squeeze(GT_data[frame, :, 2])
#
#     xdata_f = np.squeeze(prediction_data[frame, :, 0])
#     ydata_f = np.squeeze(prediction_data[frame, :, 1])
#     zdata_f = np.squeeze(prediction_data[frame, :, 2])
#
#     for i in range(len(chain)):
#         lns.append(ax.plot3D(xdata_f[chain[i][:],], ydata_f[chain[i][:],], zdata_f[chain[i][:],], linewidth=2.0, color='#f94e3e')) # red: prediction
#         lns.append(ax.plot3D(xdata[chain[i][:],], ydata[chain[i][:],], zdata[chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth
#
# ani = FuncAnimation(fig, update, frames=nframes, interval=40, repeat=False)
# plt.title(filename, fontsize=16)
#
# plt.show()



