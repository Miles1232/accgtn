import yaml
import h5py
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataloader import DataLoader

import data_utils
import space_angle_velocity
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
config = yaml.load(open('./config.yml'),Loader=yaml.FullLoader)

in_features=config['in_features']
out_features=config['out_features']
node_num = config['node_num']
input_dim=config['input_dim']
embed_size=config['embed_size']
num_layers = config['num_layers']
input_num = config['input_num']
output_T_dim = config['output_T_dim']
heads=config['heads']
dropout=config['dropout']
forward_expansion = config['forward_expansion']
base_path = config['base_dir_human36']


#load data
s80 = []
s160 = []
s320 = []
s400 = []
s560 = []
s720 = []
s1000 = []
dic = dict(
directions = np.array([1426, 374, 1087, 156, 1145, 332, 1438, 660])-2,
eating = np.array([1426, 374, 1087, 156, 1329, 955, 1145, 332])-2,
greeting = np.array([402, 1398, 63, 1180, 305, 955, 121, 332])-2,
sitting = np.array([1426, 1398, 1087, 1180, 1329, 955, 1145, 332])-2,
sittingdown= np.array([1426, 1398, 1087, 1180, 1145, 332, 1438, 1689])-2,
phoning = np.array([1426, 374, 1087, 156, 1329, 121, 332, 414])-2,
takingphoto = np.array([1426, 1087, 1180, 955, 1145, 332, 1438, 660])-2,
posing = np.array([402, 374, 63, 156, 835, 305, 955, 121])-2,
purchases= np.array([1087, 1180, 955, 1145, 332, 660, 304, 201])-2,
waiting = np.array([1426, 2068, 1398, 1087, 1180, 1145, 332, 1428])-2,
walkingdog = np.array([402, 374, 63, 156, 305, 121, 332, 414])-2,
walkingtogether= np.array([1087, 1180, 1329, 955, 1145, 337, 660, 304])-2,
discussion = np.array([1426, 2063, 1398, 1087, 1180, 1145, 332, 1438])-2,
walking= np.array([1087, 955, 1145, 332, 660, 304, 201, 142])-2,
smoking= np.array([1426, 1398, 1087, 1180, 1329, 955, 1145, 332])-2)

batch_size = 16
output_size = 25
output_n = 25
count  = 0

# for action in ["walking", "eating", "smoking", "discussion", "directions",
#                "greeting", "phoning", "posing", "purchases", "sitting",
#                "sittingdown", "waiting", "walkingdog","walkingtogether","takingphoto"]:
for action in [ "greeting",]:
    number = dic[action]
    print(number)
    for i in range (len(number)):
        #load data
        test_save_path = os.path.join(f'{base_path}{action}', f'{action}.npy')
        # test_save_path = './data/human36/new/Directions_1.npy'

        print(test_save_path)
        test_save_path = test_save_path.replace("\\","/")
        dataset = np.load(test_save_path,allow_pickle = True)
        # dataset = torch.tensor(dataset, dtype = torch.float32,requires_grad=False)
        dataset = dataset[number[i]:,:,:]
        print(dataset.shape)

        input_dataset = dataset[0:input_num] ##10 25 8
        input_dataset1 = dataset[1:input_num+1] ##10 25 8
        input_dataset2 = dataset[
                         input_num + output_T_dim + input_num - 1:input_num + output_T_dim - 1:-1].copy()
        input_dataset3 = dataset[input_num + output_T_dim + input_num:input_num + output_T_dim:-1]
        output_dataset = dataset[input_num:input_num+output_T_dim] ## 25 25 8
        v = dataset[input_num+output_T_dim:input_num+output_T_dim+1]
        input_dataset = torch.tensor( input_dataset, dtype = torch.float32,requires_grad=False)
        input_dataset1 = torch.tensor( input_dataset1, dtype = torch.float32,requires_grad=False)
        output_dataset = torch.tensor( output_dataset, dtype = torch.float32,requires_grad=False)
        input_dataset2 = torch.tensor(input_dataset2, dtype=torch.float32, requires_grad=False)
        v = torch.tensor(v, dtype=torch.float32, requires_grad=False)
        input_dataset = input_dataset.expand(batch_size,input_dataset.shape[0],input_dataset.shape[1],input_dataset.shape[2])
        input_dataset1 = input_dataset1.expand(batch_size,input_dataset1.shape[0],input_dataset1.shape[1],input_dataset1.shape[2])
        output_dataset = output_dataset.expand(batch_size,output_dataset.shape[0],output_dataset.shape[1],output_dataset.shape[2])
        input_dataset2 = input_dataset2.expand(batch_size, input_dataset2.shape[0], input_dataset2.shape[1],
                                               input_dataset2.shape[2])
        v = v.expand(batch_size, v.shape[0], v.shape[1],v.shape[2])
        # print(v.shape)

        input_dataset = input_dataset.to(device)
        input_dataset1 = input_dataset1.to(device)
        output_dataset = output_dataset.to(device)
        input_dataset2 = input_dataset2.to(device)
        v = v.to(device)

        total_samples = 0
        total_mse = 0
        total_mpjpe = 0
        # path = os.path.join(base_path,f'input{input_num}')
        path = './data/human36/model'
        model_x = torch.load(os.path.join(path, 'generator_x.pkl')).to(device)
        model_y = torch.load(os.path.join(path, 'generator_y.pkl')).to(device)
        model_z = torch.load(os.path.join(path, 'generator_z.pkl')).to(device)
        model_a = torch.load(os.path.join(path, 'generator_a.pkl')).to(device)

        model_a.eval()
        model_z.eval()
        model_x.eval()
        model_y.eval()


        input_angle = input_dataset[:, :, :, 3:6]
        input_angle1 = input_dataset2[:, :, :, 3:6]
        # input_angle = input_dataset[:, :, :, 0:3]
        # input_angle1 = input_dataset2[:, :, :, 0:3]
        input_acc_velocity = input_dataset[:, :, :, 7]
        input_acc_velocity1 = input_dataset2[:, :, :, 7]
        target_velocity = output_dataset[:,:,:,6]
        # print(input_acc_velocity)
        target_acc = output_dataset[:,:,:,7]
        # target_velocity2 = v[:,:,:,6]
        # target_velocity = torch.cat((target_velocity,target_velocity2),dim=1)
        # print(target_velocity.shape)

        target_angle = output_dataset[:, :, :, 3:6]
        # target_angle = output_dataset[:, :, :, 0:3]
        # target_angle2 = torch.cat((target_angle,v[:,:,:,3:6]),dim=1)
        # print(target_angle.shape)
        target_acc_velocity = output_dataset[:, :, :, 7]
        #read velocity
        input_acc_velocity = input_acc_velocity.float().unsqueeze(-1)
        input_acc_velocity1 = input_acc_velocity1.float().unsqueeze(-1)
        target_acc_velocity = target_acc_velocity.float()
        #read angle_x
        input_angle_x = input_angle[:,:,:,0].float().unsqueeze(-1)
        input_angle_x1 = input_angle1[:,:,:,0].float().unsqueeze(-1)
        target_angle_x = target_angle[:,:,:,0].float()
        #read angle_y
        input_angle_y = input_angle[:,:,:,1].float().unsqueeze(-1)
        input_angle_y1 = input_angle1[:,:,:,1].float().unsqueeze(-1)
        target_angle_y = target_angle[:,:,:,1].float()
        #read angle_z
        input_angle_z = input_angle[:,:,:,2].float().unsqueeze(-1)
        input_angle_z1 = input_angle1[:,:,:,2].float().unsqueeze(-1)
        target_angle_z = target_angle[:,:,:,2].float()
        #read 3D data
        input_3d_data = input_dataset[:, :, :, :3]
        input_3d_data1 = input_dataset2[:, :, :, :3]
        target_3d_data =output_dataset[:, :, :, :3]


        output_a = model_a(input_acc_velocity)
        # output_a = (output_a+output_a2)/2
        output_a = output_a.view(target_acc_velocity.shape[0],target_acc_velocity.shape[1],node_num)


        output_x = model_x(input_angle_x)
        # output_x = (output_x + output_x2) / 2
        output_x = output_x.view(target_angle_x.shape[0],target_angle_x.shape[1],node_num)


        output_y = model_y(input_angle_y)
        # output_y = (output_y + output_y2) / 2
        output_y = output_y.view(target_angle_y.shape[0],target_angle_y.shape[1],node_num)



        output_z = model_z(input_angle_z)
        # output_z = (output_z + output_z2) / 2
        output_z = output_z.view(target_angle_z.shape[0],target_angle_z.shape[1],node_num)




        angle_x = output_x
        angle_y = output_y
        angle_z = output_z
        pred_a = output_a   ## 16 25 25
        # pred_a1 = torch.cat((pred_a,v[:,:,:,7]),dim=1)


        pred_angle_set = torch.stack((angle_x,angle_y,angle_z),3)  ## 16 25 25 3

        pred_angle_set = pred_angle_set.reshape(pred_angle_set.shape[0],pred_angle_set.shape[1],-1,3)

        # pred_angle_set1 = torch.cat((pred_angle_set,v[:,:,:,3:6]),dim = 1)


        #acc-->velocity             16                              25          25
        input_vel = torch.zeros((target_acc_velocity.shape[0], output_n, input_3d_data.shape[-2],1)).to(device)
        for x in range (input_vel.shape[0]):
            input_vel[x,0,:,0] = input_dataset[x, input_num-1,:,6]
        pred_v = torch.FloatTensor([])
        for y in range (input_vel.shape[0]):
            for z in range (input_vel.shape[1]):
                re_vel = space_angle_velocity.reconstruction_velocity(pred_a[y,z,:],input_vel[y,z,:,0],node_num)
                # re_vel = space_angle_velocity.reconstruction_velocity(target_acc_velocity[y, z, :], input_vel[y, z, :, 0], node_num)
                pred_v = torch.cat([pred_v,re_vel],dim=0)
                re_vel = re_vel
                if z+1<input_vel.shape[1]:
                    input_vel[y,z+1,:,:] = re_vel
                else:
                    continue
        pred_v = pred_v.view(input_vel.shape[0],-1,25)
        
        
        #reconstruction_loss
        input_pose = torch.zeros((target_acc_velocity.shape[0], output_n, input_3d_data.shape[-2], input_3d_data.shape[-1]))
        for a in range(input_pose.shape[0]):
            input_pose[a,0,:,:] = input_3d_data[a,input_num-1,:,:]
        re_data = torch.FloatTensor([])
        # re_data2 = torch.FloatTensor([])
        for b in range (target_3d_data.shape[0]):
            for c in range (target_3d_data.shape[1]):
                reconstruction_coordinate = space_angle_velocity.reconstruction_motion(pred_v[b,c,:], pred_angle_set[b, c,:,:], input_pose[b, c, :, :],node_num)
                # reconstruction_coordinate = space_angle_velocity.reconstruction_motion(target_velocity[b, c, :],
                #                                                                        target_angle[b, c, :, :],
                #                                                                        input_pose[b, c, :, :], node_num)

                re_data = torch.cat([re_data,reconstruction_coordinate],dim=0)
                # re_data2 = torch.cat([re_data2,reconstruction_coordinate2],dim=0)
                reconstruction_coordinate = reconstruction_coordinate
                # reconstruction_coordinate2 = reconstruction_coordinate2
                if c+1<target_3d_data.shape[1]:
                    input_pose[b,c+1,:,:] = reconstruction_coordinate
                    # input_pose2[b,c+1,:,:] = reconstruction_coordinate2
                else:
                    continue
        # print(re_data.shape)
        re_data = re_data.view(target_3d_data.shape[0],-1,node_num,3)
       
        #
        frame_re_data = re_data[0]

        # frame_re_data = torch.from_numpy(frame_re_data)
        frame_target_3d_data = target_3d_data[0]
        mpjpe_set = []
        for i in range (frame_re_data.shape[0]):
            frame_re_data = frame_re_data.to(device)
            # frame_re_data2 = frame_re_data2.to(device)
            frame_target_3d_data = frame_target_3d_data.to(device)
            frame_rec_loss = torch.mean(torch.norm(frame_re_data[i] - frame_target_3d_data[i], 2, 1))
            frame_rec_loss = frame_rec_loss.cpu()
            mpjpe_set.append(frame_rec_loss)

        #save vis data
        frame_target_3d_data = frame_target_3d_data.cpu()
        # frame_re_data2 = np.array(frame_re_data2.cpu())
        frame_re_data = np.array(frame_re_data.cpu())
        # frame_re_data = frame_re_data.cpu()
        frame_target_3d_data = np.array(frame_target_3d_data[0])
        mpjpe_set = np.array(mpjpe_set)
        vis_save_path = os.path.join(f'{base_path}{action}', 'vis.npy')
        # vis_save_path2 = os.path.join(f'{base_path}{action}', 'vis2.npy')
        vis_mpjpe_save_path = os.path.join(f'{base_path}{action}', 'vis_mpjpe.npy')
        np.save(vis_save_path, frame_re_data)
        # np.save(vis_save_path2, frame_re_data)
        np.save(vis_mpjpe_save_path, mpjpe_set)
        print ('-------------------')
        print ('mpjpe_set',mpjpe_set)
        # print('mp_shape',mpjpe_set.shape)
        print ('frame_re_data.shape:\n',frame_re_data.shape)

        print('80ms:\n',mpjpe_set[1])
        print('160ms:\n',mpjpe_set[3])
        print('320ms:\n',mpjpe_set[7])
        print('400ms:\n',mpjpe_set[9])
        print('560ms:\n',mpjpe_set[13])
        print('720ms:\n',mpjpe_set[17])
        print('1000ms:\n',mpjpe_set[24])
        s80.append(mpjpe_set[1])
        s160.append(mpjpe_set[3])
        s320.append(mpjpe_set[7])
        s400.append(mpjpe_set[9])
        s560.append(mpjpe_set[13])
        s720.append(mpjpe_set[17])
        s1000.append(mpjpe_set[24])
    data = []
    for index, loss in enumerate([s80, s160, s320, s400, s560, s720, s1000]):
        if index == 0:
            data.append(sum(loss[count:count + 8]) / 8)
            print(f'mean_s80 of {action} = ', sum(loss[count:count + 8]) / 8)
        elif index == 1:
            data.append(sum(loss[count:count + 8]) / 8)
            print(f'mean_s160 of {action} = ', sum(loss[count:count + 8]) / 8)
        elif index == 2:
            data.append(sum(loss[count:count + 8]) / 8)
            print(f'mean_s320 of {action} = ', sum(loss[count:count + 8]) / 8)
        elif index == 3:
            data.append(sum(loss[count:count + 8]) / 8)
            print(f'mean_s400 of {action} = ', sum(loss[count:count + 8]) / 8)
        elif index == 4:
            data.append(sum(loss[count:count + 8]) / 8)
            print(f'mean_s560 of {action} = ', sum(loss[count:count + 8]) / 8)
        elif index == 5:
            data.append(sum(loss[count:count + 8]) / 8)
            print(f'mean_s720 of {action} = ', sum(loss[count:count + 8]) / 8)
        elif index == 6:
            data.append(sum(loss[count:count + 8]) / 8)
            print(f'mean_s1000 of {action} = ', sum(loss[count:count + 8]) / 8)
    count += 8

    header = [f'{action}80ms', '160', '320', '400','560','720','1000']


    with open('loss.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerow(data)
print(sum(s80)/len(s80),sum(s160)/len(s160),sum(s320)/len(s320),sum(s400)/len(s400),sum(s560)/len(s560),sum(s720)/len(s720),sum(s1000)/len(s1000))




























