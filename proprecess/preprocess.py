import yaml
import os
import numpy as np
import h5py
import math 
import torch
import torch.nn.functional as F

config = yaml.load(open('config.yml'),Loader=yaml.FullLoader)

# build path
base_path = 'D:/Fourth_motion_prediction_paper/prediction_transformer_model/data'
move_joint = np.array([0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30])
print(len(move_joint))
train_dataset = []
train_data_path = open(r'D:/Fourth_motion_prediction_paper/prediction_transformer_model/proprecess/train_no_sittingdown.txt')
train_save_path = os.path.join(base_path, 'train.npy')
train_save_path = train_save_path.replace("\\","/")

def angle(v1):
    x =  torch.FloatTensor([1, 0, 0])
    y =  torch.FloatTensor([0, 1, 0])
    z =  torch.FloatTensor([0, 0, 1])

    L =  torch.sqrt(v1.dot(v1))
    Lx = torch.sqrt(x.dot(x))
    Ly = torch.sqrt(y.dot(y))
    Lz = torch.sqrt(z.dot(z))
    cos_angle_x=v1.dot(x)/(L*Lx)

    cos_angle_y=v1.dot(y)/(L*Ly)

    cos_angle_z=v1.dot(z)/(L*Lz)
    return cos_angle_x, cos_angle_y, cos_angle_z

def space_angle(previous_one_frame, current_frame):

    space_angle = torch.FloatTensor([])


    A = current_frame- previous_one_frame
    angle_x, angle_y, angle_z = angle(A)
    one_joint_space_angle = torch.tensor([angle_x, angle_y, angle_z],dtype = torch.float32)
    space_angle = torch.cat((space_angle, one_joint_space_angle), dim=0)
    space_angle = space_angle.view(1, previous_one_frame.shape[0])

    return space_angle

def space_distance(previous_one_frame, current_frame):
    space_velicity = torch.FloatTensor([])

    dist = torch.sqrt(torch.sum((current_frame-previous_one_frame)**2))
    dist = torch.unsqueeze(dist, 0)
    space_velicity = torch.cat((space_velicity, dist), dim=0)
    space_velicity = space_velicity.view(1, 1)

    return space_velicity

def loc_exchange(input):
    fr = input.shape[0]

    input = input.reshape(fr, -1, 3)
    nd = input.shape[1] 
    input = torch.tensor(input, dtype = torch.float32)

    angle_velocity = torch.FloatTensor([])
    one_sequence = torch.FloatTensor([])
    for a in range (input.shape[0]-1):
        one_frame = torch.FloatTensor([])
        for b in range (input.shape[1]):
            space_angles = space_angle(input[a,b],input[a+1,b])
            space_velocity = space_distance(input[a,b],input[a+1,b])
            space_angles = torch.where(torch.isnan(space_angles), torch.full_like(space_angles, 0), space_angles)
            one_frame = torch.cat([one_frame,space_angles,space_velocity],dim=1)
        one_sequence = torch.cat([one_sequence,one_frame],dim=0)
    angle_velocity = torch.cat([angle_velocity,one_sequence],dim=0)
    # print('angle_velocity:\n',angle_velocity.shape)
    angle_velocity = angle_velocity.view(fr-1, nd, 4)
    return angle_velocity
    
    
def acc_calculate(input):
    fr = input.shape[0]
    input = input.reshape(fr, -1, 7)
    nd = input.shape[1]
    input = torch.tensor(input, dtype = torch.float32)
    # print('acc_input:\n',input.shape)
    acc_velocity = torch.FloatTensor([])
    for x in range (fr-1):
        acc_one_frame = torch.FloatTensor([])
        for y in range (nd):
            acc_one = input[x+1,y, -1] - input[x,y, -1]
            acc_one = torch.unsqueeze(acc_one,0)
            acc_one_frame = torch.cat([acc_one_frame,acc_one], dim=0)
        acc_one_frame = torch.unsqueeze(acc_one_frame,0)
        acc_velocity = torch.cat([acc_velocity,acc_one_frame], dim=0)
    # print('acc_one_frame:\n',acc_one_frame.shape)
    return acc_velocity
    # print('acc_velocity:\n',acc_velocity.shape)
    # exit()
    
'''

for train_one_data_path in train_data_path:
    
    keyword = 'Sitting'
    if keyword in train_one_data_path:
        print (train_one_data_path)
        train_one_data_path = train_one_data_path.strip('\n')
        # load train data
        train_data = h5py.File(train_one_data_path,'r')
        coordinate_normalize_joint = train_data['coordinate_normalize_joint'][:,move_joint,:] 
        train_num = int(coordinate_normalize_joint.shape[0])
        print('coordinate_normalize_joint.shape:\n',coordinate_normalize_joint.shape)
        coordinate_normalize_joint = torch.tensor(coordinate_normalize_joint)
        angle_velocity = loc_exchange(coordinate_normalize_joint)
        position_set = coordinate_normalize_joint[1:]
        position_set = torch.tensor(position_set, dtype = torch.float32)
        angle_velocity = torch.tensor(angle_velocity, dtype = torch.float32)
        

        angle_position = torch.cat([position_set, angle_velocity],2)
        
        acc_set = acc_calculate(angle_position)
        acc_set = torch.unsqueeze(acc_set,-1)
        
        position_ang_vel_acc = torch.cat([angle_position[1:], acc_set],2)
        
        train_one_dataset = []
        for i in range(position_ang_vel_acc.shape[0]):            
            train_data = position_ang_vel_acc[i]
            train_data = np.array(train_data)
            train_one_dataset.append(train_data)
        train_one_dataset = np.array(train_one_dataset)
        print ('train_one_dataset:\n',train_one_dataset.shape)
        train_dataset.append(train_one_dataset)
    else:
        continue

train_dataset = np.array(train_dataset)
# save data
np.save(train_save_path, train_dataset)
'''

test_data_path = open(r'D:/Fourth_motion_prediction_paper/prediction_transformer_model/proprecess/test.txt').readline()
print ('test_data_path:\n',test_data_path)
test_data_path = test_data_path[:-1]
test_save_path = os.path.join( base_path, 'Sitting.npy')
test_save_path = test_save_path.replace("\\","/")

# load test data
test_data = h5py.File(test_data_path,'r')
test_coordinate_normalize_joint = test_data['coordinate_normalize_joint'][:,move_joint,:]
test_num = int(test_coordinate_normalize_joint.shape[0])
test_coordinate_normalize_joint = torch.tensor(test_coordinate_normalize_joint)
test_angle_velocity = loc_exchange(test_coordinate_normalize_joint)
test_position_set = test_coordinate_normalize_joint[1:]
test_position_set = torch.tensor(test_position_set, dtype = torch.float32)
test_angle_velocity = torch.tensor(test_angle_velocity, dtype = torch.float32)
test_angle_position = torch.cat([test_position_set, test_angle_velocity],2)

test_acc_set = acc_calculate(test_angle_position)
test_acc_set = torch.unsqueeze(test_acc_set,-1)

test_position_ang_vel_acc = torch.cat([test_angle_position[1:], test_acc_set],2)



print ('test_dataset:\n',test_position_ang_vel_acc.shape)

# save data
np.save(test_save_path, test_position_ang_vel_acc)