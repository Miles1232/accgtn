import yaml
import h5py
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from generator import Generator
import data_utils
import torch.autograd as autograd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='human36', type=str, help='选择数据集')
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
lr=config['learning_rate']
batch_size = config['batch_size']

if args.dataset =='human36':
    base_path = config['base_dir_human36']
    chain = [[1], [132.95, 442.89, 454.21, 162.77, 75], [132.95, 442.89, 454.21, 162.77, 75],
             [233.58, 257.08, 121.13, 115], [257.08, 151.03, 278.88, 251.73, 100],
             [257.08, 151.03, 278.88, 251.73, 100]]
else:
    chain = [[1], [176.88, 531.56, 510.1, 155.53, 77.87],
             [176.88, 531.56, 510.1, 155.53, 77.87],
             [144.16, 144.57, 218.49, 113.86, ],
             [256.19, 340.5591, 234.8878, 53.58, 83.6], [256.19, 340.5591, 234.8878, 53.58, 83.6]]
    base_path = config['base_dir_cmu']

# chain = [[1],[176.88,531.56, 510.1,155.53,77.87],
#             [176.88,531.56, 510.1,155.53,77.87],
#             [144.16, 144.57, 218.49,113.86,],
#             [256.19, 340.5591,234.8878,53.58,83.6],[256.19, 340.5591,234.8878,53.58,83.6]]
# chain = [[1], [132.95, 442.89, 454.21, 162.77, 75], [132.95, 442.89, 454.21, 162.77, 75],
#              [233.58, 257.08, 121.13, 115], [151.03, 278.88, 251.73, 100,100],
#              [151.03, 278.88, 251.73, 100,100]]
# print(base_path)
# base_path = './data/human36/eating'
train_save_path = os.path.join(base_path, 'train.npy')
train_save_path = train_save_path.replace("\\","/")
dataset = np.load(train_save_path,allow_pickle = True)
# chain = [[1],[104.92, 378.52, 408.43, 135.63,], [107.35, 380.67, 404.25, 138.51],
#              [116.09, 137.48, 55.28, 210.49, 94.05], [136.09, 111.86, 256.50, 254.13, 84.70, ],
#              [139.24, 108.56, 255.30, 258.67, 85.08]]

# for x in chain:
#     s = sum(x)
#     if s == 0:
#         continue
#     for i in range(len(x)):
#         x[i] = (i+1)*sum(x[i:])/s
# chain = [item for sublist in chain for item in sublist]
# nodes_weight = torch.tensor(chain).cuda()
# # nodes_weight = nodes_weight.unsqueeze(1).cuda()
# print(nodes_weight.shape)
# nodes_frame_weight = nodes_weight.expand(25, 24).cuda()
# frame_weight = torch.tensor([3,2, 1.5, 1.5, 1, 0.5, 0.2, 0.2, 0.1, 0.1,
#                              0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02,
#                              0.02, 0.02, 0.02,0.02]).cuda()
# print(frame_weight.shape)
# print(nodes_frame_weight.shape)
# print(chain)
for x in chain:
    s = sum(x)
    if s == 0:
        continue
    for i in range(len(x)):
        x[i] = (i+1)*sum(x[i:])/s
chain = [item for sublist in chain for item in sublist]
nodes_weight = torch.tensor(chain).cuda()
nodes_weight = nodes_weight.unsqueeze(1).cuda()
nodes_frame_weight = nodes_weight.expand(25, 25).cuda()
# nodes_frame_weight = nodes_weight.expand(24, 24).cuda()
frame_weight = torch.tensor([3, 2, 1.5, 1.5, 1, 0.5, 0.2, 0.2, 0.1, 0.1,
                             0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02,
                             0.02, 0.02, 0.02, 0.02]).cuda()

print(frame_weight,nodes_frame_weight)

for epoch in range(config['train_epoches']):

        for i in range (dataset.shape[0]):
            data = dataset[i]

            print(np.isnan(data).sum())
            # print(np.isinf(data).sum())

            train_data = data_utils.LPDataset(data, input_num, output_T_dim)
            print('data:\n',data.shape)
            # print(type(train_data))
            train_loader = DataLoader(
                dataset=train_data,
                batch_size=config['batch_size'],
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )

            model_x = Generator(in_features, out_features, node_num, input_dim, embed_size, num_layers,input_num,
                                output_T_dim,  heads, dropout,forward_expansion).cuda()
            model_y = Generator(in_features, out_features, node_num, input_dim, embed_size, num_layers,input_num,
                                output_T_dim,  heads, dropout,forward_expansion).cuda()
            model_z = Generator(in_features, out_features, node_num, input_dim, embed_size, num_layers,input_num,
                                output_T_dim,  heads, dropout,forward_expansion).cuda()
            model_a = Generator(in_features, out_features, node_num, input_dim, embed_size, num_layers,input_num,
                                output_T_dim,  heads, dropout,forward_expansion).cuda()

            mse = nn.MSELoss(reduction='mean')
            print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model_x.parameters()) / 1000000.0))

            optimizer_x = optim.Adam(model_x.parameters(), lr)
            optimizer_y = optim.Adam(model_y.parameters(), lr)
            optimizer_z = optim.Adam(model_z.parameters(), lr)
            optimizer_a = optim.Adam(model_a.parameters(), lr)

            print('pretrain generator')
            if os.path.exists(os.path.join(base_path, 'generator_a.pkl')):
                print('---------------------------------')
                model_x.load_state_dict(torch.load(os.path.join(base_path, 'generator_x.pkl')),strict=False)
                model_y.load_state_dict(torch.load(os.path.join(base_path, 'generator_y.pkl')),strict=False)
                model_z.load_state_dict(torch.load(os.path.join(base_path, 'generator_z.pkl')),strict=False)
                model_a.load_state_dict(torch.load(os.path.join(base_path, 'generator_a.pkl')),strict=False)


                for i, data in enumerate(train_loader):
                    optimizer_x.zero_grad()
                    optimizer_y.zero_grad()
                    optimizer_z.zero_grad()
                    optimizer_a.zero_grad()

                    in_shots, out_shot,in_shots2= data
                    # print(type(in_shots2))
                    # in_shots = torch.tensor(in_shots)
                    # out_shot = torch.tensor(out_shot)
                    # in_shots2 = torch.tensor(in_shots2)
                    in_shots = in_shots.cuda()
                    out_shot = out_shot.cuda()

                    input_angle = in_shots[:, :, :, 3:6]
                    input_acc_velocity = in_shots[:, :, :, 7]   
                    target_angle = out_shot[:, :, :, 3:6]
                    target_acc_velocity = out_shot[:, :, :, 7]

                    #read velocity
                    input_acc_velocity = input_acc_velocity.unsqueeze(-1).float()
                    target_acc_velocity = target_acc_velocity.float()


                    #read angle_x
                    input_angle_x = input_angle[:,:,:,0].unsqueeze(-1).float()
                    target_angle_x = target_angle[:,:,:,0].float()

                    #read angle_y
                    input_angle_y = input_angle[:,:,:,1].unsqueeze(-1).float()
                    target_angle_y = target_angle[:,:,:,1].float()

                    #read angle_z
                    input_angle_z = input_angle[:,:,:,2].unsqueeze(-1).float()
                    target_angle_z = target_angle[:,:,:,2].float()



                    loss_a = torch.tensor(0,dtype=torch.float32).cuda()
                    loss_x = torch.tensor(0,dtype=torch.float32).cuda()
                    loss_y = torch.tensor(0,dtype=torch.float32).cuda()
                    loss_z = torch.tensor(0,dtype=torch.float32).cuda()




                    output_a = model_a(input_acc_velocity)
                    output_a = output_a.view(target_acc_velocity.shape[0],output_T_dim, node_num)
                    loss_a += torch.mean(torch.norm((output_a- target_acc_velocity)*frame_weight*nodes_frame_weight, 2, 1))
                    # loss_a += torch.mean(
                    #     torch.norm((output_a - target_acc_velocity), 2, 1))

                    output_x = model_x(input_angle_x)
                    output_x = output_x.view(target_angle_x.shape[0], output_T_dim, node_num)
                    loss_x += torch.mean(torch.norm((output_x- target_angle_x)*frame_weight*nodes_frame_weight, 2, 1))

                    output_y = model_y(input_angle_y)
                    output_y = output_y.view(target_angle_y.shape[0], output_T_dim, node_num)
                    loss_y += torch.mean(torch.norm((output_y- target_angle_y)*frame_weight*nodes_frame_weight, 2, 1))

                    output_z = model_z(input_angle_z)
                    output_z = output_z.view(target_angle_z.shape[0], output_T_dim, node_num)
                    loss_z += torch.mean(torch.norm((output_z- target_angle_z)*frame_weight*nodes_frame_weight, 2, 1))


                    total_loss = loss_a*10 +loss_x + loss_y + loss_z
                    # print(loss_a,loss_x,loss_y)

                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model_x.parameters(), config['gradient_clip'])
                    nn.utils.clip_grad_norm_(model_y.parameters(), config['gradient_clip'])
                    nn.utils.clip_grad_norm_(model_z.parameters(), config['gradient_clip'])
                    nn.utils.clip_grad_norm_(model_a.parameters(), config['gradient_clip'])


                    optimizer_x.step()
                    optimizer_y.step()
                    optimizer_z.step()
                    optimizer_a.step()
                    print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, total_loss.item()))
                torch.save(model_x.state_dict(), os.path.join(base_path, 'generator_x.pkl'))
                torch.save(model_y.state_dict(), os.path.join(base_path, 'generator_y.pkl'))
                torch.save(model_z.state_dict(), os.path.join(base_path, 'generator_z.pkl'))
                torch.save(model_a.state_dict(), os.path.join(base_path, 'generator_a.pkl'))

            else:
                for i, data in enumerate(train_loader):
                    optimizer_x.zero_grad()
                    optimizer_y.zero_grad()
                    optimizer_z.zero_grad()
                    optimizer_a.zero_grad()

                    # print(len(data))
                    in_shots, out_shot,in_shots2= data
   
                    in_shots = in_shots.cuda() ##16 20 25 8
                    out_shot = out_shot.cuda() ## 16 25 25 8
    
                    input_angle = in_shots[:, :, :, 3:6]  ## 16 20 25 3
                    input_angle2 = in_shots2[:,:,:,3:6]
    
                    input_acc_velocity = in_shots[:, :, :, 7]##16 20 25
                    input_acc_velocity2 = in_shots2[:, :, :, 7]

                    target_angle = out_shot[:, :, :, 3:6] ## 16 25 25 3
                    target_acc_velocity = out_shot[:, :, :, 7] ##16 25 25

                    #read velocity
                    input_acc_velocity = input_acc_velocity.unsqueeze(-1).float() ## 16 20 25 1
                 
                    target_acc_velocity = target_acc_velocity.float() ## 16 25 25

                    #read angle_x
                    input_angle_x = input_angle[:,:,:,0].unsqueeze(-1).float()
                  
                    target_angle_x = target_angle[:,:,:,0].float()

                    #read angle_y
                    input_angle_y = input_angle[:,:,:,1].unsqueeze(-1).float()
                 
                    target_angle_y = target_angle[:,:,:,1].float()
                    #read angle_z
                    input_angle_z = input_angle[:,:,:,2].unsqueeze(-1).float()
             
                    target_angle_z = target_angle[:,:,:,2].float()




                    loss_a = torch.tensor(0,dtype=torch.float32).cuda()
                    loss_x = torch.tensor(0,dtype = torch.float32).cuda()
                    loss_y = torch.tensor(0,dtype = torch.float32).cuda()
                    loss_z = torch.tensor(0,dtype = torch.float32).cuda()

                    # print('input_acc_velocity:\n',input_acc_velocity.shape)

                    output_a = model_a(input_acc_velocity) ## 16 25 25
                    output_a = output_a.view(target_acc_velocity.shape[0],output_T_dim, node_num)
            
                    loss_a += torch.mean(torch.norm((output_a- target_acc_velocity)*frame_weight*nodes_frame_weight, p=2, dim=1))

                    output_x = model_x(input_angle_x)
                    output_x = output_x.view(target_angle_x.shape[0], output_T_dim, node_num)
                    loss_x += torch.mean(torch.norm((output_x- target_angle_x)*frame_weight*nodes_frame_weight, 2, 1))
                    
                    output_y = model_y(input_angle_y)
                    output_y = output_y.view(target_angle_y.shape[0], output_T_dim, node_num)
                    loss_y += torch.mean(torch.norm((output_y- target_angle_y)*frame_weight*nodes_frame_weight, 2, 1))
                 

                    output_z = model_z(input_angle_z)
                    output_z = output_z.view(target_angle_z.shape[0], output_T_dim, node_num)
                    loss_z += torch.mean(torch.norm((output_z- target_angle_z)*frame_weight*nodes_frame_weight, 2, 1))
       

                    total_loss = loss_a*10 + loss_x + loss_y + loss_z
                    total_loss.backward()
      

                    nn.utils.clip_grad_norm_(model_x.parameters(), config['gradient_clip'])
                    nn.utils.clip_grad_norm_(model_y.parameters(), config['gradient_clip'])
                    nn.utils.clip_grad_norm_(model_z.parameters(), config['gradient_clip'])
                    nn.utils.clip_grad_norm_(model_a.parameters(), config['gradient_clip'])

                    optimizer_x.step()
                    optimizer_y.step()
                    optimizer_z.step()
                    optimizer_a.step()

                    print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, total_loss.item()))
                torch.save(model_x.state_dict(), os.path.join(base_path, 'generator_x.pkl'))
                torch.save(model_y.state_dict(), os.path.join(base_path, 'generator_y.pkl'))
                torch.save(model_z.state_dict(), os.path.join(base_path, 'generator_z.pkl'))
                torch.save(model_a.state_dict(), os.path.join(base_path, 'generator_a.pkl'))
torch.save(model_x, os.path.join(base_path, 'generator_x.pkl'))
torch.save(model_y, os.path.join(base_path, 'generator_y.pkl'))
torch.save(model_z, os.path.join(base_path, 'generator_z.pkl'))
torch.save(model_a, os.path.join(base_path, 'generator_a.pkl'))
print ('Parameters are stored in the generator.pkl file')


