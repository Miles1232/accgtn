import torch
import torch.nn as nn
from model.GCN import GraphConvolution, Spatial_Conv
from model.Transformer import STTransformer,PositionalEmbedding,Transformer
from model.ATGCN import SFGCN
import numpy as np
class Generator(nn.Module):
    def __init__(self, in_features, out_features, node_num, input_dim, embed_size, num_layers,input_num,
                           output_T_dim,  heads, dropout,forward_expansion):
        super(Generator, self).__init__()
        self.sp_conv1 = Spatial_Conv(node_num)
        self.sp_conv2 = Spatial_Conv(node_num)
        self.sp_conv3 = Spatial_Conv(node_num)

        # self.sp_conv11 = Spatial_Conv(node_num)
        # self.sp_conv22 = Spatial_Conv(node_num)
        # self.sp_conv33 = Spatial_Conv(node_num)

        self.gr_conv1 = GraphConvolution(in_features, out_features)
        self.gr_conv2 = GraphConvolution(out_features, out_features*forward_expansion)
        self.gr_conv3 = GraphConvolution(out_features*forward_expansion, out_features*forward_expansion)

        self.sfgcn = SFGCN(in_features,out_features,out_features*forward_expansion,dropout)
        # self.sfgcn1 = SFGCN(in_features, out_features, out_features * forward_expansion, dropout)


        self.Transformer = STTransformer(input_dim, embed_size, num_layers,input_num,
                               output_T_dim,  heads, dropout,forward_expansion)
        # self.Transformer1 = STTransformer(input_dim, embed_size, num_layers, input_num,
        #                                  output_T_dim, heads, dropout, forward_expansion)
    def forward(self, input1,input2):
        output1 = self.sp_conv1(input1)
        # print(input)
        output1 = output1.unsqueeze(-1)
        # print(output1.shape)
        output2 = self.sp_conv2(output1)
        output2 = output2.unsqueeze(-1)

        output3 = self.sp_conv3(output2)
        output3 = output3.unsqueeze(-1)
        # # print(output3.shape)
        # gr_output1 = self.gr_conv1(output1)
        # gr_output2 = self.gr_conv2(gr_output1)
        # gr_output3 = self.gr_conv3(gr_output2)
        gr_output3 = self.sfgcn(output3)
        # print(gr_output3.shape)
        pred_ouput1 = self.Transformer(gr_output3)
        # pred_ouput1 = gr_output3[:, :, :, 0]
        pred_ouput1 = pred_ouput1.permute(0, 2, 1)


        # output11 = self.sp_conv11(input2)
        # # print(input)
        # output11 = output11.unsqueeze(-1)
        #
        # output22 = self.sp_conv22(output11)
        # output22 = output22.unsqueeze(-1)
        #
        # output33 = self.sp_conv33(output22)
        # output33 = output33.unsqueeze(-1)
        # # #
        #
        # gr_output33 = self.sfgcn1(output11)
        # # pred_ouput2 = gr_output33[:,:,:,0]
        # pred_ouput2 = self.Transformer1(gr_output33)
        #
        # pred_ouput2 = pred_ouput2.permute(0, 2, 1)
        #
        # print('pred_ouput:\n',pred_ouput2.shape)
        # print(pred_ouput2)
        # print(pred_ouput1)

        # pred_ouput = (pred_ouput1+pred_ouput2)/2
        return pred_ouput1

if __name__=="__main__":
    # import numpy as np
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    # emb = PositionalEmbedding(16,25)
    # input = torch.ones(16, 20,25,1)*100
    # input = np.load('./data/human36/train.npy',allow_pickle=True)
    # input = input[0][0:1*20,:,0]
    # input = torch.tensor(input).unsqueeze(-1).reshape(-1,20,25,1)
    # s_scaler = preprocessing.StandardScaler()
    # feature = s_scaler.fit_transform(input.reshape(-1,25))
    # print(feature)
    # feature = torch.tensor(feature).reshape(-1,20,25,1).to(torch.float32)
    # # print(input[0])
    # input2 = torch.ones(1,20,25,1)
    # model = Generator(1, 4, 25, 16, 64, 3, 20, 25, 4, 0, 4)
    # # print(input)
    # output = model(input,input2)
    # print(output.shape)
    # output = output.detach().numpy()
    #
    # # 数据还原
    # origin_feature = s_scaler.inverse_transform(output.reshape(-1,25))
    # print(origin_feature)
    # origin_feature = origin_feature.reshape(-1,25,25)
    # print(origin_feature-input[0][0:1*25,:,0].reshape(-1,25,25))

    data= np.load('./data/human36/train.npy',allow_pickle=True)
    input = data[0][0:1 * 20, :, 0:3]
    print(input.shape)
    input = input.reshape(1,20,75,1)
    input = torch.tensor(input)
    a = data[0][20:45,:,0:3]
    a = a.reshape(1,25,75)
    a = torch.tensor(a)
    model = Generator(1, 4, 75, 16, 64, 3, 20, 25, 4, 0, 4)
    output = model(input)
    print(output)
    b = output+a
    b =b.reshape(-1,3)
    a = a.reshape(-1,3)
    a = torch.tensor(a)
    print(b-a)
    loss_x = torch.mean(torch.norm(b-a, 2, 1))
    print(loss_x)

