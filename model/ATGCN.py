from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import math
from model.GCN import GraphConvolution, Spatial_Conv,Spatial_Graph
import torch.nn.functional as F
from model.Transformer import STTransformer
import yaml
# config = yaml.load(open('./config.yml'),Loader=yaml.FullLoader)
# node_num = config['node_num']
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout): ## 1, 4 16
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x,adj):
        x = F.relu(self.gc1(x,adj))
        # x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x,adj)
        # x = self.gc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # print(beta.shape)
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.graph = Spatial_Graph()
        self.sadj = nn.Parameter(self.graph.A_ske)
        # self.sadj = nn.Parameter(self.graph.complete_adj)
        self.node_num = 25
        self.Adj = Parameter(torch.FloatTensor(self.node_num, self.node_num))
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        emb1 = self.SGCN1(x,self.sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x,self.sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x,self.Adj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x,self.Adj) # Special_GCN out2 -- fadj feature graph
        # emb1 = self.SGCN1(x,)  # Special_GCN out1 -- sadj structure graph
        # com1 = self.CGCN(x, )  # Common_GCN out1 -- sadj structure graph
        # com2 = self.CGCN(x, )  # Common_GCN out2 -- fadj feature graph
        # emb2 = self.SGCN2(x, )
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        # print(emb.shape)
        return emb
        # print(att.shape)
        # return att

if __name__ == '__main__':
    sp_conv1 = Spatial_Conv(25)
    sp_conv2 = Spatial_Conv(25)
    sp_conv3 = Spatial_Conv(25)

    gr_conv1 = GraphConvolution(1, 4)
    gr_conv2 = GraphConvolution(4, 16)
    gr_conv3 = GraphConvolution(16, 16)

    net= SFGCN(nfeat=1, nhid1=4,nhid2=16,dropout=0)


    input = torch.ones(16, 20, 25, 1)
    # print('input:\n',input.shape)
    output1 = sp_conv1(input)
    output1 = output1.unsqueeze(-1)
    print('output1:\n',output1.shape)

    output2 = sp_conv2(output1)
    output2 = output2.unsqueeze(-1)

    output3 = sp_conv3(output2)
    output3 = output3.unsqueeze(-1)

    output = net(output3)
    print(output.shape)

    transformer = STTransformer(input_dim=16, embed_size=64, num_layers=3,input_num=20,
                               output_T_dim=25,  heads=4, dropout=0,forward_expansion=4)

    predict = transformer(output)
    print(predict.shape)
    # gr_output1 = gr_conv1(output3)
    # print('gr_output1:\n', gr_output1.shape)
    #
    # gr_output2 = gr_conv2(gr_output1)
    # print('gr_output2:\n', gr_output2.shape)
    #
    # gr_output3 = gr_conv3(gr_output2)
    # print('gr_output3:\n', gr_output3.shape)
