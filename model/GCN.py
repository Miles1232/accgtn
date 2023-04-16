import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import yaml
# config = yaml.load(open('./config.yml'),Loader=yaml.FullLoader)
# node_num = config['node_num']
class Spatial_Graph():
    def __init__(self):
        self.get_edge()
    def get_edge(self):
        self.node_num = 25
        self_link = [(i, i) for i in range(self.node_num)]
        bone_link = [(0, 1), (1, 2), (2, 3), (3,4), (4,5), (0,6), (6,7), (7,8), (8,9), (9,10), (0,11), (11,12), (12,13), (13,14),
                     (12,15), (15,16), (16,17),(17,19), (17,18), (12,20), (20,21), (21,22), (22,24), (22,23)]
        # bone_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
        #              (11, 12), (12, 13), (11, 14), (14, 15),
        #              (15, 16), (16, 17), (17, 18), (11, 19), (19, 20), (20, 21), (21, 22), (22, 23), ]
        self.edge = self_link + bone_link

        self.pre_sem_edge = [(2,7),(3,8),(16,21),(17,22)]
        # self.pre_sem_edge = [(2, 6), (3, 7), (15, 20), (16, 21)]
        A_ske = torch.zeros((self.node_num, self.node_num))
        for i, j in self.edge:
            A_ske[j, i] = 1
            A_ske[i, j] = 1
        self.A_ske = A_ske
        A_pre_sem = torch.zeros((self.node_num, self.node_num))
        for p, q in self.pre_sem_edge:
            A_pre_sem[p, q] = 1
            A_pre_sem[q, p] = 1
        self.A_pre_sem = A_pre_sem
        
        self.complete_adj = self.A_ske + self.A_pre_sem
        # print(self.complete_adj,self.A_ske)
        return self.complete_adj

class Spatial_Conv(nn.Module):
    
    def __init__(self, node_num, bias=True):
        super(Spatial_Conv, self).__init__()
        self.node_num = node_num

        self.graph = Spatial_Graph()

        self.adj = nn.Parameter(self.graph.complete_adj)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(node_num))

            stdv = 1. / math.sqrt(self.adj.size(1))

            self.bias.data.uniform_(-stdv, stdv)
            
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        self.adj = nn.Parameter(torch.where(torch.isnan(self.adj), torch.full_like(self.adj, 0), self.adj))
        # print('self.adj:\n',self.adj.shape)
        # print(self.adj)
        # print('input:\n',input.shape)
        # input = input.unsqueeze(-1)
        output = torch.matmul(self.adj, input)
        
        output = output.squeeze(-1)
        # print('output1:\n',output.shape)
        #
        # print('self.bias:\n',self.bias.shape)
        # exit()

        # output = torch.matmul(self.adj, input)
        # print(output.shape)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

        
        
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features,  bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.node_num = 25
        self.Adj = nn.Parameter(torch.FloatTensor(self.node_num, self.node_num))
        # print(self.Adj)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x,adj):
        # x =x.reshape(-1,20,25,3)
        support = torch.matmul(x, self.weight)  ## 16 20 25 1 * 1 4  == 16 20 25 4
        # support = support.reshape(-1,20,75)
        output = torch.matmul(adj, support) ## 25 25  * 16 20 25 4
        # output = torch.matmul(self.Adj, support)  ## 25 25  * 16 20 25 4
        if self.bias is not None:
            return output + self.bias
        else:
            return output
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
        print('emb',emb.shape)
        emb, att = self.attention(emb)

        return emb
        # print(att.shape)
        # return att
if __name__=="__main__":
    
    
    sp_conv1 = Spatial_Conv(25)
    sp_conv2 = Spatial_Conv(25)
    sp_conv3 = Spatial_Conv(25)
    
    # gr_conv1 = GraphConvolution(1, 4)
    # gr_conv2 = GraphConvolution(4, 16)
    # gr_conv3 = GraphConvolution(16, 16)
    sfgcn = SFGCN(1,4,16,0)

    input = torch.ones(16, 20,25,1)*100
    # print('input:\n',input.shape)
    output1 = sp_conv1(input)
    output1 = output1.unsqueeze(-1)
    # print('output1:\n',output1)
    
    output2 = sp_conv2(output1)
    output2 = output2.unsqueeze(-1)
    
    output3 = sp_conv3(output2)
    output3 = output3.unsqueeze(-1)  ## batch input_num node 1
    # print(output3)
    
    
    # gr_output1 = gr_conv1(output3)
    # print('gr_output1:\n',gr_output1)
    #
    # gr_output2 = gr_conv2(gr_output1)
    # print('gr_output2:\n',gr_output2.shape)
    #
    # gr_output3 = gr_conv3(gr_output2)
    # print('gr_output3:\n',gr_output3.shape)
    a = sfgcn(output1)
    print(a[0][0])










