import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=25):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # print(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        
        # print('query:\n',query.shape)
        # exit()
        N, T, C = query.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, T, self.heads, self.head_dim)  # embed_size维拆成 heads×head_dim
        keys   = keys.reshape(N, T, self.heads, self.head_dim)
        query  = query.reshape(N, T, self.heads, self.head_dim)

        values  = self.values(values)  # (N, T, heads, head_dim)
        keys    = self.keys(keys)      # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])   # 时间self-attention
        # queries shape: (N, T, heads, heads_dim),
        # keys shape: (N, T, heads, heads_dim)
        # energy: (N, T, T, heads)
        
        
        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)  # 在K维做softmax，和为1
        # attention shape: (N, query_len, key_len, heads)

        out = torch.einsum("nqkh,nkhd->nqhd", [attention, values]).reshape(
                N, T, self.heads * self.head_dim
        )
        # attention shape: (N, T, T, heads)
        # values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, T, embed_size)

        return out
    
    


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        
        self.attention = TSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):

        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads,  dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.TTransformer = TTransformer(embed_size, heads, dropout, forward_expansion)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query):
        # Add skip connection,run through normalization and finally dropout
        x = self.dropout( self.norm(self.TTransformer(value, key, query) + query) )
        return x




class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        dropout,
        forward_expansion
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('x:',x)
        out = self.dropout(x)        
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out)
        return out
    
class Transformer(nn.Module):
    def __init__(
        self,
        embed_size=64,
        num_layers=3,
        heads=4,
        dropout=0,
        forward_expansion = 4
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            dropout,
            forward_expansion
        )

    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src


class STTransformer(nn.Module):
    def __init__(
        self, 
        input_dim = 16,
        embed_size = 64, 
        num_layers = 3,
        input_num = 10,
        output_T_dim = 25,  
        heads = 4,
        dropout = 0,
        forward_expansion = 4        
    ):        
        super(STTransformer, self).__init__()
        # 
        self.temporal_embedding = nn.Linear(input_dim, embed_size)

        self.Transformer = Transformer(
            embed_size, 
            num_layers, 
            heads, 
            dropout,
            forward_expansion
        )
                
        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(input_num, output_T_dim, 1)  
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.weighted_mean = torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3,padding=1, padding_mode='circular')
    
    def forward(self, x):
        # input x shape[ C, N, T] 
        # C:通道数量。  N:传感器数量。  T:时间数量
        
        bat, fra, node, fea = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        #fra and node exchange position, bat*fra*node*fea --> bat*node*fra*fea
        input_Transformer = x.permute(0, 2, 1, 3)
        
        input_Transformer = input_Transformer.reshape(-1, fra, fea)  ## 16*25 20 16


        # print('input_Transformer:\n',input_Transformer.shape)
        
        #(bat*node)*fra*16 --> (bat*node)*fra*64
        input_Transformer = self.temporal_embedding(input_Transformer)

        # print('input_Transformer:\n',input_Transformer.shape)
        # exit()
        
        #input_Transformer shape[N, T, C]
        output_Transformer = self.Transformer(input_Transformer)
        a = self.weighted_mean(input_Transformer)
        # print(a.shape)
        # output_Transformer = output_Transformer + a
        # output_Transformer = output_Transformer + input_Transformer
        # output_Transformer = self.weighted_mean(output_Transformer)
        # print('Transform',output_Transformer.shape)
        output_Transformer = output_Transformer.reshape(bat, node, fra, -1)
        # print('output_Transformer:\n',output_Transformer)
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        # print('output_Transformer:\n',output_Transformer) ## 16 10 25 64
        out = self.relu(self.conv2(output_Transformer))    # 等号左边 out shape: [1, output_T_dim, N, C]       1 25 25 64
        out = out.permute(0, 3, 2, 1)           # 等号左边 out shape: [1, C, N, output_T_dim]
        out = self.conv3(out)                   # 等号左边 out shape: [1, 1, N, output_T_dim]
        out = out.squeeze(1)
        # out = tokenConv(out)
        return out
        # return out shape: [N, output_dim]
    



















# class STTransformer(nn.Module):


if __name__=="__main__":
  
    emb = PositionalEmbedding(16, 25)
    input = torch.ones(16,20,25,16)
    # input = input.reshape(-1,25,16)*100
    nu = emb(input)
    print('nu:\n',nu.shape)
    input_pos = input + nu
    input_pos = input_pos.reshape(16,20,25,16)
    
    # print(nu)
    print(input_pos.shape)
    
    model = STTransformer(input_dim = 16, embed_size = 64, num_layers = 3,input_num = 20,
                           output_T_dim = 25,  heads = 4, dropout = 0,forward_expansion = 4)
    input_pos = torch.tensor(input_pos.numpy(), dtype=torch.float32)
    output = model(input_pos)
    print(output.shape)
    # exit()
    
    tokenConv = nn.Conv1d(in_channels=25, out_channels=25,
                                    kernel_size=3, padding=1, padding_mode='circular')

    out = tokenConv(input)
    print(out)
















