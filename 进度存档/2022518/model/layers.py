import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
import numpy as np
import math

class FFN(nn.Module):
    def __init__(self,d_model,d_ff):
        super(FFN,self).__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self,data):
        residual = data
        data = self.fc1(data)
        data = self.relu(data)
        data = self.fc2(data)
        # residual block
        data = self.norm(data+residual)
        return data

class Pos_Encoder():
    def __init__(self):
        super(Pos_Encoder, self).__init__()
    def pos_deal(self,pos,d_model):
        return [ pos/np.power(10000,2*(i//2)/d_model) for i in range(d_model) ]
    def pos_encoding(self,d_model,pos_len):
        pos_polish = [self.pos_deal(pos,d_model) for pos in range(pos_len)]
        pos_polish = np.array(pos_polish)
        pos_polish[:,0::2] = np.sin(pos_polish[:,0::2])
        pos_polish[:,1::2] = np.cos(pos_polish[:,1::2])
        return torch.FloatTensor(pos_polish)

class Self_Atten(nn.Module):
    def __init__(self):
        super(Self_Atten,self).__init__()
    def forward(self,d_k,Q,K,V,mask=None):
        value_w = torch.bmm(Q , torch.transpose(K, -1, -2)/math.sqrt(d_k))
        if mask != None:
            value_w.masked_fill_(mask, -1e9)
        value_w = torch.softmax(value_w,dim=-1)
        V = torch.bmm(value_w,V)
        return V

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Multi_Head_Atten(nn.Module):
    def __init__(self,N,d_model,d_k,d_v):
        super(Multi_Head_Atten,self).__init__()
        self.N = N
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model,N*d_k)
        self.W_K = nn.Linear(d_model,N*d_k)
        self.W_V = nn.Linear(d_model,N*d_v)
        self.W_O = nn.Linear(N*d_v,d_model)
        self.SelfAtten = Self_Atten()
        self.norm = nn.LayerNorm(d_model)

    def forward(self,Q,K,V,mask=None):
        residual, batch = Q, Q.size(0)
        # [batch,len_q,d_model] lin-> [batch,len_q,N*d_model]
        # view it -> [batch,N,len_q,d_k]
        Qv = self.W_Q(Q).view(batch,self.N,-1,self.d_k)
        Kv = self.W_K(K).view(batch,self.N,-1,self.d_k)
        Vv = self.W_V(V).view(batch, self.N, -1, self.d_v)
        if mask != None:
            # multi_mask [batch,len_q,len_k] -> [batch,N,len_q,len_k]
            multi_mask = mask.unsqueeze(1).repeat(1,self.N,1,1)
        else:
            multi_mask = None
        V_res = self.SelfAtten(self.d_k,Qv,Kv,Vv,multi_mask).transpose(1,2).contiguous().view(batch,-1,self.N*self.d_v)
        V_res = self.W_O(V_res)
        V_res = self.norm(V_res+residual)
        return V_res

class Encoder_Sublayer(nn.Module):
    def __init__(self,N,d_k,d_model,d_v,d_ff):
        super(Encoder_Sublayer,self).__init__()
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.Multi_Head_Atten = Multi_Head_Atten(N=self.N,d_model=self.d_model,d_k=self.d_k,d_v=self.d_v)
        self.FFN = FFN(d_model=self.d_model,d_ff=self.d_ff)
    def forward(self,data,mask):
        data = self.Multi_Head_Atten(data,data,data,mask)
        data = self.FFN(data)
        return data

def get_pad_mask(q_seq, k_seq):
    batch, len_q = q_seq.size()
    batch, len_k = k_seq.size()
    # 0 : PAD token
    pad_mask = k_seq.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_mask.expand(batch, len_q, len_k)  # batch_size x len_q x len_k

def get_encode_mask(seq):
    encode_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # triu : 对矩阵进行拷贝，并将第k条对角线下方的元素全部置零。
    subsequent_mask = np.triu(np.ones(encode_shape), k=1)
    # pad_mask = get_pad_mask(seq, k_seq)
    # for idx1 in range(encode_shape[0]):
    #     for idx2 in range(encode_shape[1]):
    #         for idx3 in range(encode_shape[2]):
    #             if pad_mask[idx1][idx2][idx3]==0 & subsequent_mask[idx1][idx2][idx3]==1:
    #                 subsequent_mask[idx1][idx2][idx3] = 0

    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class Encoder(nn.Module):
    def __init__(self, layer_num,N,d_k,d_model,d_v,d_ff,vocab_size,embedding_dim):
        super(Encoder,self).__init__()
        self.layer_num = layer_num
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = Pos_Encoder()
        self.layer_list = nn.ModuleList([Encoder_Sublayer(N=self.N,d_model=self.d_model,d_k=self.d_k,d_v=self.d_v) for _ in range(self.layer_num)])

    # def make_layer(self, layer_num):
    #     layers = []
    #     for _ in layer_num:
    #         layers.append(Multi_Head_Atten(N=self.N,d_model=self.d_model,d_k=self.d_k,d_v=self.d_v)
    #         layers.append(FFN(d_model=self.d_model,d_ff=self.d_ff))
    #     return nn.Sequential(*layers)

    def forward(self,encode_input,pos_len):
        data = self.embedding(encode_input)
        data *= math.sqrt(self.d_k)
        # + pos_encoder
        data += self.pos_encoder.pos_encoding(d_model=self.d_model,pos_len = pos_len)
        mask = get_pad_mask(encode_input, encode_input)
        for layer in self.layer_list:
            data = layer(data,mask)
        return data

class Decoder_Sublayer(nn.Module):
    def __init__(self,N,d_k,d_model,d_v,d_ff):
        super(Decoder_Sublayer,self).__init__()
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.Multi_Head_Atten = Multi_Head_Atten(N=self.N,d_model=self.d_model,d_k=self.d_k,d_v=self.d_v)
        self.Mask_Multi_Head_Atten = Multi_Head_Atten(N=self.N, d_model=self.d_model, d_k=self.d_k, d_v=self.d_v)
        self.FFN = FFN(d_model=self.d_model,d_ff=self.d_ff)

    def forward(self,decode_input,encode_output,pad_mask,encode_mask):
        decode_out = self.Multi_Head_Atten(decode_input,decode_input,decode_input,encode_mask)
        decode_out = self.Mask_Multi_Head_Atten(decode_out,encode_output,encode_output,pad_mask)
        decode_out = self.FFN(decode_out)
        return decode_out

class Decoder(nn.Module):
    def __init__(self, layer_num,N,d_k,d_model,d_v,d_ff,vocab_size,embedding_dim):
        super(Decoder,self).__init__()
        self.layer_num = layer_num
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = Pos_Encoder()
        self.layer_list = nn.ModuleList([Decoder_Sublayer(N=self.N,d_model=self.d_model,d_k=self.d_k,d_v=self.d_v) for _ in range(self.layer_num)])


    def forward(self,decode_input,encode_output,pos_len):
        decode_output = self.embedding(decode_input)
        decode_output *= math.sqrt(self.d_k)
        # + pos_encoder
        decode_output += self.pos_encoder.pos_encoding(d_model=self.d_model,pos_len = pos_len)
        decode_pad_mask = get_pad_mask(decode_input, decode_input)
        encode_mask = torch.gt((get_encode_mask(decode_input) + decode_pad_mask),0)
        pad_mask = get_pad_mask(encode_output, decode_input)
        for layer in self.layer_list:
            decode_output = layer(decode_output,encode_output,pad_mask,encode_mask)
        return decode_output













