# -- coding:UTF-8 --
from model.layers import *

class Transformer(nn.Module):
    def __init__(self,layer_num,N,d_k,d_model,d_v,d_ff,src_vocab_size,tgt_vocab_size,embedding_d,pos_len):
        super(Transformer,self).__init__()
        self.pos_len = pos_len
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_d
        self.encoder = Encoder(layer_num, N, d_k, d_model, d_v, d_ff, src_vocab_size, embedding_d)
        self.decoder = Decoder(layer_num, N, d_k, d_model, d_v, d_ff, tgt_vocab_size, embedding_d)
        self.fc = nn.Linear(d_model,tgt_vocab_size)

    def forward(self,encode_input,decode_input):
        encode_output = self.encoder(encode_input, self.pos_len)
        decode_output = self.decoder(decode_input, encode_input, encode_output, self.pos_len)
        data = self.fc(decode_output)
        return torch.softmax(data,dim=-1).view(-1, self.tgt_vocab_size)


