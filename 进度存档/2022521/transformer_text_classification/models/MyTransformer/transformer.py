# -- coding:UTF-8 --
from models.MyTransformer.layers import *

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'MyTransformer'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer,self).__init__()
        self.pos_len = config.pad_size
        self.N = config.num_head
        self.d_k = config.dim_model // config.num_head
        self.d_v = config.dim_model // config.num_head
        self.d_model = config.dim_model
        self.d_ff = config.hidden
        self.src_vocab_size = config.n_vocab
        # self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = self.d_model
        self.encoder = Encoder(config.num_encoder, self.N, self.d_k, self.d_model, self.d_v, self.d_ff, self.src_vocab_size, self.embedding_dim)
        if config.embedding_pretrained is not None:
            self.encoder.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            print('NO')
            self.encoder.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        # self.decoder = Decoder(layer_num, N, d_k, d_model, d_v, d_ff, tgt_vocab_size, embedding_d)
        self.fc = nn.Linear(self.pos_len * self.d_model, config.num_classes)
        # self.fc = nn.Linear(d_model,tgt_vocab_size,bias=False)
        # shared weight -- embedding & projection
        # self.encoder.embedding.weight = self.decoder.embedding.weight
        # self.fc.weight = self.decoder.embedding.weight

    def forward(self, encode_input):
        encode_output = self.encoder(encode_input[0], self.pos_len)
        # decode_output = self.decoder(decode_input, encode_input, encode_output, self.pos_len)
        # print(encode_output.shape)
        encode_output = encode_output.view(encode_output.size(0),-1)
        data = self.fc(encode_output)
        # return torch.softmax(data,dim=-1).view(-1, self.tgt_vocab_size)
        return data


