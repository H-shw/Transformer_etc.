# -- coding:UTF-8 --
import configparser
import logging
import torch
import os
import torch.nn.functional as F
from torch.optim import lr_scheduler
from model.transformer import Transformer
from utils import save_model

def train_start(train_iter,gpu_list,config_Path):

    use_gpu = True
    if not gpu_list:
        use_gpu = False
    else:
        use_gpu = True
        gpu_string = ",".join([str(x) for x in gpu_list])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string

    config = configparser.ConfigParser()
    config.read(config_Path, encoding='utf-8')

    cuda = torch.cuda.is_available()
    print("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        print("CUDA is not available but specific gpu id")
        raise NotImplementedError
    train(train_iter,config)


def train(train_iter,config):

    t_epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")
    output_path = os.path.join(config.get("output", "model_save_path"), config.get("output", "model_name"))

    if os.path.exists(output_path):
        print("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    N = config.getint('model','N')
    d_k = config.getint('model','d_k')
    d_v = config.getint('model','d_v')
    d_model = config.getint('model','d_model')
    d_ff = config.getint('model','d_ff')
    src_vocab_size = config.getint('model','src_vocab_size')
    tgt_vocab_size = config.getint('model','tgt_vocab_size')
    embedding_dim = config.getint('model','embedding_dim')
    layer_num = config.getint('model','layer_number')
    pos_len = config.getint('model','pos_len')

    model = Transformer(layer_num,N,d_k,d_model,d_v,d_ff,src_vocab_size,tgt_vocab_size,embedding_dim,pos_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.getint('model','lr') )

    print("Training start....")
    best_acc = 0
    steps = 0

    for epoch in range(1, t_epoch+1):
        print(f"epoch:{epoch}")
        for batch in train_iter:

            feature, target = batch.statement, batch.label
            if torch.cuda.is_available():
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()

            logits = model(feature)
            _, candidate = (torch.topk(logits, config.getint('model','top_k'), dim=1))

            loss = F.cross_entropy(logits, target)  # 计算损失函数 采用交叉熵损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 放在loss.backward()后进行参数的更新

            steps += 1
            if steps % config.steps_show == 0:
                corrects = (torch.max(logits, 1)[1].view(
                    target.size()).data == target.data).sum()

                train_acc = 100.0 * corrects / batch.batch_size  # 计算每个mini batch中的准确率

                print('steps:{} - loss: {:.6f}  acc:{:.4f} '.format(
                    steps,
                    loss.item(),
                    train_acc
                ))
                if train_acc > best_acc:
                    save_model(model,'./model_res','Transformer',epoch,steps)






