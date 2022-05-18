import configparser
import logging
import torch
import os
from torch.optim import lr_scheduler
from model.transformer import Transformer
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def train_start(gpu_list,config_Path):

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
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    train(config, gpu_list)


def train(config, gpu_list,dataset):

    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")
    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))

    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
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


    logger.info("Training start....")
    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)

    for epoch in range(1, config.epochs + 1):
        print(f"epoch:{epoch}")
        for batch in train_iter:
            feature, target = batch.statement, batch.label
            # print(type(feature))
            # print(feature)
            # sys.exit()
            if torch.cuda.is_available():  # 如果有GPU将特征更新放在GPU上
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()  # 将梯度初始化为0，每个batch都是独立训练地，因为每训练一个batch都需要将梯度归零
            # print("train_size",feature.shape)
            logits = model(feature)
            loss = F.cross_entropy(logits, target)  # 计算损失函数 采用交叉熵损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 放在loss.backward()后进行参数的更新
            steps += 1
            if steps % config.steps_show == 0:  # 每训练多少步计算一次准确率，我这边是1，可以自己修改
                corrects = (torch.max(logits, 1)[1].view(
                    target.size()).data == target.data).sum()
                _, candidate = (torch.topk(logits, k, dim=1))
                # print(target.data.shape,candidate.data.shape)
                correct2 = 0
                for each in range(0, target.data.shape[0]):
                    if target.data[each] in candidate.data[each]:
                        correct2 += 1
                # correct2 = (target.data in candidate.data).sum()
                # logits是[128,10],torch.max(logits, 1)也就是选出第一维中概率最大的值，
                # 输出为[128,1],torch.max(logits, 1)[1]相当于把每一个样本的预测输出取出来，
                # 然后通过view(target.size())平铺成和target一样的size (128,),然后把与target中相同的求和，
                # 统计预测正确的数量
                train_acc = 100.0 * corrects / batch.batch_size  # 计算每个mini batch中的准确率
                train_acc_k = 100.0 * correct2 / batch.batch_size
                print('steps:{} - loss: {:.6f}  acc:{:.4f} acck:{:.4f}'.format(
                    steps,
                    loss.item(),
                    train_acc,
                    train_acc_k
                ))

            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)




