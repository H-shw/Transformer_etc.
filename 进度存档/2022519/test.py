# -- coding:UTF-8 --
import os
import sys
import torch
import torch.nn.functional as F


def load_model(config):
    file_path = config.get('test','model_path')
    model = torch.load(file_path)
    return model

def test(test_iter,config):

    model = load_model(config).cuda()
    f = open(config.subject_res,"w",encoding = 'utf-8')

    model.eval()
    steps = 0

    with torch.no_grad():
        for batch in test_iter:
            feature= batch.statement
            if torch.cuda.is_available():  # 如果有GPU将特征更新放在GPU上
                feature = feature.cuda()

            logits = model(feature)
            steps += 1
            _, candidate = (torch.topk(logits, config.getint('model','top_k'), dim=1))
            for each in range(0, candidate.data.shape[0]):
                # f.write(','.join(list(dict[int(idx)] for idx in candidate.data[each])))
                f.write('\n')

    f.close()








