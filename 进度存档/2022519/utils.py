# -- coding:UTF-8 --
import os
import torch

def save_model(model, save_dir, name,epoch,step):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = f'{name}+e:{epoch}+s:{step}.pt'
    save_bestmodel_path = os.path.join(save_dir, save_path)
    torch.save(model, save_bestmodel_path)