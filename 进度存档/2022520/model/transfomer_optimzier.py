import numpy as np

class Transformer_Optimizer(object):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        # lrate = d^{−0.5}_{model} · min(step_num^{−0.5}, step_num · warmup_steps^{−1.5})
        self.step_num += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([np.power(self.step_num, -0.5),np.power(self.warmup_steps, -1.5) * self.step_num])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr