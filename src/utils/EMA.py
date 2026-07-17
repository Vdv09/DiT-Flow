import copy
import torch


class EMA:
    def __init__(self, model, beta):
        self.beta = beta
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, new_model):
        for ema_param, new_param in zip(self.ema_model.parameters(), new_model.parameters()):
            ema_param.data = self.beta * ema_param.data + (1 - self.beta) * new_param.data
    
    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)