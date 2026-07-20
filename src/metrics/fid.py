import torch
from torchmetrics.image.fid import FrechetInceptionDistance


class FID():
    def __init__(self, device):
        self.device = device
        
        self.fid = FrechetInceptionDistance(feature = 2048).to(device)
        
    @torch.no_grad()
    def update_real(self, image_uint8):
        self.fid.update(image_uint8, real=True)
        
    @torch.no_grad()
    def update_fake(self, image_uint8):
        self.fid.update(image_uint8, real=False)
        
    def compute(self):
        fid = self.fid.compute().item()
        
        return {
            "fid": fid
        }
    
    def reset(self):
        self.fid.reset()