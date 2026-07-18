from .base import DiffusionBase

import torch

from class_registry import class_registry
from sampling.cfg import cfg_predict

@class_registry.add_to_registry("ddim")
class DDIM(DiffusionBase):
    def make_timesteps(self, number_steps):
        T = self.schedule.number_steps

        return torch.linspace(T - 1, 0, number_steps, dtype=torch.long)
    
    @torch.no_grad()
    def ddim_step(
        self, 
        model, 
        x_t, 
        t, 
        prev_t = None,
        y = None,
        guidance_scale = 1,
        null_class_label = None
    ):
        eps = cfg_predict(model, x_t, t, y, guidance_scale, null_class_label)

        x_0 = self.predict_x0_from_eps(x_t, t, eps)
        
        if prev_t is None:
            return x_0

        x_prev = self.schedule.get_data(self.schedule.sqrt_alpha_cumprod, prev_t) * x_0 + \
                 self.schedule.get_data(self.schedule.sqrt_one_minus_alpha_cumprod, prev_t) * eps

        return x_prev
    
    @torch.no_grad()
    def sample(
        self, 
        model, 
        number_steps, 
        image_shape, 
        device, 
        y = None, 
        guidance_scale = 1, 
        null_class_label = None
    ):
        timesteps = self.make_timesteps(number_steps)

        x = torch.randn(image_shape, device=device)
        batch_size = image_shape[0]
    
        for i in range(number_steps):
            current_t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)
            
            if i + 1 < number_steps:
                prev_t = torch.full((batch_size,), timesteps[i + 1], device=device, dtype=torch.long)
            else:
                prev_t = None

            x = self.ddim_step(model, x, current_t, prev_t, y, guidance_scale, null_class_label)

        return x
