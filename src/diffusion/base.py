from .schedules import DiffusionSchedule
import torch
import torch.nn.functional as F


class DiffusionBase:
    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule
    
    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)

        return self.schedule.get_data(self.schedule.sqrt_alpha_cumprod, t) * x_0 + \
               self.schedule.get_data(self.schedule.sqrt_one_minus_alpha_cumprod, t) * noise, \
               noise
    
    def training_loss(self, model, x_0, t, y = None):
        x_t, noise = self.q_sample(x_0, t)

        prediction = model(x_t, t, y)

        return F.mse_loss(prediction, noise)
    
    def predict_x0_from_eps(self, x_t, t, noise):
        return (x_t - self.schedule.get_data(self.schedule.sqrt_one_minus_alpha_cumprod, t) * noise) / \
               self.schedule.get_data(self.schedule.sqrt_alpha_cumprod, t)