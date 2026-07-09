from .schedules import DiffusionSchedule
import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule
    
    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)

        return self.schedule.get_data(self.schedule.sqrt_alpha_cumprod, t) * x_0 + \
               self.schedule.get_data(self.schedule.sqrt_one_minus_alpha_cumprod, t) * noise, \
               noise
    
    @torch.no_grad()
    def p_sample(self, model, x_t, t, variance_type = "posterior"):
        mean_noise_coefficient = self.schedule.get_data(self.schedule.betas, t) / \
                                 self.schedule.get_data(self.schedule.sqrt_one_minus_alpha_cumprod, t)

        mean_val = self.schedule.get_data(self.schedule.sqrt_inverse_alphas, t) * \
                   (x_t - mean_noise_coefficient * model(x_t, t))
        
        if variance_type == "initial_beta":
            variance = self.schedule.get_data(self.schedule.betas, t)
        elif variance_type == "posterior":
            variance = self.schedule.get_data(self.schedule.posterior_variances, t)
        else:
            raise ValueError(f"Invalid variance type: {variance_type}")

        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return mean_val + nonzero_mask * torch.sqrt(variance) * torch.randn_like(x_t)
    
    @torch.no_grad()
    def sample(self, model, image_shape, device, variance_type = "posterior"):
        x = torch.randn(image_shape, device=device)
        number_steps = self.schedule.number_steps

        for i in range(number_steps - 1, -1, -1):
            timesteps = torch.full((image_shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, timesteps, variance_type)
        
        return x
    
    def training_loss(self, model, x_0, t):
        x_t, noise = self.q_sample(x_0, t)

        prediction = model(x_t, t)

        return F.mse_loss(prediction, noise)