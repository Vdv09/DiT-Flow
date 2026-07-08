import torch


class DiffusionSchedule:
    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1 - self.betas

        self.alpha_cumprod = torch.cumprod(self.alphas, dim = 0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)

        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])

        self.sqrt_inverse_alphas = torch.sqrt(1 / self.alphas)

        self.posterior_variances = (1 - self.alpha_cumprod_prev) / (1 - self.alpha_cumprod) * self.betas

        self._buffer_names = [
            'betas',
            'alphas',
            'alpha_cumprod',
            'sqrt_alpha_cumprod',
            'sqrt_one_minus_alpha_cumprod',
            'alpha_cumprod_prev',
            'sqrt_inverse_alphas',
            'posterior_variances',
        ]
    
    @classmethod
    def linear_schedule(cls, min_beta = 1e-4, max_beta = 0.02, number_steps = 1000):
        betas = torch.linspace(min_beta, max_beta, number_steps)
        return cls(betas)
    
    @classmethod
    def cosine_schedule(cls, number_steps = 1000, s = 8e-3):
        indicies = torch.arange(number_steps + 1)
        f_values = torch.cos((indicies / number_steps + s) / (1 + s) * torch.pi / 2) ** 2

        alpha_cumprod = f_values / f_values[0]
        betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]

        betas = torch.clip(betas, min = 1e-4, max = 0.999)

        return cls(betas)
    
    def get_data(self, data, timesteps):
        needed_information = data[timesteps]

        return needed_information.view(-1, 1, 1, 1)
    
    def to(self, device):
        for name in self._buffer_names:
            buffer = getattr(self, name)
            setattr(self, name, buffer.to(device))

        return self