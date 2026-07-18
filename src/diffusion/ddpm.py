from .base import DiffusionBase
import torch
from sampling.cfg import cfg_predict
from sampling.thresholding import apply_threshold

from class_registry import class_registry


@class_registry.add_to_registry("ddpm")
class DDPM(DiffusionBase):
    @torch.no_grad()
    def p_sample(
        self, 
        model, 
        x_t, 
        t, 
        y,
        guidance_scale = 1,
        null_class_label = None,
        variance_type = "posterior",
        threshold = "none",
        threshold_quantile = 0.995
    ):
        model_prediction = cfg_predict(model, x_t, t, y, guidance_scale, null_class_label)

        x_0 = self.predict_x0_from_eps(x_t, t, model_prediction)
        x_0 = apply_threshold(x_0, threshold, threshold_quantile)

        mean_val = self.schedule.get_data(self.schedule.posterior_coef_x0, t) * x_0 + \
                   self.schedule.get_data(self.schedule.posterior_coef_xt, t) * x_t
        
        if variance_type == "initial_beta":
            variance = self.schedule.get_data(self.schedule.betas, t)
        elif variance_type == "posterior":
            variance = self.schedule.get_data(self.schedule.posterior_variances, t)
        else:
            raise ValueError(f"Invalid variance type: {variance_type}")

        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return mean_val + nonzero_mask * torch.sqrt(variance) * torch.randn_like(x_t)
    
    @torch.no_grad()
    def sample(
        self, 
        model, 
        image_shape, 
        device, 
        y, 
        guidance_scale = 1, 
        null_class_label = None, 
        variance_type = "posterior",
        threshold = "none",
        threshold_quantile = 0.995
    ):
        x = torch.randn(image_shape, device=device)
        number_steps = self.schedule.number_steps

        for i in range(number_steps - 1, -1, -1):
            timesteps = torch.full((image_shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(
                model, 
                x, 
                timesteps, 
                y, 
                guidance_scale, 
                null_class_label, 
                variance_type,
                threshold,
                threshold_quantile
            )

        return x
