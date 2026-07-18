import torch


@torch.no_grad()
def cfg_predict(model, x, t, y, guidance_scale = 1, null_class_label = None):
    if guidance_scale == 1:
        return model(x, t, y)

    class_prediction = model(x, t, y)
    uncond_prediction = model(x, t, torch.full_like(y, null_class_label))

    return uncond_prediction + guidance_scale * (class_prediction - uncond_prediction)