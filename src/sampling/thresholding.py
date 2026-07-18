import torch


def none_threshold(x, **kwargs):
    return x

def static_threshold(x, **kwargs):
    return x.clamp(-1, 1)


def dynamic_threshold(x, quantile, **kwargs):
    batch_size = x.shape[0]
    flatten_information = x.view(batch_size, -1).abs()
    
    threshold = flatten_information.quantile(quantile, dim = -1)
    threshold = threshold.clamp(min = 1)
    threshold = threshold.view(batch_size, 1, 1, 1)
    
    return x.clamp(-threshold, threshold) / threshold


THRESHOLD_TYPES = {
    "none": none_threshold,
    "static": static_threshold,
    "dynamic": dynamic_threshold,
}

def apply_threshold(x, mode = "none", quantile = 0.995):
    if mode not in THRESHOLD_TYPES:
        raise ValueError(f"Invalid threshold mode: {mode}")

    return THRESHOLD_TYPES[mode](x, quantile = quantile)