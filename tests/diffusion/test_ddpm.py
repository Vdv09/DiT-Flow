from diffusion.ddpm import DDPM
from diffusion.schedules import DiffusionSchedule

import torch


def test_q_sample_shape():
    schedule = DiffusionSchedule.linear_schedule()

    ddpm = DDPM(schedule)

    q_sample, _ = ddpm.q_sample(torch.randn(1, 3, 32, 32), torch.tensor([0]))

    assert q_sample.shape == (1, 3, 32, 32)


def test_q_sample_not_equal_to_original_image():
    schedule = DiffusionSchedule.linear_schedule()

    ddpm = DDPM(schedule)

    original_image = torch.randn(1, 3, 32, 32)
    q_sample, _ = ddpm.q_sample(original_image, torch.tensor([0]))

    assert not torch.allclose(q_sample, original_image)
