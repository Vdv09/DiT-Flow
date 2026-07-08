from diffusion.schedules import DiffusionSchedule
import torch


def test_linear_schedule_alpha_cumprod_decreasing():
    schedule = DiffusionSchedule.linear_schedule()

    assert torch.all(schedule.alpha_cumprod[:-1] > schedule.alpha_cumprod[1:])