from diffusion.ddim import DDIM
from diffusion.schedules import DiffusionSchedule


def test_make_timesteps():
    schedule = DiffusionSchedule.linear_schedule()

    ddim = DDIM(schedule)

    timesteps = ddim.make_timesteps(10)

    assert timesteps.shape == (10,)
    assert timesteps.min() == 0
    assert timesteps.max() == schedule.number_steps - 1