import pytest
import torch

from ple.predictors import identity, residual_extrapolation, velocity


def _trajectory(steps: int, shape=(2, 7, 32)) -> list[torch.Tensor]:
    torch.manual_seed(0)
    start, step = torch.randn(shape), torch.randn(shape)
    return [start + i * step for i in range(steps)]


@pytest.mark.parametrize("predict", [identity, residual_extrapolation, velocity])
def test_preserves_shape_dtype_device(predict):
    history = _trajectory(3)
    out = predict(history)
    assert out.shape == history[-1].shape
    assert out.dtype == history[-1].dtype
    assert out.device == history[-1].device


def test_residual_extrapolation_is_exact_on_constant_updates():
    history = _trajectory(4)
    torch.testing.assert_close(residual_extrapolation(history[:3]), history[3])


def test_velocity_second_order_term():
    h0, h1, h2 = (torch.full((3,), v) for v in (0.0, 1.0, 3.0))
    # v_l = 2, v_(l-1) = 1 → 3 + 0.5*2 + 0.25*1
    torch.testing.assert_close(velocity([h0, h1, h2], alpha=0.5, beta=0.25), torch.full((3,), 4.25))


def test_velocity_rejects_short_history():
    with pytest.raises(ValueError):
        velocity(_trajectory(2), beta=0.5)
