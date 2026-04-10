"""Tests for torch_relativistic.utils — the math primitives."""

import math

import torch

from torch_relativistic.utils import (
    calculate_gamma,
    clamp_velocity,
    lorentz_contraction,
    relativistic_doppler_factor,
    terrell_rotation_angle,
    time_dilation,
    velocity_addition,
)


class TestClampVelocity:
    def test_within_range_unchanged(self):
        v = torch.tensor([0.0, 0.3, 0.7, 0.95])
        out = clamp_velocity(v)
        assert torch.allclose(out, v)

    def test_above_max_clamped(self):
        v = torch.tensor([1.0, 1.5, 2.0])
        out = clamp_velocity(v, max_velocity=0.999)
        assert torch.all(out <= 0.999)

    def test_below_min_clamped(self):
        v = torch.tensor([-0.5, -1.0])
        out = clamp_velocity(v, min_velocity=0.0)
        assert torch.all(out >= 0.0)


class TestCalculateGamma:
    def test_gamma_at_zero_velocity(self):
        """γ(0) = 1"""
        v = torch.tensor(0.0)
        gamma = calculate_gamma(v)
        assert torch.allclose(gamma, torch.tensor(1.0), atol=1e-4)

    def test_gamma_monotonic_in_velocity(self):
        """γ is strictly increasing in |v| when evaluated as scalars."""
        # calculate_gamma treats the last dim of its input as velocity
        # components, so to get "six independent scalar velocities" we
        # have to call it six times.
        vs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        gammas = [calculate_gamma(torch.tensor(v)).item() for v in vs]
        diffs = [b - a for a, b in zip(gammas, gammas[1:])]
        assert all(d > 0 for d in diffs), f"γ not strictly increasing: {gammas}"

    def test_gamma_finite_at_max(self):
        """γ should stay finite thanks to clamping + eps, even at v=1"""
        v = torch.tensor(1.0)
        gamma = calculate_gamma(v)
        assert torch.isfinite(gamma)

    def test_gamma_no_nan_for_negative_velocity(self):
        v = torch.tensor(-0.7)
        gamma = calculate_gamma(v)
        assert torch.isfinite(gamma)
        assert gamma.item() > 0

    def test_gamma_textbook_value(self):
        """γ(0.6) = 1.25 (classic worked example)"""
        v = torch.tensor(0.6)
        gamma = calculate_gamma(v)
        assert abs(gamma.item() - 1.25) < 1e-3

    def test_gamma_vector_input_is_magnitude(self):
        """
        API contract: calculate_gamma on a rank >= 1 tensor interprets
        the last dim as 3-vector velocity components and reduces over it.
        So gamma([0.6, 0, 0]) == gamma(0.6) == 1.25.

        This is a non-obvious convention and worth locking down in a test.
        """
        v_scalar = torch.tensor(0.6)
        v_vector = torch.tensor([0.6, 0.0, 0.0])
        g_scalar = calculate_gamma(v_scalar).item()
        g_vector = calculate_gamma(v_vector).item()
        assert abs(g_scalar - g_vector) < 1e-4
        assert abs(g_vector - 1.25) < 1e-3

    def test_gamma_vector_batch_shape(self):
        """Batched velocity vectors of shape (N, 3) -> gamma of shape (N, 1)."""
        v = torch.rand(4, 3) * 0.5  # four 3D velocity vectors, |v| < 1
        gamma = calculate_gamma(v)
        assert gamma.shape == (4, 1)
        assert torch.all(torch.isfinite(gamma))
        assert torch.all(gamma >= 1.0)  # γ is always >= 1


class TestLorentzContraction:
    def test_no_contraction_at_zero_velocity(self):
        """L(v=0) = L0"""
        L = torch.tensor(10.0)
        v = torch.tensor(0.0)
        contracted = lorentz_contraction(L, v)
        assert torch.allclose(contracted, L, atol=1e-3)

    def test_contraction_shrinks(self):
        L = torch.tensor(10.0)
        v = torch.tensor(0.6)  # γ = 1.25
        contracted = lorentz_contraction(L, v)
        # L / γ = 10 / 1.25 = 8
        assert abs(contracted.item() - 8.0) < 1e-2


class TestTimeDilation:
    def test_no_dilation_at_zero(self):
        t = torch.tensor(5.0)
        v = torch.tensor(0.0)
        assert torch.allclose(time_dilation(t, v), t, atol=1e-3)

    def test_dilation_stretches(self):
        """moving clocks tick slower → observer sees longer interval"""
        t = torch.tensor(1.0)
        v = torch.tensor(0.6)  # γ = 1.25
        dilated = time_dilation(t, v)
        assert dilated.item() > t.item()
        assert abs(dilated.item() - 1.25) < 1e-2


class TestVelocityAddition:
    def test_scalar_zero_identity(self):
        """v ⊕ 0 = v"""
        v1 = torch.tensor(0.5)
        v2 = torch.tensor(0.0)
        assert torch.allclose(velocity_addition(v1, v2), v1, atol=1e-4)

    def test_scalar_sub_c(self):
        """Relativistic sum of two sub-c velocities stays below c."""
        v1 = torch.tensor(0.8)
        v2 = torch.tensor(0.8)
        result = velocity_addition(v1, v2)
        # Classical 0.8+0.8=1.6, relativistic gives ~0.9756
        assert result.item() < 1.0
        assert abs(result.item() - (1.6 / 1.64)) < 1e-3

    def test_scalar_commutative(self):
        v1 = torch.tensor(0.3)
        v2 = torch.tensor(0.5)
        assert torch.allclose(velocity_addition(v1, v2), velocity_addition(v2, v1))


class TestTerrellRotationAngle:
    def test_no_rotation_at_rest(self):
        v = torch.tensor(0.0)
        angle = terrell_rotation_angle(v)
        assert torch.allclose(angle, torch.tensor(0.0), atol=1e-6)

    def test_rotation_monotonic(self):
        vs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        angles = torch.stack([terrell_rotation_angle(v) for v in vs])
        assert torch.all(angles[1:] - angles[:-1] > 0)


class TestDopplerFactor:
    def test_head_on_has_value(self):
        v = torch.tensor(0.5)
        d = relativistic_doppler_factor(v)
        assert torch.isfinite(d)
        assert d.item() > 1.0  # approaching → blueshift

    def test_transverse_is_pure_gamma(self):
        """At θ = π/2 the Doppler factor reduces to γ (transverse Doppler effect)."""
        v = torch.tensor(0.6)
        theta = torch.tensor(math.pi / 2)
        d = relativistic_doppler_factor(v, observer_angle=theta)
        gamma = calculate_gamma(v)
        assert abs(d.item() - gamma.item()) < 1e-4
