"""Numerical contracts for the released DCT + MoE method."""
import unittest

import numpy as np
import torch

from dct_moe.codec import DCTCodec
from dct_moe.model import UViT_greyscale_MoE, Block_MoE, TopKRouter
from dct_moe.diffusion import diffusion_loss, sample_tokens
from dct_moe.dpm_solver import NoiseScheduleVP


class CodecTests(unittest.TestCase):
    def test_full_coefficient_roundtrip_and_spatial_order(self):
        image = np.random.default_rng(4).integers(0, 256, (16, 16), dtype=np.uint8)
        codec = DCTCodec(resolution=16, block_size=4, low_freqs=16, bound=512)
        tokens = codec.encode(image)
        self.assertEqual(tokens.shape, (4, 64))
        np.testing.assert_allclose(codec.decode(tokens, as_uint8=False), image, atol=4e-5)
        # Four adjacent constant blocks have distinguishable DC coefficients.
        image = np.zeros((16, 16), dtype=np.uint8)
        image[:4, :4], image[:4, 4:8] = 10, 20
        image[4:8, :4], image[4:8, 4:8] = 30, 40
        np.testing.assert_allclose(codec.encode(image)[0].reshape(4, 16)[:, 0],
                                   (np.array([10, 20, 30, 40]) - 128) * 4 / 512)

    def test_truncation_reconstructs_constant_image(self):
        codec = DCTCodec(16, 4, 1, 512)
        image = np.full((16, 16), 90, dtype=np.uint8)
        np.testing.assert_allclose(codec.decode(codec.encode(image), as_uint8=False), image, atol=1e-5)

    def test_invalid_geometry_and_input_rejected(self):
        for args in [(12, 4, 16, 512), (16, 4, 17, 512), (16, 4, 16, 0)]:
            with self.assertRaises(ValueError):
                DCTCodec(*args)
        codec = DCTCodec(16, 4, 16, 512)
        with self.assertRaises(ValueError):
            codec.encode(np.zeros((8, 8), dtype=np.uint8))
        with self.assertRaises(ValueError):
            codec.decode(np.zeros((4, 63)))


class ModelTests(unittest.TestCase):
    def make_model(self, checkpoint=False):
        return UViT_greyscale_MoE(tokens=4, low_freqs=4, embed_dim=32,
            depth=2, num_heads=4, mlp_time_embed=True, use_checkpoint=checkpoint,
            MoE={'depth': 1, 'num_experts': 4, 'top_k': 2, 'noise_eps': 0.01})

    def test_sparse_router_selection_and_gradient(self):
        router = TopKRouter(8, 4, 2, noise_eps=0).eval()
        x = torch.randn(2, 5, 8, requires_grad=True)
        weights, indices, aux = router(x)
        probabilities = router.gate(x).clamp(-100, 100).softmax(-1)
        expected = torch.zeros_like(probabilities).scatter(-1, indices, probabilities.gather(-1, indices))
        torch.testing.assert_close(weights, expected)
        self.assertTrue((weights.sum(-1) <= 1 + 1e-6).all())
        self.assertEqual(torch.count_nonzero(weights).item(), 20)
        (weights * torch.arange(4.)).sum().backward(retain_graph=True)
        self.assertGreater(router.gate.weight.grad.abs().sum().item(), 0)
        self.assertEqual(indices.shape, (2, 5, 2))
        aux.backward()
        self.assertTrue(torch.isfinite(router.gate.weight.grad).all())

    def test_both_moe_blocks_use_configured_noise(self):
        for epsilon in (0., .37):
            model = UViT_greyscale_MoE(tokens=4, low_freqs=4, embed_dim=32,
                depth=2, num_heads=4, MoE={'noise_eps': epsilon})
            self.assertEqual(model.in_blocks[0].router.noise_eps, epsilon)
            self.assertEqual(model.out_blocks[-1].router.noise_eps, epsilon)

    def test_model_boundary_moe_backward_and_deterministic_eval(self):
        model = self.make_model()
        self.assertIsInstance(model.in_blocks[0], Block_MoE)
        self.assertIsInstance(model.out_blocks[-1], Block_MoE)
        x, t = torch.randn(2, 4, 16), torch.tensor([100., 500.])
        prediction, aux = model(x, t)
        self.assertEqual(prediction.shape, x.shape)
        (prediction.square().mean() + .001 * aux).backward()
        self.assertIsNotNone(model.in_blocks[0].router.gate.weight.grad)
        model.eval()
        with torch.no_grad():
            a, _ = model(x, t)
            b, _ = model(x, t)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(a.requires_grad)

    def test_checkpointed_gradients_match(self):
        ordinary, checkpointed = self.make_model(), self.make_model(True)
        checkpointed.load_state_dict(ordinary.state_dict())
        x, t = torch.randn(2, 4, 16), torch.tensor([100., 500.])
        for model in (ordinary, checkpointed):
            torch.manual_seed(31)
            p, aux = model(x, t)
            (p.square().mean() + .001 * aux).backward()
        for a, b in zip(ordinary.parameters(), checkpointed.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad)

    def test_diffusion_backward(self):
        model = self.make_model()
        main, aux = diffusion_loss(model, torch.randn(2, 4, 16), 4., [1., 2., 3., 4.])
        (main + .001 * aux).backward()
        self.assertTrue(torch.isfinite(main))

    def test_solver_uses_requested_nfe_and_is_repeatable(self):
        class ZeroNoise(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x, timesteps):
                self.calls += 1
                return torch.zeros_like(x), x.new_zeros(())
        model, noise = ZeroNoise().eval(), torch.randn(1, 4, 16)
        a = sample_tokens(model, noise.clone(), snr_scale=4., steps=5)
        self.assertEqual(model.calls, 5)
        b = sample_tokens(model, noise.clone(), snr_scale=4., steps=5)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(a).all())

    def test_schedule_inverse(self):
        schedule = NoiseScheduleVP(SNR_scale=4.)
        t = torch.linspace(.01, 1., 20)
        torch.testing.assert_close(schedule.inverse_lambda(schedule.marginal_lambda(t)), t,
                                   atol=3e-5, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
