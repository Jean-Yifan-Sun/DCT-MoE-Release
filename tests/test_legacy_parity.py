"""Optional comparison against an author checkout: set DCT_MOE_LEGACY_ROOT."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dct_moe.codec import DCTCodec
from dct_moe.model import UViT_greyscale_MoE
from dct_moe.diffusion import diffusion_loss

LEGACY = os.environ.get('DCT_MOE_LEGACY_ROOT')


@unittest.skipUnless(LEGACY, 'Set DCT_MOE_LEGACY_ROOT to compare with the original research checkout')
class LegacyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, LEGACY)
        from libs.uvit import UViT_greyscale_MoE as Original
        from datasets import DCT_4Y
        from utils import DCT_to_greyscale
        from DCT_utils import reverse_zigzag_order
        import sde
        cls.original_class = Original
        cls.original_data = DCT_4Y
        cls.original_decode = staticmethod(DCT_to_greyscale)
        cls.reverse = staticmethod(reverse_zigzag_order)
        cls.sde = sde

    def test_dct_tokens_and_decode_match_legacy(self):
        for block, low in [(4, 16), (4, 12), (8, 16), (8, 48)]:
            with tempfile.TemporaryDirectory() as directory:
                size = block * 4
                image = np.random.default_rng(17).integers(0, 256, (size, size), dtype=np.uint8)
                Image.fromarray(image).save(Path(directory) / 'image.png')
                codec = DCTCodec(size, block, low, 512.)
                old = self.original_data(directory, img_sz=size, tokens=codec.tokens,
                                         low_freqs=low, block_sz=block, Y_bound=[512.])
                torch.manual_seed(1)  # Legacy RandomHorizontalFlip chooses no flip.
                np.testing.assert_allclose(codec.encode(image), old[0].numpy(), atol=1e-7)
                tokens = np.random.default_rng(29).normal(0, .1, codec.shape)
                original = self.original_decode(tokens, tokens=codec.tokens, low_freqs=low,
                    block_sz=block, reverse_order=self.reverse(block), resolution=size, Y_bound=[512.])
                np.testing.assert_array_equal(codec.decode(tokens), original)

    def test_full_resolution_float32_samples_decode_pixel_exactly(self):
        codec = DCTCodec(256, 8, 16, 1016.)
        tokens = np.random.default_rng(1).normal(0, .1, codec.shape).astype(np.float32)
        original = self.original_decode(tokens, tokens=codec.tokens, low_freqs=16,
            block_sz=8, reverse_order=self.reverse(8), resolution=256, Y_bound=[1016.])
        np.testing.assert_array_equal(codec.decode(tokens), original)

    def test_legacy_backbone_with_approved_routing_matches(self):
        settings = dict(tokens=4, low_freqs=4, embed_dim=32, depth=2, num_heads=4,
                        mlp_time_embed=True, MoE=dict(depth=1, num_experts=4, top_k=2, noise_eps=.01))
        original, released = self.original_class(**settings), UViT_greyscale_MoE(**settings)
        released.load_state_dict(original.state_dict(), strict=True)
        # The author intentionally corrected routing in the release. Adapt only
        # the old router output weights; compare the remaining legacy backbone.
        for module in original.modules():
            if module.__class__.__name__ != 'TopKRouter':
                continue
            old_forward = module.forward
            def probability_forward(x, router=module, old_forward=old_forward):
                state = torch.get_rng_state()
                mask, indices, aux = old_forward(x)
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(state)
                    logits = router.gate(x.reshape(-1, router.d_model))
                    if router.training and router.noise_eps > 0:
                        logits = logits + torch.randn_like(logits) * router.noise_eps
                    probabilities = logits.clamp(-100., 100.).float().softmax(-1).reshape_as(mask)
                return mask * probabilities, indices, aux
            module.forward = probability_forward

        x, t = torch.randn(2, 4, 16), torch.tensor([100., 500.])
        for training in (False, True):
            results = []
            for model in (original, released):
                model.train(training)
                model.zero_grad(set_to_none=True)
                torch.manual_seed(41)
                results.append(model(x, t))
            for a, b in zip(*results):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
            if training:
                for prediction, aux in results:
                    (prediction.square().mean() + .001 * aux).backward()
                for (name, a), (_, b) in zip(original.named_parameters(), released.named_parameters()):
                    if a.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-6, msg=name)

    def test_training_objective_matches_legacy(self):
        model = UViT_greyscale_MoE(tokens=4, low_freqs=4, embed_dim=32, depth=2, num_heads=4)
        clean = torch.randn(2, 4, 16)
        score = self.sde.ScoreModel(model, 'noise_pred', self.sde.VPSDE(SNR_scale=4.))
        criterion = self.sde.EntropyWeightedMSELoss(torch.tensor([1., 2., 3., 4.]).repeat(4), temperature=0., dim=1)
        torch.manual_seed(39)
        old = self.sde.LSimple(score, clean, use_moe=True, criterion=criterion)
        torch.manual_seed(39)
        new = diffusion_loss(model, clean, 4., [1., 2., 3., 4.])
        for a, b in zip(old, new):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
