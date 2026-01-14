"""
Model Tests
============
Unit tests for Squeezeformer model components.

Uses mindspore.ops for random data generation.
"""

import unittest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_mindspore(self):
        """Test MindSpore import."""
        try:
            import mindspore
            import mindspore.numpy as mnp
            import mindspore.ops as ops
            self.assertTrue(True)
        except ImportError:
            self.skipTest("MindSpore not installed")
    
    def test_import_models(self):
        """Test model imports."""
        try:
            from models import (
                ISLRModelV2, ISLRModelV2Lite, create_islr_model_v2,
                SqueezeformerBlock, SqueezeformerEncoder,
                RotaryEmbedding, SwiGLU, CrossRegionAttention
            )
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Import error: {e}")


class TestSqueezeformerComponents(unittest.TestCase):
    """Test Squeezeformer model components."""
    
    @classmethod
    def setUpClass(cls):
        try:
            import mindspore
            import mindspore.ops as ops
            from mindspore import Tensor
            cls.mindspore_available = True
            cls.ops = ops
        except ImportError:
            cls.mindspore_available = False
    
    def setUp(self):
        if not self.mindspore_available:
            self.skipTest("MindSpore not installed")
    
    def test_rotary_embedding(self):
        """Test RotaryEmbedding."""
        from models import RotaryEmbedding
        from mindspore import Tensor
        import mindspore.ops as ops
        
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        
        batch, heads, seq_len, head_dim = 2, 8, 100, 64
        q = Tensor(ops.StandardNormal()((batch, heads, seq_len, head_dim)).asnumpy().astype('float32'))
        k = Tensor(ops.StandardNormal()((batch, heads, seq_len, head_dim)).asnumpy().astype('float32'))
        
        q_rot, k_rot = rope(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
    
    def test_swiglu_ffn(self):
        """Test SwiGLU FFN."""
        from models import SwiGLUFFN
        from mindspore import Tensor
        import mindspore.ops as ops
        
        ffn = SwiGLUFFN(dim=256, hidden_dim=512)
        x = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_squeezeformer_block(self):
        """Test SqueezeformerBlock."""
        from models import SqueezeformerBlock
        from mindspore import Tensor
        import mindspore.ops as ops
        
        block = SqueezeformerBlock(dim=256, num_heads=8)
        x = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
        out = block(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_cross_attention(self):
        """Test CrossRegionAttention."""
        from models import CrossRegionAttention
        from mindspore import Tensor
        import mindspore.ops as ops
        
        cross_attn = CrossRegionAttention(dim=256, num_heads=4)
        hands = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
        face = Tensor(ops.StandardNormal()((2, 100, 256)).asnumpy().astype('float32'))
        
        hand_out, face_out = cross_attn(hands, face)
        self.assertEqual(hand_out.shape, hands.shape)
        self.assertEqual(face_out.shape, face.shape)


class TestSqueezeformerModel(unittest.TestCase):
    """Test complete Squeezeformer models."""
    
    @classmethod
    def setUpClass(cls):
        try:
            import mindspore
            cls.mindspore_available = True
        except ImportError:
            cls.mindspore_available = False
    
    def setUp(self):
        if not self.mindspore_available:
            self.skipTest("MindSpore not installed")
    
    def test_islr_model_v2(self):
        """Test ISLRModelV2 output shape."""
        from models import ISLRModelV2
        from mindspore import Tensor
        import mindspore.ops as ops
        
        model = ISLRModelV2(
            input_dim=708,
            num_classes=250,
            num_blocks=2  # Reduced for testing
        )
        
        x = Tensor(ops.StandardNormal()((2, 100, 708)).asnumpy().astype('float32'))
        out = model(x)
        
        self.assertEqual(out.shape, (2, 250))
    
    def test_islr_model_v2_lite(self):
        """Test ISLRModelV2Lite output shape."""
        from models import ISLRModelV2Lite
        from mindspore import Tensor
        import mindspore.ops as ops
        
        model = ISLRModelV2Lite(num_classes=250)
        
        x = Tensor(ops.StandardNormal()((2, 100, 708)).asnumpy().astype('float32'))
        out = model(x)
        
        self.assertEqual(out.shape, (2, 250))
    
    def test_model_parameters(self):
        """Test model has trainable parameters."""
        from models import ISLRModelV2
        
        model = ISLRModelV2(num_blocks=2)
        num_params = model.get_num_params()
        
        # Model should have > 1M parameters
        self.assertGreater(num_params, 1_000_000)


class TestVocabulary(unittest.TestCase):
    """Test vocabulary loading."""
    
    def test_load_vocabulary(self):
        """Test vocabulary file exists and loads correctly."""
        import json
        
        vocab_path = Path(__file__).parent.parent / "dataset" / "sign_vocabulary.json"
        
        self.assertTrue(vocab_path.exists(), f"Vocabulary file not found: {vocab_path}")
        
        with open(vocab_path) as f:
            data = json.load(f)
        
        self.assertEqual(data["vocabulary_size"], 250)
        self.assertEqual(len(data["signs"]), 250)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImports))
    suite.addTests(loader.loadTestsFromTestCase(TestSqueezeformerComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestSqueezeformerModel))
    suite.addTests(loader.loadTestsFromTestCase(TestVocabulary))
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
