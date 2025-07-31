"""
Unit tests for model module.
"""

import unittest
import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model import MiniGPT


class TestMiniGPT(unittest.TestCase):
    """Test cases for MiniGPT model."""
    
    def setUp(self):
        """Set up test model."""
        self.vocab_size = 100
        self.context_len = 8
        self.embed_dim = 32
        self.num_heads = 2
        self.num_layers = 2
        
        self.model = MiniGPT(
            vocab_size=self.vocab_size,
            context_len=self.context_len,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Check model components exist
        self.assertIsInstance(self.model.token_embed, nn.Embedding)
        self.assertIsInstance(self.model.pos_embed, nn.Parameter)
        self.assertIsInstance(self.model.transformer, nn.TransformerEncoder)
        self.assertIsInstance(self.model.ln_final, nn.LayerNorm)
        self.assertIsInstance(self.model.output, nn.Linear)
        
        # Check dimensions
        self.assertEqual(self.model.token_embed.num_embeddings, self.vocab_size)
        self.assertEqual(self.model.token_embed.embedding_dim, self.embed_dim)
        self.assertEqual(self.model.pos_embed.shape, (1, self.context_len, self.embed_dim))
        self.assertEqual(self.model.output.in_features, self.embed_dim)
        self.assertEqual(self.model.output.out_features, self.vocab_size)
    
    def test_forward_pass(self):
        """Test forward pass with different input shapes."""
        batch_size = 4
        seq_len = 6
        
        # Create test input
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits = self.model(x)
        
        # Check output shape
        expected_shape = (batch_size, self.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        batch_size = 2
        
        for seq_len in [1, 3, self.context_len]:
            with self.subTest(seq_len=seq_len):
                x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
                logits = self.model(x)
                
                expected_shape = (batch_size, self.vocab_size)
                self.assertEqual(logits.shape, expected_shape)
    
    def test_max_sequence_length(self):
        """Test model at maximum sequence length."""
        batch_size = 2
        x = torch.randint(0, self.vocab_size, (batch_size, self.context_len))
        
        logits = self.model(x)
        expected_shape = (batch_size, self.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
    
    def test_parameter_count(self):
        """Test parameter counting methods."""
        total_params = self.model.get_num_params()
        trainable_params = self.model.get_num_trainable_params()
        
        # Both should be positive
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        
        # For a model without frozen parameters, they should be equal
        self.assertEqual(total_params, trainable_params)
        
        # Verify manual count matches
        manual_count = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, manual_count)
    
    def test_model_modes(self):
        """Test training and evaluation modes."""
        # Test training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        # Create dummy input and target
        batch_size = 2
        seq_len = 4
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        target = torch.randint(0, self.vocab_size, (batch_size,))
        
        # Forward pass
        logits = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
    
    def test_device_compatibility(self):
        """Test model device compatibility."""
        # Test CPU
        x = torch.randint(0, self.vocab_size, (2, 4))
        logits = self.model(x)
        self.assertEqual(logits.device.type, 'cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = self.model.cuda()
            x_gpu = x.cuda()
            logits_gpu = model_gpu(x_gpu)
            self.assertEqual(logits_gpu.device.type, 'cuda')
    
    def test_different_model_sizes(self):
        """Test different model configurations."""
        configs = [
            {'vocab_size': 50, 'context_len': 4, 'embed_dim': 16, 'num_heads': 1, 'num_layers': 1},
            {'vocab_size': 200, 'context_len': 16, 'embed_dim': 64, 'num_heads': 4, 'num_layers': 3},
        ]
        
        for config in configs:
            with self.subTest(config=config):
                model = MiniGPT(**config)
                
                # Test forward pass
                x = torch.randint(0, config['vocab_size'], (2, config['context_len']))
                logits = model(x)
                
                expected_shape = (2, config['vocab_size'])
                self.assertEqual(logits.shape, expected_shape)
    
    def test_attention_mask(self):
        """Test that attention masking works (causal attention)."""
        # This is a bit tricky to test directly, but we can check
        # that the model produces different outputs for different
        # input orders, which suggests the attention is working
        
        batch_size = 1
        seq_len = 4
        
        # Create two different sequences
        x1 = torch.tensor([[1, 2, 3, 4]])
        x2 = torch.tensor([[4, 3, 2, 1]])
        
        with torch.no_grad():
            logits1 = self.model(x1)
            logits2 = self.model(x2)
        
        # Outputs should be different for different inputs
        self.assertFalse(torch.allclose(logits1, logits2, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
