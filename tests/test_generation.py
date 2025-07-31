"""
Unit tests for generation module.
"""

import unittest
import torch
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.generation import sample_top_k, sample_top_p, generate, generate_from_string
from src.model import MiniGPT
from src.tokenizer import build_vocab


class TestGeneration(unittest.TestCase):
    """Test cases for generation functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create simple vocabulary
        self.words = ["hello", "world", "test", "sample", "text"]
        self.word2idx, self.idx2word = build_vocab(self.words, min_freq=1)
        self.vocab_size = len(self.word2idx)
        
        # Create small test model
        self.context_len = 4
        self.model = MiniGPT(
            vocab_size=self.vocab_size,
            context_len=self.context_len,
            embed_dim=16,
            num_heads=2,
            num_layers=1
        )
        self.model.eval()
    
    def test_sample_top_k(self):
        """Test top-k sampling."""
        # Create test probability distribution
        probs = torch.tensor([0.1, 0.3, 0.2, 0.15, 0.25])
        
        # Test different k values
        for k in [1, 2, 3, len(probs)]:
            with self.subTest(k=k):
                sampled_idx = sample_top_k(probs, k=k)
                
                # Should return valid index
                self.assertGreaterEqual(sampled_idx, 0)
                self.assertLess(sampled_idx, len(probs))
                
                # For k=1, should always return highest probability index
                if k == 1:
                    expected_idx = torch.argmax(probs).item()
                    self.assertEqual(sampled_idx, expected_idx)
    
    def test_sample_top_p(self):
        """Test top-p (nucleus) sampling."""
        # Create test probability distribution
        probs = torch.tensor([0.1, 0.4, 0.3, 0.1, 0.1])
        
        # Test different p values
        for p in [0.1, 0.5, 0.9, 1.0]:
            with self.subTest(p=p):
                sampled_idx = sample_top_p(probs, p=p)
                
                # Should return valid index
                self.assertGreaterEqual(sampled_idx, 0)
                self.assertLess(sampled_idx, len(probs))
    
    def test_generate(self):
        """Test text generation from token sequence."""
        # Create seed tokens
        seed = [self.word2idx['hello'], self.word2idx['world']]
        
        # Generate text with safe top_k
        generated = generate(
            self.model, seed, self.idx2word, self.word2idx,
            steps=5, temperature=1.0, top_k=min(10, self.vocab_size)
        )
        
        # Should return a string
        self.assertIsInstance(generated, str)
        
        # Should contain some words (split by spaces)
        words = generated.split()
        self.assertGreater(len(words), 0)
        self.assertLessEqual(len(words), 5)  # Should not exceed steps
    
    def test_generate_from_string(self):
        """Test text generation from string prompt."""
        seed_text = "hello world"
        
        generated = generate_from_string(
            self.model, seed_text, self.word2idx, self.idx2word,
            context_len=self.context_len, steps=3, temperature=1.0, top_k=self.vocab_size
        )
        
        # Should return a string
        self.assertIsInstance(generated, str)
        
        # Should contain words
        words = generated.split()
        self.assertGreater(len(words), 0)
    
    def test_different_temperatures(self):
        """Test generation with different temperatures."""
        seed = [self.word2idx['hello']] * self.context_len
        
        # Test different temperatures
        for temperature in [0.1, 1.0, 2.0]:
            with self.subTest(temperature=temperature):
                generated = generate(
                    self.model, seed, self.idx2word, self.word2idx,
                    steps=3, temperature=temperature, top_k=self.vocab_size
                )
                
                self.assertIsInstance(generated, str)
                # With low temperature, generation should be more deterministic
                # With high temperature, more random (harder to test deterministically)
    
    def test_different_top_k_values(self):
        """Test generation with different top-k values."""
        seed = [self.word2idx['hello']] * self.context_len
        
        for top_k in [1, 2, min(5, self.vocab_size)]:
            with self.subTest(top_k=top_k):
                generated = generate(
                    self.model, seed, self.idx2word, self.word2idx,
                    steps=3, temperature=1.0, top_k=top_k
                )
                
                self.assertIsInstance(generated, str)
    
    def test_top_p_generation(self):
        """Test generation with top-p sampling."""
        seed = [self.word2idx['hello']] * self.context_len
        
        for top_p in [0.1, 0.5, 0.9]:
            with self.subTest(top_p=top_p):
                generated = generate(
                    self.model, seed, self.idx2word, self.word2idx,
                    steps=3, temperature=1.0, top_k=None, top_p=top_p
                )
                
                self.assertIsInstance(generated, str)
    
    def test_unknown_words_in_seed(self):
        """Test generation with unknown words in seed."""
        # Use a word not in vocabulary
        seed_text = "unknown_word test"
        
        generated = generate_from_string(
            self.model, seed_text, self.word2idx, self.idx2word,
            context_len=self.context_len, steps=3
        )
        
        # Should still generate something
        self.assertIsInstance(generated, str)
    
    def test_empty_seed(self):
        """Test generation with empty seed."""
        generated = generate_from_string(
            self.model, "", self.word2idx, self.idx2word,
            context_len=self.context_len, steps=3
        )
        
        # Should still generate something (will use padding)
        self.assertIsInstance(generated, str)
    
    def test_long_seed(self):
        """Test generation with seed longer than context length."""
        long_seed_text = " ".join(self.words * 3)  # Much longer than context_len
        
        generated = generate_from_string(
            self.model, long_seed_text, self.word2idx, self.idx2word,
            context_len=self.context_len, steps=3
        )
        
        # Should still work (will truncate to context_len)
        self.assertIsInstance(generated, str)
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with fixed seed."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        seed = [self.word2idx['hello']] * self.context_len
        
        # Generate twice with same parameters
        gen1 = generate(
            self.model, seed.copy(), self.idx2word, self.word2idx,
            steps=5, temperature=1.0, top_k=1  # top_k=1 should be deterministic
        )
        
        torch.manual_seed(42)  # Reset seed
        gen2 = generate(
            self.model, seed.copy(), self.idx2word, self.word2idx,
            steps=5, temperature=1.0, top_k=1
        )
        
        # With top_k=1, should get same result
        self.assertEqual(gen1, gen2)
    
    def test_steps_parameter(self):
        """Test that steps parameter controls output length."""
        seed = [self.word2idx['hello']] * self.context_len
        
        for steps in [1, 3, 5]:
            with self.subTest(steps=steps):
                generated = generate(
                    self.model, seed, self.idx2word, self.word2idx,
                    steps=steps, temperature=1.0, top_k=self.vocab_size
                )
                
                words = generated.split()
                # Should generate exactly 'steps' number of tokens
                # (though some might be duplicates when joined)
                self.assertLessEqual(len(words), steps)
                self.assertGreater(len(words), 0)
    
    def test_model_eval_mode(self):
        """Test that model stays in eval mode during generation."""
        self.model.train()  # Set to training mode
        
        seed = [self.word2idx['hello']] * self.context_len
        
        # Generation should set model to eval mode
        generate(
            self.model, seed, self.idx2word, self.word2idx,
            steps=3, temperature=1.0, top_k=self.vocab_size
        )
        
        # Model should be in eval mode after generation
        self.assertFalse(self.model.training)


class TestSamplingDistributions(unittest.TestCase):
    """Test sampling functions with various probability distributions."""
    
    def test_uniform_distribution(self):
        """Test sampling from uniform distribution."""
        probs = torch.ones(5) / 5  # Uniform distribution
        
        # Test top-k sampling
        for k in [1, 3, 5]:
            sampled = sample_top_k(probs, k=k)
            self.assertIn(sampled, range(5))
        
        # Test top-p sampling
        for p in [0.2, 0.6, 1.0]:
            sampled = sample_top_p(probs, p=p)
            self.assertIn(sampled, range(5))
    
    def test_peaked_distribution(self):
        """Test sampling from peaked distribution."""
        probs = torch.tensor([0.8, 0.1, 0.05, 0.03, 0.02])
        
        # With top-k=1, should always pick the highest
        sampled = sample_top_k(probs, k=1)
        self.assertEqual(sampled, 0)
        
        # With small top-p, should pick from high probability tokens
        # Note: this test is probabilistic, so we'll just check it's valid
        sampled = sample_top_p(probs, p=0.1)
        self.assertIn(sampled, range(len(probs)))  # Should be valid index


if __name__ == '__main__':
    unittest.main()
