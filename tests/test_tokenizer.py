"""
Unit tests for tokenizer module.
"""

import unittest
import tempfile
import os
import torch
from collections import Counter

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.tokenizer import (
    tokenize, build_vocab, encode, save_vocab, load_vocab
)


class TestTokenizer(unittest.TestCase):
    """Test cases for tokenizer functions."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_text = "Hello world! This is a test. Hello again world."
        self.sample_words = ["hello", "world", "this", "is", "a", "test", "hello", "again", "world"]
        
    def test_tokenize(self):
        """Test text tokenization."""
        tokens = tokenize(self.sample_text)
        expected = ["hello", "world", "this", "is", "a", "test", "hello", "again", "world"]
        self.assertEqual(tokens, expected)
        
        # Test empty string
        self.assertEqual(tokenize(""), [])
        
        # Test string with no words
        self.assertEqual(tokenize("!@#$%"), [])
        
        # Test mixed case
        mixed_text = "Hello WORLD test"
        tokens = tokenize(mixed_text)
        self.assertEqual(tokens, ["hello", "world", "test"])
    
    def test_build_vocab(self):
        """Test vocabulary building."""
        word2idx, idx2word = build_vocab(self.sample_words, min_freq=1)
        
        # Check special tokens
        self.assertEqual(word2idx['<PAD>'], 0)
        self.assertEqual(word2idx['<UNK>'], 1)
        
        # Check vocabulary contains expected words
        expected_words = {"hello", "world", "this", "is", "a", "test", "again"}
        for word in expected_words:
            self.assertIn(word, word2idx)
        
        # Check reverse mapping
        for word, idx in word2idx.items():
            self.assertEqual(idx2word[idx], word)
        
        # Test min frequency filtering
        word2idx_filtered, _ = build_vocab(self.sample_words, min_freq=2)
        # Only "hello" and "world" appear twice
        vocab_words = set(word2idx_filtered.keys()) - {'<PAD>', '<UNK>'}
        self.assertEqual(vocab_words, {"hello", "world"})
    
    def test_encode(self):
        """Test word encoding."""
        word2idx, _ = build_vocab(self.sample_words, min_freq=1)
        
        # Test normal encoding
        test_words = ["hello", "world", "test"]
        encoded = encode(test_words, word2idx)
        
        # Should be valid indices
        for idx in encoded:
            self.assertIn(idx, word2idx.values())
        
        # Test unknown word handling
        unknown_words = ["unknown", "word"]
        encoded_unknown = encode(unknown_words, word2idx)
        self.assertEqual(encoded_unknown, [word2idx['<UNK>'], word2idx['<UNK>']])
    
    def test_save_load_vocab(self):
        """Test vocabulary save and load."""
        word2idx, idx2word = build_vocab(self.sample_words, min_freq=1)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save vocabulary
            save_vocab(word2idx, idx2word, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load vocabulary
            loaded_word2idx, loaded_idx2word = load_vocab(temp_path)
            
            # Check they match
            self.assertEqual(word2idx, loaded_word2idx)
            self.assertEqual(idx2word, loaded_idx2word)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_vocab_consistency(self):
        """Test vocabulary consistency across operations."""
        words = ["apple", "banana", "cherry", "apple", "banana"]
        word2idx, idx2word = build_vocab(words, min_freq=1)
        
        # Test that encoding and decoding are consistent
        encoded = encode(words, word2idx)
        decoded = [idx2word[idx] for idx in encoded]
        self.assertEqual(decoded, words)
    
    def test_empty_vocabulary(self):
        """Test handling of empty vocabulary."""
        word2idx, idx2word = build_vocab([], min_freq=1)
        
        # Should only have special tokens
        self.assertEqual(len(word2idx), 2)
        self.assertIn('<PAD>', word2idx)
        self.assertIn('<UNK>', word2idx)


if __name__ == '__main__':
    unittest.main()
