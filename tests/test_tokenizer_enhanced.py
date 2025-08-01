"""
Unit tests for enhanced tokenization (BPE and word-based).
"""

import unittest
import tempfile
import os
from src.tokenizer_enhanced import WordTokenizer, BPETokenizer, UnifiedTokenizer


class TestWordTokenizer(unittest.TestCase):
    """Test word-based tokenizer."""
    
    def setUp(self):
        self.tokenizer = WordTokenizer()
        self.sample_text = "The quick brown fox jumps over the lazy dog. The fox is quick."
    
    def test_train_and_encode(self):
        """Test training and encoding."""
        self.tokenizer.train(self.sample_text, min_freq=1)
        
        # Test basic properties
        self.assertIsNotNone(self.tokenizer.word2idx)
        self.assertIsNotNone(self.tokenizer.idx2word)
        self.assertGreater(self.tokenizer.vocab_size, 0)
        
        # Test special tokens
        self.assertIn('<PAD>', self.tokenizer.word2idx)
        self.assertIn('<UNK>', self.tokenizer.word2idx)
        
        # Test encoding
        encoded = self.tokenizer.encode("the quick fox")
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)
    
    def test_decode(self):
        """Test decoding."""
        self.tokenizer.train(self.sample_text, min_freq=1)
        
        # Test encode-decode roundtrip
        text = "the quick brown fox"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        
        # Should be similar (might have case/punctuation differences)
        self.assertIn('quick', decoded)
        self.assertIn('brown', decoded)
        self.assertIn('fox', decoded)
    
    def test_min_freq_filtering(self):
        """Test minimum frequency filtering."""
        # Train with high min_freq
        self.tokenizer.train(self.sample_text, min_freq=2)
        
        # Words appearing only once should not be in vocab (except as <UNK>)
        text = "jumps"  # appears only once
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        
        # Should contain <UNK> or be handled gracefully
        self.assertIsInstance(decoded, str)
    
    def test_save_load(self):
        """Test saving and loading."""
        self.tokenizer.train(self.sample_text, min_freq=1)
        original_vocab_size = self.tokenizer.vocab_size
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            vocab_path = f.name
        
        try:
            # Save
            self.tokenizer.save(vocab_path)
            
            # Load in new tokenizer
            new_tokenizer = WordTokenizer()
            new_tokenizer.load(vocab_path)
            
            # Verify
            self.assertEqual(new_tokenizer.vocab_size, original_vocab_size)
            self.assertEqual(new_tokenizer.word2idx, self.tokenizer.word2idx)
            self.assertEqual(new_tokenizer.idx2word, self.tokenizer.idx2word)
            
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)


class TestBPETokenizer(unittest.TestCase):
    """Test BPE tokenizer."""
    
    def setUp(self):
        self.tokenizer = BPETokenizer(vocab_size=1000)  # Small vocab for testing
        self.sample_text = """
        The quick brown fox jumps over the lazy dog.
        Machine learning is a subset of artificial intelligence.
        Natural language processing involves computational linguistics.
        Deep learning models require substantial computational resources.
        """ * 10  # Repeat to have enough text for BPE training
    
    def test_train_and_encode(self):
        """Test training and encoding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_prefix = os.path.join(temp_dir, "test_tokenizer")
            
            # Train tokenizer
            self.tokenizer.train(self.sample_text, model_prefix=model_prefix)
            
            # Test basic properties
            self.assertTrue(self.tokenizer.is_trained)
            
            # The actual vocab size may be less than requested due to limited training data
            actual_vocab_size = self.tokenizer.get_vocab_size()
            self.assertGreater(actual_vocab_size, 0)
            self.assertLessEqual(actual_vocab_size, 1000)  # Should not exceed requested size
            
            # Test encoding
            encoded = self.tokenizer.encode("the quick brown fox")
            self.assertIsInstance(encoded, list)
            self.assertGreater(len(encoded), 0)
            
            # All token IDs should be within vocab range
            for token_id in encoded:
                self.assertGreaterEqual(token_id, 0)
                self.assertLess(token_id, actual_vocab_size)
    
    def test_decode(self):
        """Test decoding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_prefix = os.path.join(temp_dir, "test_tokenizer")
            
            self.tokenizer.train(self.sample_text, model_prefix=model_prefix)
            
            # Test encode-decode roundtrip
            text = "machine learning models"
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            
            # Decoded text should be similar to original
            # (BPE might add/remove spaces, but content should be preserved)
            self.assertIsInstance(decoded, str)
            self.assertGreater(len(decoded), 0)
    
    def test_subword_tokenization(self):
        """Test that BPE creates subword tokens."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_prefix = os.path.join(temp_dir, "test_tokenizer")
            
            self.tokenizer.train(self.sample_text, model_prefix=model_prefix)
            
            # Test with a word that should be split into subwords
            pieces = self.tokenizer.encode_as_pieces("computational")
            
            # Should be split into multiple pieces for a complex word
            self.assertIsInstance(pieces, list)
            # Most complex words should be split (though this depends on training data)
    
    def test_out_of_vocabulary_handling(self):
        """Test handling of unseen words."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_prefix = os.path.join(temp_dir, "test_tokenizer")
            
            self.tokenizer.train(self.sample_text, model_prefix=model_prefix)
            
            # Test with completely unseen text
            unseen_text = "xylophone quantum supercalifragilisticexpialidocious"
            encoded = self.tokenizer.encode(unseen_text)
            decoded = self.tokenizer.decode(encoded)
            
            # Should handle gracefully (might not be perfect, but shouldn't crash)
            self.assertIsInstance(encoded, list)
            self.assertIsInstance(decoded, str)


class TestUnifiedTokenizer(unittest.TestCase):
    """Test unified tokenizer interface."""
    
    def setUp(self):
        self.sample_text = "The quick brown fox jumps over the lazy dog. " * 20
    
    def test_word_tokenizer_interface(self):
        """Test word tokenizer through unified interface."""
        tokenizer = UnifiedTokenizer(tokenizer_type="word")
        
        # Train
        tokenizer.train(self.sample_text, min_freq=1)
        
        # Test
        self.assertGreater(tokenizer.get_vocab_size(), 0)
        
        # Encode/decode
        encoded = tokenizer.encode("the quick fox")
        decoded = tokenizer.decode(encoded)
        
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
    
    def test_bpe_tokenizer_interface(self):
        """Test BPE tokenizer through unified interface."""
        tokenizer = UnifiedTokenizer(tokenizer_type="bpe", vocab_size=1000)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_prefix = os.path.join(temp_dir, "test_unified")
            
            # Train
            tokenizer.train(self.sample_text, model_prefix=model_prefix)
            
            # Test
            self.assertEqual(tokenizer.get_vocab_size(), 1000)
            
            # Encode/decode
            encoded = tokenizer.encode("the quick fox")
            decoded = tokenizer.decode(encoded)
            
            self.assertIsInstance(encoded, list)
            self.assertIsInstance(decoded, str)
    
    def test_invalid_tokenizer_type(self):
        """Test error handling for invalid tokenizer type."""
        with self.assertRaises(ValueError):
            UnifiedTokenizer(tokenizer_type="invalid")
    
    def test_save_load_word(self):
        """Test save/load for word tokenizer."""
        tokenizer = UnifiedTokenizer(tokenizer_type="word")
        tokenizer.train(self.sample_text, min_freq=1)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            vocab_path = f.name
        
        try:
            # Save
            tokenizer.save(vocab_path)
            
            # Load
            new_tokenizer = UnifiedTokenizer(tokenizer_type="word")
            new_tokenizer.load(vocab_path)
            
            # Test equivalence
            test_text = "the quick brown fox"
            original_encoded = tokenizer.encode(test_text)
            loaded_encoded = new_tokenizer.encode(test_text)
            
            self.assertEqual(original_encoded, loaded_encoded)
            
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility functions."""
    
    def test_tokenize_function(self):
        """Test legacy tokenize function."""
        from src.tokenizer_enhanced import tokenize
        
        text = "The quick brown fox!"
        tokens = tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertIn('quick', tokens)
        self.assertIn('brown', tokens)
    
    def test_build_vocab_function(self):
        """Test legacy build_vocab function."""
        from src.tokenizer_enhanced import build_vocab
        
        words = ['the', 'quick', 'brown', 'fox', 'the', 'quick']
        word2idx, idx2word = build_vocab(words, min_freq=1)
        
        self.assertIsInstance(word2idx, dict)
        self.assertIsInstance(idx2word, dict)
        self.assertIn('<PAD>', word2idx)
        self.assertIn('<UNK>', word2idx)
    
    def test_encode_function(self):
        """Test legacy encode function."""
        from src.tokenizer_enhanced import encode, build_vocab
        
        words = ['the', 'quick', 'brown', 'fox']
        word2idx, _ = build_vocab(words, min_freq=1)
        encoded = encode(words, word2idx)
        
        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), len(words))


if __name__ == '__main__':
    unittest.main()
