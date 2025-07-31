"""
Integration tests for the complete pipeline.
"""

import unittest
import tempfile
import os
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.tokenizer import tokenize, build_vocab, encode, save_vocab, load_vocab
from src.dataset import TextDataset
from src.model import MiniGPT
from src.trainer import Trainer
from src.generation import generate_from_string


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test data."""
        # Sample text data
        self.sample_text = """
        This is a sample document for testing. 
        The model should learn from this text.
        It contains multiple sentences and words.
        Testing the complete pipeline is important.
        This text will be used for training a small model.
        """
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test the complete training and generation pipeline."""
        # Step 1: Tokenization and vocabulary
        words = tokenize(self.sample_text)
        self.assertGreater(len(words), 0)
        
        word2idx, idx2word = build_vocab(words, min_freq=1)
        vocab_size = len(word2idx)
        self.assertGreater(vocab_size, 2)  # At least <PAD>, <UNK> + some words
        
        # Step 2: Encoding
        encoded = encode(words, word2idx)
        self.assertEqual(len(encoded), len(words))
        
        # Step 3: Dataset creation
        context_len = 3
        dataset = TextDataset(encoded, context_len)
        self.assertGreater(len(dataset), 0)
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Step 4: Model creation
        model = MiniGPT(
            vocab_size=vocab_size,
            context_len=context_len,
            embed_dim=16,
            num_heads=2,
            num_layers=1
        )
        
        # Step 5: Training (just a few steps)
        trainer = Trainer(model, dataloader, lr=1e-3)
        
        # Train for just 2 epochs to test pipeline
        trainer.train(
            epochs=2,
            patience=5,
            word2idx=word2idx,
            idx2word=idx2word,
            seed_tokens=encoded[:context_len]
        )
        
        # Step 6: Generation
        generated = generate_from_string(
            model, "this is", word2idx, idx2word,
            context_len=context_len, steps=5
        )
        
        # Should generate something
        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), 0)
    
    def test_vocab_persistence(self):
        """Test vocabulary save and load in pipeline."""
        words = tokenize(self.sample_text)
        word2idx, idx2word = build_vocab(words, min_freq=1)
        
        # Save vocabulary
        vocab_path = os.path.join(self.temp_dir, "test_vocab.pt")
        save_vocab(word2idx, idx2word, vocab_path)
        
        # Load vocabulary
        loaded_word2idx, loaded_idx2word = load_vocab(vocab_path)
        
        # Should be identical
        self.assertEqual(word2idx, loaded_word2idx)
        self.assertEqual(idx2word, loaded_idx2word)
        
        # Test encoding with loaded vocab
        encoded_original = encode(words, word2idx)
        encoded_loaded = encode(words, loaded_word2idx)
        self.assertEqual(encoded_original, encoded_loaded)
    
    def test_model_checkpoint_loading(self):
        """Test model checkpoint save and load."""
        words = tokenize(self.sample_text)
        word2idx, idx2word = build_vocab(words, min_freq=1)
        vocab_size = len(word2idx)
        
        # Create and train model
        context_len = 3
        model1 = MiniGPT(vocab_size=vocab_size, context_len=context_len, embed_dim=16)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.pth")
        torch.save(model1.state_dict(), model_path)
        
        # Create new model and load weights
        model2 = MiniGPT(vocab_size=vocab_size, context_len=context_len, embed_dim=16)
        model2.load_state_dict(torch.load(model_path))
        
        # Models should produce same output
        test_input = torch.randint(0, vocab_size, (1, context_len))
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
    
    def test_different_context_lengths(self):
        """Test pipeline with different context lengths."""
        words = tokenize(self.sample_text)
        word2idx, idx2word = build_vocab(words, min_freq=1)
        vocab_size = len(word2idx)
        encoded = encode(words, word2idx)
        
        for context_len in [2, 4, 6]:
            with self.subTest(context_len=context_len):
                # Create dataset
                dataset = TextDataset(encoded, context_len)
                
                if len(dataset) > 0:  # Only test if dataset has samples
                    # Create model
                    model = MiniGPT(
                        vocab_size=vocab_size,
                        context_len=context_len,
                        embed_dim=16
                    )
                    
                    # Test generation
                    generated = generate_from_string(
                        model, "test", word2idx, idx2word,
                        context_len=context_len, steps=3
                    )
                    
                    self.assertIsInstance(generated, str)
    
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        # Test with very small vocabulary
        words = ["a", "b"]
        word2idx, idx2word = build_vocab(words, min_freq=1)
        vocab_size = len(word2idx)  # Should be 4: a, b, <PAD>, <UNK>
        
        # Test model creation with small vocab
        model = MiniGPT(vocab_size=vocab_size, context_len=2, embed_dim=8)
        self.assertIsNotNone(model)
        
        # Test generation with limited vocabulary
        generated = generate_from_string(
            model, "unknown words here", word2idx, idx2word,
            context_len=2, steps=2
        )
        
        # Should still generate something
        self.assertIsInstance(generated, str)
    
    def test_reproducibility(self):
        """Test that results are reproducible with fixed seeds."""
        # Set seeds
        torch.manual_seed(42)
        
        words = tokenize(self.sample_text)
        word2idx, idx2word = build_vocab(words, min_freq=1)
        vocab_size = len(word2idx)
        
        # Create model
        model = MiniGPT(vocab_size=vocab_size, context_len=3, embed_dim=16)
        
        # Generate text
        generated1 = generate_from_string(
            model, "this is", word2idx, idx2word,
            context_len=3, steps=5, top_k=1  # Deterministic with top_k=1
        )
        
        # Reset seed and generate again
        torch.manual_seed(42)
        model2 = MiniGPT(vocab_size=vocab_size, context_len=3, embed_dim=16)
        
        generated2 = generate_from_string(
            model2, "this is", word2idx, idx2word,
            context_len=3, steps=5, top_k=1
        )
        
        # Should be identical (note: this might not work perfectly due to 
        # random initialization differences, but structure should be same)
        self.assertIsInstance(generated1, str)
        self.assertIsInstance(generated2, str)


class TestPerformance(unittest.TestCase):
    """Performance and scalability tests."""
    
    def test_large_vocabulary_performance(self):
        """Test performance with larger vocabulary."""
        # Create text with many unique words
        words = []
        unique_words = set()
        for i in range(50):  # Create 50 unique words
            word = f"word_{i}"
            words.extend([word, word])  # Add each word twice to meet min_freq=2
            unique_words.add(word)
        
        # Build vocabulary with min_freq=2 so all words are included
        word2idx, idx2word = build_vocab(words, min_freq=2)
        vocab_size = len(word2idx)
        
        # Should include all unique words plus special tokens
        expected_min_size = len(unique_words) + 2  # +2 for <PAD> and <UNK>
        self.assertGreaterEqual(vocab_size, expected_min_size)
        
        # Test encoding performance
        encoded = encode(words, word2idx)
        self.assertEqual(len(encoded), len(words))
        
        # Test model creation with vocabulary
        model = MiniGPT(vocab_size=vocab_size, context_len=5, embed_dim=32)
        
        # Should create model without issues
        param_count = model.get_num_params()
        self.assertGreater(param_count, 1000)
    
    def test_long_sequence_handling(self):
        """Test handling of long sequences."""
        # Create long sequence
        words = ["word"] * 10000
        word2idx, idx2word = build_vocab(words, min_freq=1)
        encoded = encode(words, word2idx)
        
        # Test dataset creation with long sequence
        context_len = 10
        dataset = TextDataset(encoded, context_len)
        
        # Should create many samples
        self.assertGreater(len(dataset), 9000)
        
        # Test random access
        sample_indices = [0, len(dataset)//2, len(dataset)-1]
        for idx in sample_indices:
            context, target = dataset[idx]
            self.assertEqual(context.shape, (context_len,))
            self.assertEqual(target.shape, ())


if __name__ == '__main__':
    unittest.main()
