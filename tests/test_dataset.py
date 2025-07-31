"""
Unit tests for dataset module.
"""

import unittest
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.dataset import TextDataset


class TestTextDataset(unittest.TestCase):
    """Test cases for TextDataset."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample encoded sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.encoded = list(range(1, 11))
        self.context_len = 3
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = TextDataset(self.encoded, self.context_len)
        
        # Check dataset length
        expected_len = len(self.encoded) - self.context_len
        self.assertEqual(len(dataset), expected_len)
        
        # Dataset should have 7 samples: [1,2,3]->4, [2,3,4]->5, ..., [7,8,9]->10
        self.assertEqual(len(dataset), 7)
    
    def test_dataset_items(self):
        """Test individual dataset items."""
        dataset = TextDataset(self.encoded, self.context_len)
        
        # Test first item
        context, target = dataset[0]
        expected_context = torch.tensor([1, 2, 3])
        expected_target = torch.tensor(4)
        
        self.assertTrue(torch.equal(context, expected_context))
        self.assertTrue(torch.equal(target, expected_target))
        
        # Test middle item
        context, target = dataset[3]
        expected_context = torch.tensor([4, 5, 6])
        expected_target = torch.tensor(7)
        
        self.assertTrue(torch.equal(context, expected_context))
        self.assertTrue(torch.equal(target, expected_target))
        
        # Test last item
        context, target = dataset[-1]
        expected_context = torch.tensor([7, 8, 9])
        expected_target = torch.tensor(10)
        
        self.assertTrue(torch.equal(context, expected_context))
        self.assertTrue(torch.equal(target, expected_target))
    
    def test_tensor_types(self):
        """Test that dataset returns proper tensor types."""
        dataset = TextDataset(self.encoded, self.context_len)
        context, target = dataset[0]
        
        # Check types
        self.assertIsInstance(context, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        
        # Check data types
        self.assertEqual(context.dtype, torch.long)
        self.assertEqual(target.dtype, torch.long)
        
        # Check shapes
        self.assertEqual(context.shape, (self.context_len,))
        self.assertEqual(target.shape, ())  # scalar tensor
    
    def test_different_context_lengths(self):
        """Test dataset with different context lengths."""
        for context_len in [1, 2, 5]:
            with self.subTest(context_len=context_len):
                dataset = TextDataset(self.encoded, context_len)
                
                expected_len = len(self.encoded) - context_len
                self.assertEqual(len(dataset), expected_len)
                
                if len(dataset) > 0:
                    context, target = dataset[0]
                    self.assertEqual(context.shape, (context_len,))
                    self.assertEqual(target.shape, ())
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with minimum viable sequence
        min_encoded = [1, 2]  # Only enough for context_len=1
        dataset = TextDataset(min_encoded, context_len=1)
        self.assertEqual(len(dataset), 1)
        
        context, target = dataset[0]
        self.assertTrue(torch.equal(context, torch.tensor([1])))
        self.assertTrue(torch.equal(target, torch.tensor(2)))
        
        # Test with exact length (should produce empty dataset)
        exact_encoded = [1, 2, 3]
        dataset = TextDataset(exact_encoded, context_len=3)
        self.assertEqual(len(dataset), 0)
    
    def test_dataset_iteration(self):
        """Test iteration over dataset."""
        dataset = TextDataset(self.encoded, self.context_len)
        
        # Collect all items
        items = list(dataset)
        self.assertEqual(len(items), len(dataset))
        
        # Check each item
        for i, (context, target) in enumerate(items):
            expected_context = torch.tensor(self.encoded[i:i + self.context_len])
            expected_target = torch.tensor(self.encoded[i + self.context_len])
            
            self.assertTrue(torch.equal(context, expected_context))
            self.assertTrue(torch.equal(target, expected_target))
    
    def test_indexing_bounds(self):
        """Test dataset indexing bounds."""
        dataset = TextDataset(self.encoded, self.context_len)
        
        # Test valid indices
        for i in range(len(dataset)):
            context, target = dataset[i]
            self.assertIsInstance(context, torch.Tensor)
            self.assertIsInstance(target, torch.Tensor)
        
        # Test invalid indices
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]
        
        with self.assertRaises(IndexError):
            _ = dataset[-len(dataset) - 1]
    
    def test_dataloader_compatibility(self):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = TextDataset(self.encoded, self.context_len)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Test that we can iterate through dataloader
        batch_count = 0
        for batch_contexts, batch_targets in dataloader:
            batch_count += 1
            
            # Check batch shapes
            batch_size = batch_contexts.shape[0]
            self.assertEqual(batch_contexts.shape, (batch_size, self.context_len))
            self.assertEqual(batch_targets.shape, (batch_size,))
        
        # Should have some batches
        self.assertGreater(batch_count, 0)
    
    def test_large_dataset(self):
        """Test with larger dataset to ensure performance is reasonable."""
        large_encoded = list(range(1000))
        context_len = 10
        
        dataset = TextDataset(large_encoded, context_len)
        
        # Should create correct number of samples
        expected_len = len(large_encoded) - context_len
        self.assertEqual(len(dataset), expected_len)
        
        # Test a few random samples
        import random
        for _ in range(10):
            idx = random.randint(0, len(dataset) - 1)
            context, target = dataset[idx]
            
            expected_context = torch.tensor(large_encoded[idx:idx + context_len])
            expected_target = torch.tensor(large_encoded[idx + context_len])
            
            self.assertTrue(torch.equal(context, expected_context))
            self.assertTrue(torch.equal(target, expected_target))


if __name__ == '__main__':
    unittest.main()
