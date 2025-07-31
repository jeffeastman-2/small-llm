"""
Custom PyTorch dataset for text data.
"""

import torch
from torch.utils.data import Dataset
from typing import List


class TextDataset(Dataset):
    """
    Dataset for next-token prediction training.
    
    Creates input-target pairs where:
    - Input: sequence of context_len tokens
    - Target: next token in sequence
    """
    
    def __init__(self, encoded: List[int], context_len: int):
        """
        Initialize dataset.
        
        Args:
            encoded (List[int]): List of encoded token IDs
            context_len (int): Length of context window
        """
        self.context_len = context_len
        self.data = []
        
        # Create input-target pairs
        for i in range(len(encoded) - context_len):
            context = torch.tensor(encoded[i:i + context_len])
            target = torch.tensor(encoded[i + context_len])
            self.data.append((context, target))
        
        print(f"Created dataset with {len(self.data)} samples")
        print(f"Context length: {context_len}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx]
