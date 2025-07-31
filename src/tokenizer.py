"""
Text tokenization and vocabulary building utilities.
"""

import re
import torch
from collections import Counter
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    """
    Simple word-based tokenization.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        List[str]: List of tokens
    """
    return re.findall(r'\b\w+\b', text.lower())


def build_vocab(words: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from list of words.
    
    Args:
        words (List[str]): List of words
        min_freq (int): Minimum frequency for a word to be included
        
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: word2idx and idx2word mappings
    """
    counter = Counter(words)
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    
    # Sort vocabulary for consistent indexing
    word2idx = {word: idx + 2 for idx, word in enumerate(sorted(vocab))}
    
    # Add special tokens
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    
    # Create reverse mapping
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"Built vocabulary with {len(word2idx)} tokens")
    print(f"Special tokens: <PAD>={word2idx['<PAD>']}, <UNK>={word2idx['<UNK>']}")
    
    return word2idx, idx2word


def encode(words: List[str], word2idx: Dict[str, int]) -> List[int]:
    """
    Encode words to token IDs.
    
    Args:
        words (List[str]): List of words to encode
        word2idx (Dict[str, int]): Word to index mapping
        
    Returns:
        List[int]: List of token IDs
    """
    return [word2idx.get(word, word2idx['<UNK>']) for word in words]


def save_vocab(word2idx: Dict[str, int], idx2word: Dict[int, str], filepath: str) -> None:
    """
    Save vocabulary to file.
    
    Args:
        word2idx (Dict[str, int]): Word to index mapping
        idx2word (Dict[int, str]): Index to word mapping
        filepath (str): Path to save vocabulary
    """
    torch.save({
        "word2idx": word2idx,
        "idx2word": idx2word
    }, filepath)
    print(f"✅ Vocabulary saved to {filepath}")


def load_vocab(filepath: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load vocabulary from file.
    
    Args:
        filepath (str): Path to vocabulary file
        
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: word2idx and idx2word mappings
    """
    vocab_data = torch.load(filepath)
    word2idx = vocab_data["word2idx"]
    idx2word = vocab_data["idx2word"]
    print(f"✅ Vocabulary loaded from {filepath} ({len(word2idx)} tokens)")
    return word2idx, idx2word
