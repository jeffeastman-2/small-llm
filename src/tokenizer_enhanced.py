"""
Enhanced tokenization utilities supporting both word-based and BPE tokenization.

This module provides backward compatibility with existing word-based tokenization
while adding HuggingFace tokenizers BPE support for improved subword handling.
"""

import re
import os
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class WordTokenizer:
    """Original word-based tokenizer for backward compatibility."""
    
    def __init__(self):
        self.word2idx = None
        self.idx2word = None
        self.vocab_size = 0
    
    def train(self, text: str, min_freq: int = 1) -> None:
        """Train word-based tokenizer."""
        words = self._tokenize_words(text)
        self.word2idx, self.idx2word = self._build_vocab(words, min_freq)
        self.vocab_size = len(self.word2idx)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.word2idx is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        words = self._tokenize_words(text)
        return [self.word2idx.get(word, self.word2idx.get('<UNK>', 1)) for word in words]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.idx2word is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        words = [self.idx2word.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join(words)
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Simple word-based tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _build_vocab(self, words: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from words."""
        # Count word frequencies
        word_counts = Counter(words)
        
        # Filter by minimum frequency
        filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # Create special tokens
        special_tokens = ['<PAD>', '<UNK>']
        vocab_words = special_tokens + sorted(filtered_words)
        
        # Create mappings
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        return word2idx, idx2word
    
    def save(self, vocab_path: str) -> None:
        """Save vocabulary."""
        if self.word2idx is None or self.idx2word is None:
            raise ValueError("No vocabulary to save. Train tokenizer first.")
        
        torch.save({
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'tokenizer_type': 'word'
        }, vocab_path)
    
    def load(self, vocab_path: str) -> None:
        """Load vocabulary."""
        data = torch.load(vocab_path, weights_only=False)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']


class BPETokenizer:
    """HuggingFace BPE tokenizer for subword tokenization."""
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.is_trained = False
    
    def train(self, text: str, model_prefix: str = "tokenizer") -> None:
        """
        Train HuggingFace BPE tokenizer.
        
        Args:
            text: Training text
            model_prefix: Prefix for model files (will create .json file)
        """
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        )
        
        # Create temporary text file for training
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(text)
            text_file = f.name
        
        try:
            # Train the tokenizer
            self.tokenizer.train([text_file], trainer)
            self.is_trained = True
            
            # Save the tokenizer
            model_path = f"{model_prefix}.json"
            self.tokenizer.save(model_path)
            
            print(f"✅ BPE tokenizer trained with {self.vocab_size} vocabulary size")
            print(f"   Model saved to: {model_path}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(text_file):
                os.remove(text_file)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.is_trained:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self.is_trained:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        return self.tokenizer.decode(token_ids)
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces (for debugging)."""
        if not self.is_trained:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.is_trained:
            return self.tokenizer.get_vocab_size()
        return self.vocab_size
    
    def save(self, vocab_path: str) -> None:
        """Save tokenizer."""
        if not self.is_trained:
            raise ValueError("No tokenizer to save. Train tokenizer first.")
        
        # Save the tokenizer directly
        self.tokenizer.save(vocab_path)
    
    def load(self, vocab_path: str) -> None:
        """Load tokenizer."""
        self.tokenizer = Tokenizer.from_file(vocab_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.is_trained = True


class UnifiedTokenizer:
    """
    Unified tokenizer interface supporting both word-based and BPE tokenization.
    
    This class provides a single interface for both tokenization methods,
    allowing easy switching and comparison between approaches.
    """
    
    def __init__(self, tokenizer_type: str = "word", vocab_size: int = 16000):
        """
        Initialize unified tokenizer.
        
        Args:
            tokenizer_type: "word" or "bpe"
            vocab_size: Vocabulary size (only used for BPE)
        """
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == "word":
            self.tokenizer = WordTokenizer()
        elif tokenizer_type == "bpe":
            self.tokenizer = BPETokenizer(vocab_size=vocab_size)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def train(self, text: str, min_freq: int = 1, model_prefix: str = "tokenizer") -> None:
        """Train the tokenizer."""
        if self.tokenizer_type == "word":
            self.tokenizer.train(text, min_freq=min_freq)
        else:  # bpe
            self.tokenizer.train(text, model_prefix=model_prefix)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'get_vocab_size'):
            return self.tokenizer.get_vocab_size()
        else:
            return 0
    
    def save(self, vocab_path: str) -> None:
        """Save tokenizer."""
        self.tokenizer.save(vocab_path)
    
    def load(self, vocab_path: str) -> None:
        """Load tokenizer."""
        self.tokenizer.load(vocab_path)


# Backward compatibility functions (preserve existing API)
def tokenize(text: str) -> List[str]:
    """Simple word-based tokenization (backward compatibility)."""
    return re.findall(r'\b\w+\b', text.lower())


def build_vocab(words: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary from words (backward compatibility)."""
    tokenizer = WordTokenizer()
    text = ' '.join(words)
    tokenizer.train(text, min_freq=min_freq)
    return tokenizer.word2idx, tokenizer.idx2word


def encode(words: List[str], word2idx: Dict[str, int]) -> List[int]:
    """Encode words to indices (backward compatibility)."""
    return [word2idx.get(word, word2idx.get('<UNK>', 1)) for word in words]


def save_vocab(word2idx: Dict[str, int], idx2word: Dict[int, str], vocab_path: str) -> None:
    """Save vocabulary (backward compatibility)."""
    torch.save({
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(word2idx),
        'tokenizer_type': 'word'
    }, vocab_path)


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load vocabulary (backward compatibility)."""
    data = torch.load(vocab_path, weights_only=False)
    return data['word2idx'], data['idx2word']


def load_tokenizer(vocab_path: str):
    """Load the appropriate tokenizer based on the vocab file."""
    data = torch.load(vocab_path, weights_only=False)
    
    if data.get('tokenizer_type') == 'bpe':
        # Load BPE tokenizer
        tokenizer = BPETokenizer()
        tokenizer.load(vocab_path)
        return tokenizer
    else:
        # Load word tokenizer
        tokenizer = WordTokenizer() 
        tokenizer.load(vocab_path)
        return tokenizer
