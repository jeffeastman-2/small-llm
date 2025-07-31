"""
Mini GPT-style Transformer model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    """
    A small GPT-style transformer model for text generation.
    
    Features:
    - Token and positional embeddings
    - Multi-layer transformer with causal attention
    - Layer normalization
    - Linear output projection
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        context_len: int = 10, 
        embed_dim: int = 64, 
        num_heads: int = 2, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the MiniGPT model.
        
        Args:
            vocab_size (int): Size of vocabulary
            context_len (int): Maximum sequence length
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.context_len = context_len
        self.embed_dim = embed_dim
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, context_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Logits for next token [batch_size, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.token_embed(x)  # [batch_size, seq_len, embed_dim]
        
        # Positional embeddings
        pos_emb = self.pos_embed[:, :seq_len, :]  # [1, seq_len, embed_dim]
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Transformer layers
        x = self.transformer(x, mask=mask)
        
        # Final layer norm and output projection
        x = self.ln_final(x[:, -1, :])  # Use only the last token
        logits = self.output(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
