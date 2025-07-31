"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from .generation import generate, generate_from_string


class Trainer:
    """
    Trainer class for the MiniGPT model.
    """
    
    def __init__(
        self,
        model,
        dataloader: DataLoader,
        lr: float = 1e-4,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            lr: Learning rate
            device: Device to use for training
        """
        self.model = model
        self.dataloader = dataloader
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Training state
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0
        
        print(f"Model has {self.model.get_num_trainable_params():,} trainable parameters")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.dataloader)
        
        for batch_idx, (context, target) in enumerate(self.dataloader):
            context, target = context.to(self.device), target.to(self.device)
            
            # Forward pass
            logits = self.model(context)
            loss = self.loss_fn(logits, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress occasionally
            if batch_idx % (num_batches // 5 + 1) == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def save_model(self, filepath: str) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), filepath)
    
    def train(
        self,
        epochs: int = 100,
        save_path: str = "best_model.pth",
        patience: int = 20,
        print_every: int = 5,
        word2idx: Optional[Dict[str, int]] = None,
        idx2word: Optional[Dict[int, str]] = None,
        seed_tokens: Optional[List[int]] = None
    ) -> None:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            save_path: Path to save best model
            patience: Early stopping patience
            print_every: Generate samples every N epochs
            word2idx: Word to index mapping (for generation)
            idx2word: Index to word mapping (for generation)
            seed_tokens: Seed tokens for generation samples
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Early stopping patience: {patience}")
        print(f"Saving best model to: {save_path}")
        
        for epoch in range(epochs):
            # Train for one epoch
            avg_loss = self.train_epoch()
            
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Check if this is the best model so far
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_since_improvement = 0
                self.save_model(save_path)
                print(f"  ✅ New best model saved (Loss: {self.best_loss:.4f})")
            else:
                self.epochs_since_improvement += 1
                print(f"  No improvement for {self.epochs_since_improvement} epochs")
            
            # Generate sample text
            if (word2idx and idx2word and seed_tokens and 
                (epoch % print_every == 0 or epoch == epochs - 1)):
                self._generate_samples(seed_tokens, idx2word, word2idx)
            
            # Early stopping
            if self.epochs_since_improvement >= patience:
                print(f"\n🛑 Early stopping after {patience} epochs without improvement")
                break
        
        print(f"\n✅ Training completed! Best loss: {self.best_loss:.4f}")
    
    def _generate_samples(
        self, 
        seed_tokens: List[int], 
        idx2word: Dict[int, str], 
        word2idx: Dict[str, int]
    ) -> None:
        """Generate sample text during training."""
        self.model.eval()
        
        print("\n" + "="*50)
        print("GENERATION SAMPLES:")
        
        # Generate from token sequence
        print("\nFrom token sequence:")
        sample1 = generate(
            self.model, seed_tokens, idx2word, word2idx, 
            steps=15, top_k=10
        )
        print(f"  {sample1}")
        
        # Generate from text prompt
        print("\nFrom text prompt 'the model could':")
        sample2 = generate_from_string(
            self.model, "the model could", word2idx, idx2word, 
            context_len=self.model.context_len, steps=15, top_k=10
        )
        print(f"  {sample2}")
        
        print("="*50 + "\n")
        
        self.model.train()
