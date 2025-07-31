"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional


class Trainer:
    """
    Trainer class for the MiniGPT model.
    """
    
    def __init__(
        self,
        model,
        dataloader: DataLoader,
        lr: float = 1e-4,
        device: str = None,
        patience: int = 20,
        print_every: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            lr: Learning rate
            device: Device to use for training
            patience: Early stopping patience
            print_every: Print progress every N epochs
        """
        self.model = model
        self.dataloader = dataloader
        self.patience = patience
        self.print_every = print_every
        
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
    
    def train_with_accumulation(self, dataloader, epochs: int, model_path: str, 
                              accumulation_steps: int = 1, max_grad_norm: float = 1.0) -> float:
        """
        Train with gradient accumulation for larger effective batch sizes.
        
        Args:
            dataloader: Training data loader
            epochs: Number of epochs to train
            model_path: Path to save best model
            accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        
        Returns:
            float: Best validation loss achieved
        """
        self.dataloader = dataloader
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = len(dataloader)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            for batch_idx, (context, target) in enumerate(dataloader):
                context, target = context.to(self.device), target.to(self.device)
                
                # Forward pass
                logits = self.model(context)
                loss = self.loss_fn(logits, target)
                
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == num_batches - 1:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps  # Unscale for logging
                
                # Print progress
                if batch_idx % (num_batches // 5 + 1) == 0:
                    print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item() * accumulation_steps:.4f}")
            
            avg_loss = total_loss / num_batches
            
            # Early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.epochs_since_improvement = 0
                self.save_model(model_path)
                print(f"Epoch {epoch+1}: New best loss {avg_loss:.4f} - Model saved!")
            else:
                self.epochs_since_improvement += 1
                print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} (no improvement for {self.epochs_since_improvement} epochs)")
            
            # Check for early stopping
            if self.epochs_since_improvement >= self.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
            
            # Print epoch summary
            if (epoch + 1) % self.print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} Summary:")
                print(f"  Average Loss: {avg_loss:.4f}")
                print(f"  Best Loss: {best_loss:.4f}")
                print(f"  Device: {self.device}")
                print("-" * 50)
        
        return best_loss

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
        print("TRAINING COMPLETED")
        print("="*50 + "\n")
        
        self.model.train()
