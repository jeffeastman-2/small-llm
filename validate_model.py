#!/usr/bin/env python3
"""
Validation script to check for overfitting.
Splits the data and evaluates the model on unseen text.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

# Set tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import extract_text_from_pdfs, clean_text
from src.tokenizer_enhanced import UnifiedTokenizer
from src.dataset import TextDataset
from src.model import MiniGPT


def collate_fn(batch):
    """Custom collate function to handle (context, target) pairs."""
    contexts = []
    targets = []
    
    for context, target in batch:
        contexts.append(context)
        targets.append(target)
    
    return torch.stack(contexts), torch.stack(targets)


def calculate_loss(model, dataloader, device):
    """Calculate average loss on a dataset."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            contexts, targets = batch
            contexts = contexts.to(device)  # [batch_size, context_len]
            targets = targets.to(device)    # [batch_size]
            
            # Forward pass through model with context
            logits = model(contexts)  # Check actual output shape
            
            # Debug: print shapes to understand the model output
            # print(f"Context shape: {contexts.shape}, Logits shape: {logits.shape}, Targets shape: {targets.shape}")
            
            # Handle different logit shapes
            if len(logits.shape) == 3:
                # [batch_size, seq_len, vocab_size] - use last position
                last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            elif len(logits.shape) == 2:
                # [batch_size, vocab_size] - already at right shape
                last_logits = logits
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(last_logits, targets)
            
            total_loss += loss.item() * contexts.size(0)
            total_samples += contexts.size(0)
    
    return total_loss / total_samples


def main():
    """Check for overfitting by evaluating on train/validation split."""
    
    print("🔍 Overfitting Analysis")
    print("=" * 50)
    
    # Configuration
    config = {
        'training_dir': 'Training',
        'model_path': 'best_model_bpe.pth',
        'tokenizer_model_prefix': 'tokenizer_bpe',
        'tokenizer_type': 'bpe',
        'context_len': 64,
        'batch_size': 32,  # Smaller batch for evaluation
        'validation_split': 0.2  # 20% for validation
    }
    
    # Load data
    print("📖 Loading and processing text data...")
    text_data = extract_text_from_pdfs(config['training_dir'])
    clean_text_data = clean_text(text_data)
    print(f"   ✅ Loaded {len(clean_text_data):,} characters")
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = UnifiedTokenizer(tokenizer_type=config['tokenizer_type'], vocab_size=16000)
    tokenizer.load('tokenizer_bpe.json')  # Load the saved tokenizer file
    
    # Encode text
    print("📊 Encoding text...")
    encoded = tokenizer.encode(clean_text_data)
    print(f"   ✅ Encoded {len(encoded):,} tokens")
    
    # Split data into train/validation
    split_idx = int(len(encoded) * (1 - config['validation_split']))
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training tokens:   {len(train_tokens):,}")
    print(f"   Validation tokens: {len(val_tokens):,}")
    
    # Create datasets
    train_dataset = TextDataset(train_tokens, config['context_len'])
    val_dataset = TextDataset(val_tokens, config['context_len'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"   Training samples:   {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    
    # Load model
    print(f"\n🤖 Loading model from {config['model_path']}...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model state
    checkpoint = torch.load(config['model_path'], map_location=device, weights_only=True)
    
    # Initialize model with correct parameters (using the model's actual vocab size)
    model = MiniGPT(
        vocab_size=16000,  # Use the vocab size from training config
        embed_dim=512,  # From training config
        context_len=config['context_len'],
        num_heads=8,
        num_layers=6
    )
    
    model.load_state_dict(checkpoint)
    model.to(device)
    
    print(f"   ✅ Model loaded on {device}")
    
    # Evaluate on both splits
    print(f"\n📊 Evaluating model performance...")
    
    train_loss = calculate_loss(model, train_loader, device)
    val_loss = calculate_loss(model, val_loader, device)
    
    print(f"\n🎯 Results:")
    print(f"   Training Loss:   {train_loss:.4f}")
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Difference:      {val_loss - train_loss:.4f}")
    
    # Analysis
    print(f"\n🔍 Overfitting Analysis:")
    if val_loss > train_loss * 1.5:
        print("   ⚠️  SEVERE OVERFITTING detected!")
        print("      Validation loss is >50% higher than training loss")
    elif val_loss > train_loss * 1.2:
        print("   ⚠️  MODERATE OVERFITTING detected")
        print("      Validation loss is 20-50% higher than training loss")
    elif val_loss > train_loss * 1.1:
        print("   ⚠️  MILD OVERFITTING detected")
        print("      Validation loss is 10-20% higher than training loss")
    else:
        print("   ✅ No significant overfitting detected")
        print("      Validation loss is within 10% of training loss")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if val_loss > train_loss * 1.2:
        print("   • Add regularization (dropout, weight decay)")
        print("   • Reduce model complexity")
        print("   • Use early stopping based on validation loss")
        print("   • Increase training data if possible")
    else:
        print("   • Model appears to generalize well")
        print("   • Current performance is suitable for the task")


if __name__ == "__main__":
    main()
