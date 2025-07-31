#!/usr/bin/env python3
"""
Optimized training script for Mac M4 with 32GB RAM.

This version takes advantage of your powerful hardware:
- Larger batch sizes for better GPU utilization
- Bigger model for better learning capacity
- Optimized memory usage
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import extract_text_from_pdfs, clean_text
from src.tokenizer import tokenize, build_vocab, encode, save_vocab
from src.dataset import TextDataset
from src.model import MiniGPT
from src.trainer import Trainer
from src.generation import generate_from_string


def main():
    """Optimized training script for Mac M4."""
    
    # Optimized Configuration for Mac M4 + 32GB RAM
    config = {
        'training_dir': 'Training',
        'vocab_path': 'vocab_optimized.pt',
        'model_path': 'best_model_optimized.pth',
        
        # 🚀 OPTIMIZED FOR M4 + 32GB RAM
        'context_len': 32,        # 3x larger context (was 10)
        'embed_dim': 256,         # 4x larger embeddings (was 64)
        'num_heads': 8,           # 4x more attention heads (was 2)
        'num_layers': 6,          # 3x deeper model (was 2)
        'batch_size': 128,        # 4x larger batches (was 32)
        
        # Training parameters
        'learning_rate': 3e-4,    # Slightly higher for larger model
        'epochs': 1000,           # More epochs for complex model
        'patience': 50,           # More patience for convergence
        'min_freq': 2,            # Slightly higher min frequency
        'print_every': 20,        # Less frequent printing
        
        # Memory optimization
        'gradient_accumulation_steps': 2,  # Effective batch size = 256
        'max_grad_norm': 1.0,     # Gradient clipping
    }
    
    print("🚀 Starting OPTIMIZED Mini GPT Training for Mac M4")
    print("="*60)
    print(f"🖥️  Hardware: Mac M4 with MPS acceleration")
    print(f"🧠 Model: {config['embed_dim']}d, {config['num_layers']} layers, {config['num_heads']} heads")
    print(f"📊 Batch: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})")
    print(f"📏 Context: {config['context_len']} tokens")
    print("="*60)
    
    # Step 1: Extract and preprocess text
    print("\n📄 Step 1: Extracting text from PDFs...")
    try:
        raw_text = extract_text_from_pdfs(config['training_dir'])
        clean_text_data = clean_text(raw_text)
        print(f"   ✅ Extracted {len(clean_text_data):,} characters")
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return
    
    # Step 2: Tokenization and vocabulary building
    print("\n🔤 Step 2: Building vocabulary...")
    words = tokenize(clean_text_data)
    print(f"   ✅ Tokenized into {len(words):,} words")
    
    word2idx, idx2word = build_vocab(words, min_freq=config['min_freq'])
    vocab_size = len(word2idx)
    print(f"   ✅ Vocabulary size: {vocab_size:,} unique tokens")
    
    # Save vocabulary
    save_vocab(word2idx, idx2word, config['vocab_path'])
    
    # Step 3: Create dataset
    print("\n📊 Step 3: Creating dataset...")
    encoded = encode(words, word2idx)
    print(f"   ✅ Encoded {len(encoded):,} tokens")
    
    dataset = TextDataset(encoded, config['context_len'])
    
    # Smart pin_memory: only use when supported (CUDA, not MPS)
    use_pin_memory = torch.cuda.is_available()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,  # Utilize multiple CPU cores
        pin_memory=use_pin_memory  # Only use when supported
    )
    
    if use_pin_memory:
        print(f"   🚀 Using pinned memory for faster GPU transfers")
    else:
        print(f"   📋 Pinned memory not supported on MPS - using regular memory")
    
    print(f"   ✅ Created {len(dataset):,} training samples")
    
    # Step 4: Initialize model
    print(f"\n🤖 Step 4: Initializing larger model...")
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        context_len=config['context_len'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    )
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Step 5: Training with optimizations
    print(f"\n🎯 Step 5: Training with M4 optimizations...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        lr=config['learning_rate'],
        patience=config['patience'],
        print_every=config['print_every']
    )
    
    print(f"   🖥️  Training device: {trainer.device}")
    
    # Train with gradient accumulation for effective larger batch size
    best_loss = trainer.train_with_accumulation(
        dataloader,
        epochs=config['epochs'],
        model_path=config['model_path'],
        accumulation_steps=config['gradient_accumulation_steps'],
        max_grad_norm=config['max_grad_norm']
    )
    
    print(f"\n✅ Training completed! Best loss: {best_loss:.4f}")
    
    # Step 6: Generate sample text
    print(f"\n📝 Step 6: Generating sample text...")
    try:
        sample_text = generate_from_string(
            prompt="The key findings of this research",
            model_path=config['model_path'],
            vocab_path=config['vocab_path'],
            max_tokens=100,
            temperature=0.8
        )
        print("Generated text:")
        print("-" * 40)
        print(sample_text)
        print("-" * 40)
    except Exception as e:
        print(f"❌ Error generating text: {e}")
    
    print(f"\n🎉 Optimized training pipeline completed!")
    print(f"📁 Model saved to: {config['model_path']}")
    print(f"📁 Vocab saved to: {config['vocab_path']}")


if __name__ == "__main__":
    main()
