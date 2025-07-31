#!/usr/bin/env python3
"""
Train a Mini GPT model on PDF documents.

This script demonstrates the complete training pipeline:
1. Extract text from PDF files
2. Build vocabulary and tokenize
3. Create dataset and train model
4. Generate sample text
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
    """Main training script."""
    
    # Configuration
    config = {
        'training_dir': 'Training',
        'vocab_path': 'vocab.pt',
        'model_path': 'best_model.pth',
        'context_len': 10,
        'embed_dim': 64,
        'num_heads': 2,
        'num_layers': 2,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 500,
        'patience': 20,
        'min_freq': 1,
        'print_every': 10
    }
    
    print("🚀 Starting Mini GPT Training Pipeline")
    print("="*50)
    
    # Step 1: Extract and preprocess text
    print("📄 Step 1: Extracting text from PDFs...")
    try:
        raw_text = extract_text_from_pdfs(config['training_dir'])
        clean_text_data = clean_text(raw_text)
        print(f"   Extracted and cleaned {len(clean_text_data)} characters")
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return
    
    # Step 2: Tokenization and vocabulary building
    print("\n🔤 Step 2: Building vocabulary...")
    words = tokenize(clean_text_data)
    print(f"   Tokenized into {len(words)} words")
    
    word2idx, idx2word = build_vocab(words, min_freq=config['min_freq'])
    vocab_size = len(word2idx)
    
    # Save vocabulary
    save_vocab(word2idx, idx2word, config['vocab_path'])
    
    # Step 3: Create dataset
    print("\n📊 Step 3: Creating dataset...")
    encoded = encode(words, word2idx)
    print(f"   Encoded {len(encoded)} tokens")
    
    dataset = TextDataset(encoded, config['context_len'])
    
    # Smart pin_memory: only use when supported (CUDA, not MPS)
    use_pin_memory = torch.cuda.is_available()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        pin_memory=use_pin_memory
    )
    
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Batch size: {config['batch_size']}")
    
    # Step 4: Initialize model
    print("\n🧠 Step 4: Initializing model...")
    model = MiniGPT(
        vocab_size=vocab_size,
        context_len=config['context_len'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    )
    
    print(f"   Model parameters: {model.get_num_trainable_params():,}")
    print(f"   Context length: {config['context_len']}")
    print(f"   Embedding dimension: {config['embed_dim']}")
    print(f"   Number of layers: {config['num_layers']}")
    print(f"   Number of heads: {config['num_heads']}")
    
    # Step 5: Train model
    print("\n🏋️ Step 5: Training model...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        lr=config['learning_rate']
    )
    
    # Get seed tokens for generation samples
    seed_tokens = encoded[:config['context_len']]
    
    trainer.train(
        epochs=config['epochs'],
        save_path=config['model_path'],
        patience=config['patience'],
        print_every=config['print_every'],
        word2idx=word2idx,
        idx2word=idx2word,
        seed_tokens=seed_tokens
    )
    
    # Step 6: Final generation test
    print("\n🎯 Step 6: Final generation test...")
    print("Loading best model for final test...")
    
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    
    test_prompts = [
        "the model could",
        "in this paper we",
        "the results show",
        "it is important to"
    ]
    
    print("\nGeneration samples:")
    for prompt in test_prompts:
        generated = generate_from_string(
            model, prompt, word2idx, idx2word,
            context_len=config['context_len'],
            steps=20, top_k=10
        )
        print(f"  '{prompt}' → {generated}")
    
    print("\n✅ Training pipeline completed successfully!")
    print(f"   Best model saved to: {config['model_path']}")
    print(f"   Vocabulary saved to: {config['vocab_path']}")


if __name__ == "__main__":
    import torch
    main()
