#!/usr/bin/env python3
"""
Training script with BPE tokenization support.

This script demonstrates the enhanced tokenization capabilities
while maintaining backward compatibility with existing models.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Set tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import extract_text_from_pdfs, clean_text
from src.tokenizer_enhanced import UnifiedTokenizer
from src.dataset import TextDataset
from src.model import MiniGPT
from src.trainer import Trainer
from src.generation import generate_from_string


def main():
    """Training script with BPE tokenization."""
    
    # Configuration
    config = {
        'training_dir': 'Training',
        'vocab_path': 'vocab_bpe.pt',
        'model_path': 'best_model_bpe.pth',
        'tokenizer_model_prefix': 'tokenizer_bpe',
        
        # Tokenization settings
        'tokenizer_type': 'bpe',  # 'word' or 'bpe'
        'vocab_size': 16000,      # For BPE tokenization
        'min_freq': 2,            # For word tokenization
        
        # Model architecture (larger for BPE)
        'context_len': 64,        # Larger context for subwords
        'embed_dim': 512,         # Larger embeddings for bigger vocab
        'num_heads': 8,
        'num_layers': 6,
        'batch_size': 64,         # Adjusted for larger model
        
        # Training parameters
        'learning_rate': 1e-4,
        'epochs': 300,
        'patience': 30,
        'print_every': 10
    }
    
    print("🚀 Starting Mini GPT Training with BPE Tokenization")
    print("="*60)
    print(f"🔤 Tokenizer: {config['tokenizer_type'].upper()}")
    print(f"📊 Vocabulary size: {config['vocab_size']:,}")
    print(f"🧠 Model: {config['embed_dim']}d, {config['num_layers']} layers, {config['num_heads']} heads")
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
    
    # Step 2: Initialize and train tokenizer
    print(f"\n🔤 Step 2: Training {config['tokenizer_type'].upper()} tokenizer...")
    tokenizer = UnifiedTokenizer(
        tokenizer_type=config['tokenizer_type'],
        vocab_size=config['vocab_size']
    )
    
    if config['tokenizer_type'] == 'bpe':
        tokenizer.train(
            clean_text_data, 
            model_prefix=config['tokenizer_model_prefix']
        )
    else:
        tokenizer.train(
            clean_text_data,
            min_freq=config['min_freq']
        )
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"   ✅ Trained tokenizer with {vocab_size:,} tokens")
    
    # Save tokenizer
    tokenizer.save(config['vocab_path'])
    print(f"   ✅ Tokenizer saved to {config['vocab_path']}")
    
    # Step 3: Tokenize and create dataset
    print("\n📊 Step 3: Encoding text and creating dataset...")
    encoded = tokenizer.encode(clean_text_data)
    print(f"   ✅ Encoded {len(encoded):,} tokens")
    
    # Show tokenization examples
    print("\n🔍 Tokenization Examples:")
    sample_texts = [
        "The research methodology demonstrates significant improvements.",
        "Implementation of advanced algorithms yields better performance.",
        "Experimental results indicate promising outcomes for future work."
    ]
    
    for text in sample_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"   Original: {text}")
        print(f"   Tokens:   {tokens[:10]}{'...' if len(tokens) > 10 else ''} ({len(tokens)} total)")
        print(f"   Decoded:  {decoded}")
        print()
    
    # Create dataset
    dataset = TextDataset(encoded, config['context_len'])
    
    # Smart pin_memory: only use when supported (CUDA, not MPS)
    use_pin_memory = torch.cuda.is_available()
    
    # Use fewer workers to avoid tokenizer fork issues
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # Changed from 4 to 0 to avoid multiprocessing issues
        pin_memory=use_pin_memory
    )
    
    print(f"   ✅ Created {len(dataset):,} training samples")
    
    # Step 4: Initialize model
    print(f"\n🤖 Step 4: Initializing model...")
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
    
    # Step 5: Train model
    print(f"\n🎯 Step 5: Training model...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        lr=config['learning_rate'],
        patience=config['patience'],
        print_every=config['print_every']
    )
    
    print(f"   🖥️  Training device: {trainer.device}")
    
    # Train model
    trainer.train(
        epochs=config['epochs'],
        save_path=config['model_path'],
        patience=config['patience'],
        print_every=config['print_every']
    )
    
    # Step 6: Test generation
    print(f"\n📝 Step 6: Testing text generation...")
    
    # Load best model
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    
    test_prompts = [
        "The research findings indicate",
        "In this paper we present",
        "The experimental results demonstrate",
        "Future work should focus on"
    ]
    
    print("Generation samples:")
    for prompt in test_prompts:
        try:
            # Encode prompt
            prompt_tokens = tokenizer.encode(prompt)
            
            # Generate (simplified version for testing)
            with torch.no_grad():
                # Move to same device as model
                device = next(model.parameters()).device
                input_ids = torch.tensor(prompt_tokens[-config['context_len']:]).unsqueeze(0).to(device)
                
                # Generate one token
                logits = model(input_ids)
                next_token = torch.argmax(logits[0, -1, :]).item()
                
                # Decode result
                generated_tokens = prompt_tokens + [next_token]
                generated_text = tokenizer.decode(generated_tokens)
                
            print(f"   '{prompt}' → {generated_text}")
            
        except Exception as e:
            print(f"   '{prompt}' → Error: {e}")
    
    print(f"\n✅ BPE training pipeline completed!")
    print(f"📁 Model saved to: {config['model_path']}")
    print(f"📁 Tokenizer saved to: {config['vocab_path']}")
    
    if config['tokenizer_type'] == 'bpe':
        print(f"📁 BPE model: {config['tokenizer_model_prefix']}.model")
        print(f"📁 BPE vocab: {config['tokenizer_model_prefix']}.vocab")


if __name__ == "__main__":
    main()
