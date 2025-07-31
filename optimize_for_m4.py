#!/usr/bin/env python3
"""
Memory optimization tool for Mac M4 with 32GB RAM.

Finds the optimal batch size and model configuration for your hardware.
"""

import torch
import psutil
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MiniGPT


def get_memory_info():
    """Get current memory usage."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent
    }


def estimate_model_memory(model, batch_size, context_len, num_layers):
    """Estimate memory usage for a model configuration."""
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    # Estimate memory usage (rough approximation)
    # Parameters: 4 bytes per float32 parameter
    param_memory_mb = param_count * 4 / (1024**2)
    
    # Activations: depends on batch size and model size
    # This is a rough estimate
    activation_memory_mb = batch_size * context_len * model.embed_dim * num_layers * 4 / (1024**2)
    
    # Gradients: same size as parameters
    gradient_memory_mb = param_memory_mb
    
    # Optimizer state (Adam): roughly 2x parameters
    optimizer_memory_mb = param_memory_mb * 2
    
    total_memory_mb = param_memory_mb + activation_memory_mb + gradient_memory_mb + optimizer_memory_mb
    
    return {
        'parameters_mb': param_memory_mb,
        'activations_mb': activation_memory_mb,
        'gradients_mb': gradient_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'total_mb': total_memory_mb,
        'total_gb': total_memory_mb / 1024
    }


def find_optimal_batch_size(model_config, max_memory_gb=24):
    """Find the largest batch size that fits in memory."""
    print(f"🔍 Finding optimal batch size (max memory: {max_memory_gb}GB)")
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    optimal_batch = 32
    
    for batch_size in batch_sizes:
        # Create temporary model (exclude 'name' from config)
        model_params = {k: v for k, v in model_config.items() if k != 'name'}
        model = MiniGPT(**model_params)
        
        # Estimate memory
        memory_est = estimate_model_memory(
            model, 
            batch_size, 
            model_config['context_len'],
            model_config['num_layers']
        )
        
        print(f"   Batch {batch_size:4d}: {memory_est['total_gb']:.1f}GB estimated")
        
        if memory_est['total_gb'] <= max_memory_gb:
            optimal_batch = batch_size
        else:
            break
    
    return optimal_batch


def main():
    """Find optimal configuration for Mac M4."""
    print("🧠 Mac M4 Memory Optimization Tool")
    print("="*50)
    
    # System info
    mem_info = get_memory_info()
    print(f"💾 System Memory: {mem_info['total_gb']:.1f}GB total")
    print(f"   Available: {mem_info['available_gb']:.1f}GB")
    print(f"   Used: {mem_info['used_gb']:.1f}GB ({mem_info['percent']:.1f}%)")
    
    # Safe memory limit (leave 8GB for system)
    safe_memory_limit = mem_info['total_gb'] - 8
    print(f"   Safe limit for training: {safe_memory_limit:.1f}GB")
    
    # Test configurations
    configs = [
        {
            'name': 'Conservative (Good for learning)',
            'vocab_size': 10000,
            'embed_dim': 256,
            'context_len': 32,
            'num_heads': 8,
            'num_layers': 4
        },
        {
            'name': 'Balanced (Recommended)',
            'vocab_size': 10000,
            'embed_dim': 384,
            'context_len': 48,
            'num_heads': 12,
            'num_layers': 6
        },
        {
            'name': 'Aggressive (Max performance)',
            'vocab_size': 10000,
            'embed_dim': 512,
            'context_len': 64,
            'num_heads': 16,
            'num_layers': 8
        }
    ]
    
    print(f"\n📊 OPTIMAL CONFIGURATIONS")
    print("="*80)
    
    for config in configs:
        print(f"\n🎯 {config['name']}")
        print("-" * 50)
        
        # Create model for estimation
        model = MiniGPT(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            context_len=config['context_len'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers']
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {param_count:,}")
        
        # Find optimal batch size
        config_params = {k: v for k, v in config.items() if k != 'name'}
        optimal_batch = find_optimal_batch_size(config_params, safe_memory_limit)
        
        # Calculate final memory estimate
        memory_est = estimate_model_memory(model, optimal_batch, config['context_len'], config['num_layers'])
        
        print(f"   ✅ Optimal batch size: {optimal_batch}")
        print(f"   📊 Estimated memory: {memory_est['total_gb']:.1f}GB")
        print(f"   🎯 Effective batch: {optimal_batch * 2} (with gradient accumulation)")
        
        # Performance estimate based on our benchmark
        # Scale from the optimized config we tested
        baseline_params = 13_029_136
        baseline_tokens_per_sec = 61_653
        
        scale_factor = baseline_params / param_count
        estimated_tokens_per_sec = baseline_tokens_per_sec * scale_factor
        
        print(f"   ⚡ Estimated speed: {estimated_tokens_per_sec:,.0f} tokens/second")
    
    print(f"\n💡 RECOMMENDATIONS FOR YOUR MAC M4:")
    print("• Start with 'Balanced' configuration")
    print("• Use gradient accumulation for larger effective batch sizes") 
    print("• Your M4 GPU is being fully utilized with MPS")
    print(f"• With {mem_info['total_gb']:.0f}GB RAM, you can train much larger models than typical setups")
    print("• Consider longer context lengths for better text understanding")


if __name__ == "__main__":
    main()
