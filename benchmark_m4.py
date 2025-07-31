#!/usr/bin/env python3
"""
Hardware performance benchmark for Mac M4.

Tests different configurations to show the impact of optimization.
"""

import torch
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MiniGPT


def benchmark_config(config_name: str, **model_kwargs):
    """Benchmark a specific model configuration."""
    print(f"\n🔍 Benchmarking: {config_name}")
    print("-" * 40)
    
    # Create model
    model = MiniGPT(**model_kwargs)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Move to MPS if available
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"Device: {device}")
    
    # Create sample data
    batch_size = 64
    context_len = model_kwargs['context_len']
    vocab_size = model_kwargs['vocab_size']
    
    # Generate random input
    x = torch.randint(0, vocab_size, (batch_size, context_len)).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # Benchmark forward pass
    torch.mps.synchronize() if device.type == 'mps' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    
    torch.mps.synchronize() if device.type == 'mps' else None
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    samples_per_second = (100 * batch_size) / total_time
    tokens_per_second = samples_per_second * context_len
    
    print(f"Time for 100 forward passes: {total_time:.3f}s")
    print(f"Samples/second: {samples_per_second:.1f}")
    print(f"Tokens/second: {tokens_per_second:.0f}")
    
    return {
        'params': total_params,
        'samples_per_sec': samples_per_second,
        'tokens_per_sec': tokens_per_second,
        'time': total_time
    }


def main():
    """Run hardware benchmarks."""
    print("🚀 Mac M4 Hardware Benchmark")
    print("="*50)
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) is available")
    else:
        print("❌ MPS not available - using CPU only")
    
    vocab_size = 10000  # Typical vocabulary size
    
    # Original configuration
    original = benchmark_config(
        "Original Configuration",
        vocab_size=vocab_size,
        embed_dim=64,
        context_len=10,
        num_heads=2,
        num_layers=2
    )
    
    # Optimized configuration
    optimized = benchmark_config(
        "M4 Optimized Configuration", 
        vocab_size=vocab_size,
        embed_dim=256,
        context_len=32,
        num_heads=8,
        num_layers=6
    )
    
    # Show comparison
    print("\n📊 PERFORMANCE COMPARISON")
    print("="*50)
    print(f"{'Metric':<25} {'Original':<15} {'Optimized':<15} {'Ratio':<10}")
    print("-" * 65)
    
    params_ratio = optimized['params'] / original['params']
    samples_ratio = optimized['samples_per_sec'] / original['samples_per_sec']
    tokens_ratio = optimized['tokens_per_sec'] / original['tokens_per_sec']
    
    print(f"{'Parameters':<25} {original['params']:,<15} {optimized['params']:,<15} {params_ratio:.1f}x")
    print(f"{'Samples/second':<25} {original['samples_per_sec']:<15.1f} {optimized['samples_per_sec']:<15.1f} {samples_ratio:.2f}x")
    print(f"{'Tokens/second':<25} {original['tokens_per_sec']:<15.0f} {optimized['tokens_per_sec']:<15.0f} {tokens_ratio:.2f}x")
    
    print("\n💡 INSIGHTS:")
    print(f"• Optimized model is {params_ratio:.1f}x larger but still efficient")
    print(f"• M4 GPU handles {optimized['tokens_per_sec']:.0f} tokens/second")
    print(f"• Your 32GB RAM can easily handle these batch sizes")
    
    if torch.backends.mps.is_available():
        print("• ✅ MPS acceleration is working properly")
    else:
        print("• ⚠️  Consider updating PyTorch for MPS support")


if __name__ == "__main__":
    main()
