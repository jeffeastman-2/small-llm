# Small Language Model (Mini-GPT)

A PyTorch implementation of a small GPT-style transformer model that can be trained on PDF documents and generate text.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **PDF Text Extraction**: Robust multi-file PDF processing with error handling
- **Advanced Tokenization**: Word-based tokenization with vocabulary management
- **Modern Transformer**: Pre-norm architecture with causal attention masking
- **Flexible Training**: Configurable training pipeline with early stopping
- **Interactive Generation**: Feature-rich text generation with multiple sampling strategies
- **Multiple Sampling Methods**: Top-k and top-p (nucleus) sampling support
- **Device Agnostic**: Automatic GPU/MPS detection and fallback to CPU

## Project Structure

```
Small-LLM/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # PDF text extraction
│   ├── tokenizer.py             # Tokenization and vocabulary
│   ├── dataset.py               # PyTorch dataset
│   ├── model.py                 # MiniGPT model implementation
│   ├── trainer.py               # Training utilities
│   └── generation.py            # Text generation utilities
├── tests/                       # Comprehensive test suite
│   ├── __init__.py
│   ├── test_tokenizer.py        # Tokenizer tests
│   ├── test_dataset.py          # Dataset tests
│   ├── test_model.py            # Model tests
│   ├── test_generation.py       # Generation tests
│   ├── test_data_loader.py      # Data loading tests
│   └── test_integration.py      # Integration tests
├── train.py                     # Main training script
├── generate.py                  # Interactive text generation
├── run_tests.py                 # Test runner script
├── test_with_pytest.py          # Alternative pytest runner
├── Training/                    # PDF training documents
├── requirements.txt             # Python dependencies
├── pytest.ini                  # Pytest configuration
├── TESTING.md                   # Testing documentation
├── *.pth                        # Saved model checkpoints
└── *.pt                         # Vocabulary files
```

## Model Architecture

- **Embedding Dimension**: 64
- **Context Length**: 10 tokens
- **Number of Heads**: 2
- **Number of Layers**: 2
- **Attention**: Causal self-attention with masking

## Testing

The project includes a comprehensive test suite with 50+ tests covering all modules:

```bash
# Run all tests
python run_tests.py

# Run specific test module
python run_tests.py --test test_tokenizer

# Run with pytest (includes coverage)
python test_with_pytest.py
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually install the core dependencies:
```bash
pip install torch PyMuPDF
```

## Usage

### Training

1. Place PDF files in the `Training/` directory
2. Run the training script:
```bash
python train.py
```

### Generation

After training, use the interactive generation script:
```bash
python generate.py
```

The generation script provides an interactive interface with commands:
- Enter any text prompt to generate continuations
- `help` - Show available commands and settings
- `config` - Display current generation parameters
- `set <param> <value>` - Adjust generation settings
- `quit` or `exit` - Exit the program

## Training Features

- **Early Stopping**: Stops training if no improvement for 20 epochs
- **Gradient Clipping**: Prevents exploding gradients
- **Best Model Saving**: Automatically saves the best performing model
- **Progress Monitoring**: Regular generation samples during training
- **Device Support**: Automatic GPU/MPS detection

## Generation Features

- **Top-k Sampling**: Configurable top-k sampling for diverse outputs
- **Temperature Control**: Adjustable temperature for creativity/coherence balance
- **Interactive Mode**: Command-line interface for real-time generation
- **Flexible Input**: Support for both token sequences and text strings

## Model Parameters

- Vocabulary size: Determined by training data
- Context window: 10 tokens
- Batch size: 32
- Learning rate: 1e-4
- Max epochs: 500 (with early stopping)

## Module Overview

### Core Modules

- **`data_loader.py`**: PDF text extraction with robust error handling
- **`tokenizer.py`**: Text tokenization, vocabulary building, and persistence
- **`dataset.py`**: PyTorch dataset for next-token prediction training
- **`model.py`**: MiniGPT transformer implementation with modern architecture
- **`trainer.py`**: Training orchestration with monitoring and checkpointing
- **`generation.py`**: Text generation with multiple sampling strategies

### Scripts

- **`train.py`**: Complete training pipeline from PDFs to trained model
- **`generate.py`**: Interactive text generation interface

## Future Improvements

- [ ] Add BPE/SentencePiece tokenization for better subword handling
- [ ] Implement beam search and other advanced decoding strategies
- [ ] Add validation set evaluation and perplexity metrics
- [ ] Support for larger context windows and streaming attention
- [ ] Model architecture experiments (different sizes, attention variants)
- [ ] Better text preprocessing and document structure preservation
- [ ] Multi-GPU training support
- [ ] Model quantization and optimization for inference
- [ ] Web interface for text generation
- [ ] Fine-tuning capabilities for domain adaptation
