# Small Language Model (Mini-GPT)

A PyTorch implementation of a small GPT-style transformer model that can be trained on PDF documents and generate text.

## Features

- **PDF Text Extraction**: Reads and processes multiple PDF files from a training directory
- **Custom Tokenization**: Simple word-based tokenization with vocabulary building
- **Mini GPT Architecture**: Multi-layer transformer with self-attention
- **Training Pipeline**: Complete training loop with early stopping and model checkpointing
- **Text Generation**: Interactive text generation with top-k sampling
- **Multiple Inference Options**: Generate from token sequences or string prompts

## Files

- `Reading from Trainning Directory3.py` - Main training script
- `Generate from Best Model.py` - Interactive text generation script
- `Training/` - Directory containing PDF training documents
- `*.pth` - Saved model checkpoints
- `*.pt` - Vocabulary files

## Model Architecture

- **Embedding Dimension**: 64
- **Context Length**: 10 tokens
- **Number of Heads**: 2
- **Number of Layers**: 2
- **Attention**: Causal self-attention with masking

## Requirements

```bash
pip install torch PyMuPDF
```

## Usage

### Training

1. Place PDF files in the `Training/` directory
2. Run the training script:
```bash
python "Reading from Trainning Directory3.py"
```

### Generation

After training, use the generation script:
```bash
python "Generate from Best Model.py"
```

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

## File Structure

```
Small-LLM/
├── Reading from Trainning Directory3.py  # Training script
├── Generate from Best Model.py           # Generation script
├── Training/                             # PDF training data
│   ├── dissertationPurchase_JEastman_1749111609.pdf
│   └── Too Early Retrospective.pdf
├── best_model.pth                        # Best model weights
├── vocab.pt                              # Training vocabulary
├── use_model.pth                         # Generation model weights
└── use_vocab.pt                          # Generation vocabulary
```

## Future Improvements

- [ ] Add BPE tokenization for better subword handling
- [ ] Implement beam search for generation
- [ ] Add validation set evaluation
- [ ] Support for larger context windows
- [ ] Model architecture experiments (different sizes)
- [ ] Better text preprocessing and cleaning
