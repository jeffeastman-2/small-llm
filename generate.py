#!/usr/bin/env python3
"""
Interactive text generation using a trained Mini GPT model.

This script loads a trained model and vocabulary, then provides
an interactive interface for text generation.
"""

import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MiniGPT
from src.tokenizer import load_vocab
from src.generation import generate_from_string


class TextGenerator:
    """Interactive text generator."""
    
    def __init__(self, model_path: str, vocab_path: str, context_len: int = 10):
        """
        Initialize the text generator.
        
        Args:
            model_path: Path to trained model
            vocab_path: Path to vocabulary file
            context_len: Context length used during training
        """
        self.context_len = context_len
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}...")
        self.word2idx, self.idx2word = load_vocab(vocab_path)
        vocab_size = len(self.word2idx)
        
        # Initialize and load model
        print(f"Initializing model (vocab_size={vocab_size}, context_len={context_len})...")
        self.model = MiniGPT(vocab_size=vocab_size, context_len=context_len)
        
        print(f"Loading model weights from {model_path}...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {self.model.get_num_trainable_params():,}")
        print(f"   Vocabulary size: {vocab_size}")
    
    def generate(
        self, 
        prompt: str, 
        steps: int = 20, 
        temperature: float = 1.0, 
        top_k: int = 10,
        top_p: float = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt to start generation
            steps: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated text
        """
        with torch.no_grad():
            return generate_from_string(
                self.model, prompt, self.word2idx, self.idx2word,
                context_len=self.context_len, steps=steps,
                temperature=temperature, top_k=top_k, top_p=top_p
            )
    
    def interactive_mode(self):
        """Run interactive text generation."""
        print("\n" + "="*60)
        print("🎯 INTERACTIVE TEXT GENERATION")
        print("="*60)
        print("Enter prompts to generate text. Commands:")
        print("  'quit' or 'exit' - Exit the program")
        print("  'help' - Show this help message")
        print("  'config' - Show current generation settings")
        print("  'set <param> <value>' - Change generation parameters")
        print("="*60 + "\n")
        
        # Default generation settings
        settings = {
            'steps': 20,
            'temperature': 1.0,
            'top_k': 10,
            'top_p': None
        }
        
        while True:
            try:
                prompt = input("Prompt> ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif prompt.lower() == 'help':
                    self._show_help()
                    continue
                
                elif prompt.lower() == 'config':
                    self._show_config(settings)
                    continue
                
                elif prompt.lower().startswith('set '):
                    self._handle_config_change(prompt, settings)
                    continue
                
                # Generate text
                print("Generating...", end=" ", flush=True)
                generated = self.generate(
                    prompt, 
                    steps=settings['steps'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p']
                )
                print(f"\n> {generated}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help message."""
        print("\n📖 HELP:")
        print("  Enter any text prompt to generate a continuation")
        print("  Examples:")
        print("    'the model could'")
        print("    'in this paper we'")
        print("    'the results show that'")
        print("\n⚙️  SETTINGS:")
        print("  steps - Number of tokens to generate (default: 20)")
        print("  temperature - Randomness (0.1=conservative, 2.0=creative)")
        print("  top_k - Consider top K tokens (1-50, default: 10)")
        print("  top_p - Nucleus sampling threshold (0.1-1.0, default: None)")
        print("\n💡 TIPS:")
        print("  - Lower temperature for more coherent text")
        print("  - Higher temperature for more creative text")
        print("  - Lower top_k for more focused generation")
        print("  - Use top_p instead of top_k for dynamic token selection")
        print()
    
    def _show_config(self, settings):
        """Show current configuration."""
        print(f"\n⚙️  CURRENT SETTINGS:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        print()
    
    def _handle_config_change(self, command, settings):
        """Handle configuration changes."""
        try:
            parts = command.split()
            if len(parts) != 3:
                print("❌ Usage: set <parameter> <value>")
                return
            
            param = parts[1].lower()
            value = parts[2]
            
            if param == 'steps':
                settings['steps'] = max(1, min(100, int(value)))
                print(f"✅ Set steps to {settings['steps']}")
            
            elif param == 'temperature':
                settings['temperature'] = max(0.1, min(3.0, float(value)))
                print(f"✅ Set temperature to {settings['temperature']}")
            
            elif param == 'top_k':
                settings['top_k'] = max(1, min(50, int(value)))
                print(f"✅ Set top_k to {settings['top_k']}")
            
            elif param == 'top_p':
                if value.lower() == 'none':
                    settings['top_p'] = None
                    print("✅ Set top_p to None (using top_k)")
                else:
                    settings['top_p'] = max(0.1, min(1.0, float(value)))
                    print(f"✅ Set top_p to {settings['top_p']}")
            
            else:
                print(f"❌ Unknown parameter: {param}")
                print("Available parameters: steps, temperature, top_k, top_p")
        
        except ValueError:
            print("❌ Invalid value format")


def main():
    """Main function."""
    # Configuration
    model_path = "use_model.pth"  # Can be changed to "best_model.pth"
    vocab_path = "use_vocab.pt"   # Can be changed to "vocab.pt"
    context_len = 10
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available model files:")
        for f in os.listdir("."):
            if f.endswith(".pth"):
                print(f"  - {f}")
        return
    
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary file not found: {vocab_path}")
        print("Available vocabulary files:")
        for f in os.listdir("."):
            if f.endswith(".pt"):
                print(f"  - {f}")
        return
    
    try:
        # Initialize generator
        generator = TextGenerator(model_path, vocab_path, context_len)
        
        # Run interactive mode
        generator.interactive_mode()
        
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")


if __name__ == "__main__":
    main()
