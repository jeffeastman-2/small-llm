"""
Text generation utilities and sampling strategies.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from .tokenizer import tokenize


def sample_top_k(probs: torch.Tensor, k: int = 10) -> int:
    """
    Sample from top-k most likely tokens.
    
    Args:
        probs (torch.Tensor): Probability distribution over vocabulary
        k (int): Number of top tokens to consider
        
    Returns:
        int: Sampled token ID
    """
    topk = torch.topk(probs, k)
    probs_topk = F.softmax(topk.values, dim=-1)
    idx = torch.multinomial(probs_topk, num_samples=1).item()
    return topk.indices[idx].item()


def sample_top_p(probs: torch.Tensor, p: float = 0.9) -> int:
    """
    Sample from nucleus (top-p) of probability distribution.
    
    Args:
        probs (torch.Tensor): Probability distribution over vocabulary
        p (float): Cumulative probability threshold
        
    Returns:
        int: Sampled token ID
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_probs, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0
    
    # Set probabilities to 0 for removed indices
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0
    
    # Renormalize and sample
    probs = F.softmax(probs, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate(
    model, 
    seed: List[int], 
    idx2word: Dict[int, str], 
    word2idx: Dict[str, int], 
    steps: int = 20, 
    temperature: float = 1.0, 
    top_k: int = 10,
    top_p: float = None
) -> str:
    """
    Generate text from token seed.
    
    Args:
        model: Trained model
        seed (List[int]): Initial token sequence
        idx2word (Dict[int, str]): Index to word mapping
        word2idx (Dict[str, int]): Word to index mapping
        steps (int): Number of tokens to generate
        temperature (float): Sampling temperature
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p sampling parameter (if None, use top-k)
        
    Returns:
        str: Generated text
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Use only the last context_len tokens
    context_len = model.context_len
    seed = seed[-context_len:]
    result = []

    for _ in range(steps):
        # Prepare input
        x = torch.tensor(seed).unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(x)[0] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            if top_p is not None:
                next_token = sample_top_p(probs, p=top_p)
            else:
                next_token = sample_top_k(probs, k=top_k)

        # Add to result and update seed
        result.append(idx2word.get(next_token, "<UNK>"))
        seed = seed[1:] + [next_token]

    return ' '.join(result)


def generate_from_string(
    model, 
    seed_text: str, 
    word2idx: Dict[str, int], 
    idx2word: Dict[int, str], 
    context_len: int = 10, 
    steps: int = 20, 
    temperature: float = 1.0, 
    top_k: int = 10,
    top_p: float = None
) -> str:
    """
    Generate text from string seed.
    
    Args:
        model: Trained model
        seed_text (str): Initial text prompt
        word2idx (Dict[str, int]): Word to index mapping
        idx2word (Dict[int, str]): Index to word mapping
        context_len (int): Context length
        steps (int): Number of tokens to generate
        temperature (float): Sampling temperature
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p sampling parameter (if None, use top-k)
        
    Returns:
        str: Generated text
    """
    # Tokenize and encode the seed text
    words = tokenize(seed_text)
    tokens = [word2idx.get(w, word2idx['<UNK>']) for w in words]
    
    # Pad if necessary
    if len(tokens) < context_len:
        tokens = [word2idx['<PAD>']] * (context_len - len(tokens)) + tokens
    
    # Take only the last context_len tokens
    tokens = tokens[-context_len:]
    
    return generate(
        model, tokens, idx2word, word2idx, 
        steps, temperature, top_k, top_p
    )
