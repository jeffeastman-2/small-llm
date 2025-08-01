"""
ENGRAF-Small-LLM Integration Module

This module provides the foundation for integrating the BPE-trained Small-LLM
transformer with the ENGRAF semantic vector system.

The integration maps between:
- Small-LLM: 512-dimensional BPE embeddings from transformer model
- ENGRAF: Real ENGRAF VectorSpace with actual VECTOR_DIMENSIONS

Historical context: Bridging 50 years from the 1975 "N-Space Model for Visual 
and Verbal Concepts" dissertation to modern implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Import actual ENGRAF classes
from Engraf.vector_space import VectorSpace, vector_from_features, VECTOR_DIMENSIONS, VECTOR_LENGTH
from Engraf.vocabulary import SEMANTIC_VECTOR_SPACE

class SemanticVectorMapper:
    """
    Maps between Small-LLM BPE embeddings and ENGRAF VectorSpace objects.
    
    This class handles the dimensional transformation from 512-dim transformer
    embeddings to the actual ENGRAF semantic vector space.
    """
    
    def __init__(self, bpe_model, tokenizer):
        """
        Initialize the semantic vector mapper.
        
        Args:
            bpe_model: Trained Small-LLM transformer model
            tokenizer: BPE tokenizer used during training
        """
        self.bpe_model = bpe_model
        self.tokenizer = tokenizer
        self.device = next(bpe_model.parameters()).device
        
        # Use actual ENGRAF dimensions
        self.engraf_dimensions = VECTOR_DIMENSIONS
        self.vector_length = VECTOR_LENGTH
        
        print(f"ENGRAF Integration: Using {self.vector_length} dimensions")
        print(f"Dimensions: {self.engraf_dimensions}")
        
        # Learned mapping from 512-dim to ENGRAF dimensions
        self.dimension_mappers = {}
        self._initialize_mappers()
        
        # Cache for frequent mappings
        self.embedding_cache = {}
        
    def _initialize_mappers(self):
        """
        Initialize dimension-specific mapping functions.
        These map from 512-dim BPE space to individual ENGRAF dimensions.
        """
        for dim_name in self.engraf_dimensions:
            # Create a simple linear mapper for each dimension
            # In practice, these could be learned through training
            mapper = torch.nn.Linear(512, 1)
            mapper.to(self.device)  # Move to same device as model
            self.dimension_mappers[dim_name] = mapper
    
    def get_bpe_embedding(self, word: str) -> torch.Tensor:
        """
        Get BPE embedding for a word using the trained model.
        
        Args:
            word: Input word to embed
            
        Returns:
            512-dimensional embedding tensor
        """
        if word in self.embedding_cache:
            return self.embedding_cache[word]
        
        # Tokenize the word - handle HuggingFace tokenizer output
        if hasattr(self.tokenizer, 'encode'):
            # HuggingFace tokenizer
            encoding = self.tokenizer.encode(word)
            if hasattr(encoding, 'ids'):
                tokens = encoding.ids
            else:
                tokens = encoding
        else:
            # Our custom tokenizer
            tokens = self.tokenizer.encode(word)
            
        token_ids = torch.tensor([tokens], device=self.device)
        
        # Get embeddings from the model
        with torch.no_grad():
            # Get the embedding layer output
            if hasattr(self.bpe_model, 'token_embed'):
                # BPE model structure
                embeddings = self.bpe_model.token_embed(token_ids)
            else:
                # Original model structure
                embeddings = self.bpe_model.embedding(token_ids)
            # Average embeddings if multiple tokens
            word_embedding = embeddings.mean(dim=1).squeeze()
        
        self.embedding_cache[word] = word_embedding
        return word_embedding
    
    def map_to_engraf_vector(self, word: str) -> VectorSpace:
        """
        Map a word to an ENGRAF VectorSpace object.
        
        Args:
            word: Input word to map
            
        Returns:
            ENGRAF VectorSpace object with predicted semantic features
        """
        # Get BPE embedding
        bpe_embedding = self.get_bpe_embedding(word)
        
        # Create VectorSpace object
        vector_space = VectorSpace(word=word)
        
        # Map to each ENGRAF dimension
        for dim_name in self.engraf_dimensions:
            # Use the dimension-specific mapper
            with torch.no_grad():
                mapped_value = self.dimension_mappers[dim_name](bpe_embedding.unsqueeze(0))
                
            # Apply sigmoid to get 0-1 range, then scale appropriately
            normalized_value = torch.sigmoid(mapped_value).item()
            
            # Set different ranges for different dimension types
            if dim_name in ['locX', 'locY', 'locZ']:
                # Spatial location: -10 to 10
                scaled_value = -10 + normalized_value * 20
            elif dim_name in ['scaleX', 'scaleY', 'scaleZ']:
                # Scale: -2 to 3 
                scaled_value = -2 + normalized_value * 5
            elif dim_name in ['number']:
                # Number: 0 to 10
                scaled_value = normalized_value * 10
            elif dim_name in ['adv']:
                # Adverb modifier: 0.5 to 2.0
                scaled_value = 0.5 + normalized_value * 1.5
            elif dim_name in ['texture', 'transparency']:
                # Visual properties: 0 to 2
                scaled_value = normalized_value * 2
            elif dim_name in ['spatial_vertical', 'directional_target']:
                # Bipolar semantic: -1 to 1
                scaled_value = -1 + normalized_value * 2
            else:
                # Most dimensions: 0 to 1
                scaled_value = normalized_value
                
            vector_space[dim_name] = scaled_value
        
        return vector_space
    
    def analyze_semantic_similarity(self, word1: str, word2: str) -> Dict[str, float]:
        """
        Analyze semantic similarity between two words using ENGRAF VectorSpace.
        
        Args:
            word1, word2: Words to compare
            
        Returns:
            Dictionary of similarity scores by dimension category
        """
        vector1 = self.map_to_engraf_vector(word1)
        vector2 = self.map_to_engraf_vector(word2)
        
        similarities = {}
        
        # Define dimension categories
        categories = {
            'linguistic': ['verb', 'tobe', 'action', 'prep', 'det', 'def', 'adv', 'adj', 'noun', 'pronoun', 'conj', 'disj', 'unit', 'comp'],
            'spatial': ['locX', 'locY', 'locZ', 'scaleX', 'scaleY', 'scaleZ', 'rotX', 'rotY', 'rotZ'],
            'visual': ['red', 'green', 'blue', 'texture', 'transparency'],
            'semantic': ['spatial_vertical', 'spatial_proximity', 'directional_target', 'directional_agency', 'relational_possession', 'relational_comparison'],
            'action': ['create', 'transform', 'style', 'organize', 'edit', 'select'],
            'grammar': ['number', 'vector', 'singular', 'plural', 'quoted']
        }
        
        # Calculate similarity for each category
        for category, dims in categories.items():
            category_dims = [d for d in dims if d in self.engraf_dimensions]
            if category_dims:
                vals1 = np.array([vector1[d] for d in category_dims])
                vals2 = np.array([vector2[d] for d in category_dims])
                
                # Cosine similarity
                dot_product = np.dot(vals1, vals2)
                norm1 = np.linalg.norm(vals1)
                norm2 = np.linalg.norm(vals2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                else:
                    similarity = 0.0
                    
                similarities[category] = float(similarity)
        
        return similarities
    
    def get_engraf_vocabulary_words(self) -> List[str]:
        """Get the list of words from ENGRAF's semantic vector space."""
        return list(SEMANTIC_VECTOR_SPACE.keys())


class ENGRAFIntegrationBridge:
    """
    Main integration bridge between Small-LLM and ENGRAF systems.
    Uses actual ENGRAF VectorSpace objects and vocabulary.
    """
    
    def __init__(self, small_llm_model, tokenizer):
        """
        Initialize the integration bridge.
        
        Args:
            small_llm_model: Trained Small-LLM model (already loaded)
            tokenizer: BPE tokenizer (already loaded)
        """
        self.model = small_llm_model
        self.tokenizer = tokenizer
        
        # Initialize the semantic vector mapper
        self.mapper = SemanticVectorMapper(self.model, self.tokenizer)
        
    def generate_engraf_vocabulary_vectors(self, words: List[str] = None) -> Dict[str, VectorSpace]:
        """
        Generate ENGRAF VectorSpace objects for a vocabulary list.
        
        Args:
            words: List of words to generate vectors for (defaults to ENGRAF vocabulary)
            
        Returns:
            Dictionary mapping words to VectorSpace objects
        """
        if words is None:
            words = self.mapper.get_engraf_vocabulary_words()
            
        vocabulary = {}
        
        for word in words:
            try:
                vector_space = self.mapper.map_to_engraf_vector(word)
                vocabulary[word] = vector_space
                print(f"Generated VectorSpace for '{word}'")
            except Exception as e:
                print(f"Failed to process '{word}': {e}")
                
        return vocabulary
    
    def compare_with_original_engraf(self, word: str) -> Dict:
        """
        Compare Small-LLM generated vector with original ENGRAF vector.
        
        Args:
            word: Word to compare
            
        Returns:
            Comparison dictionary with similarities and differences
        """
        if word not in SEMANTIC_VECTOR_SPACE:
            return {'error': f"Word '{word}' not in original ENGRAF vocabulary"}
        
        # Get original ENGRAF vector
        original_vector = SEMANTIC_VECTOR_SPACE[word]
        
        # Get Small-LLM generated vector
        generated_vector = self.mapper.map_to_engraf_vector(word)
        
        # Compare dimensions
        comparison = {
            'word': word,
            'dimensions_compared': len(VECTOR_DIMENSIONS),
            'differences': {},
            'cosine_similarity': 0.0
        }
        
        # Calculate differences for each dimension
        total_diff = 0
        for dim in VECTOR_DIMENSIONS:
            orig_val = original_vector[dim] if dim in VECTOR_DIMENSIONS else 0
            gen_val = generated_vector[dim]
            diff = abs(orig_val - gen_val)
            comparison['differences'][dim] = {
                'original': float(orig_val),
                'generated': float(gen_val), 
                'difference': float(diff)
            }
            total_diff += diff
        
        # Calculate overall similarity
        orig_array = np.array([original_vector[d] for d in VECTOR_DIMENSIONS])
        gen_array = np.array([generated_vector[d] for d in VECTOR_DIMENSIONS])
        
        dot_product = np.dot(orig_array, gen_array)
        norm_orig = np.linalg.norm(orig_array)
        norm_gen = np.linalg.norm(gen_array)
        
        if norm_orig > 0 and norm_gen > 0:
            comparison['cosine_similarity'] = float(dot_product / (norm_orig * norm_gen))
        
        comparison['average_difference'] = total_diff / len(VECTOR_DIMENSIONS)
        
        return comparison
    
    def compare_with_original_engraf(self, word: str):
        """
        Compare Small-LLM generated vector with original ENGRAF vector.
        
        Args:
            word: Word to compare
            
        Returns:
            Dictionary with comparison results
        """
        if word not in SEMANTIC_VECTOR_SPACE:
            return None
        
        # Get original ENGRAF vector
        original_vector_space = SEMANTIC_VECTOR_SPACE[word]
        original_features = original_vector_space.features
        
        # Generate Small-LLM vector
        generated_vector_space = self.mapper.map_to_engraf_vector(word)
        generated_features = generated_vector_space.features
        
        # Calculate cosine similarity
        cosine_sim = np.dot(original_features, generated_features) / (
            np.linalg.norm(original_features) * np.linalg.norm(generated_features)
        )
        
        # Calculate per-dimension differences
        differences = {}
        for i, dim_name in enumerate(VECTOR_DIMENSIONS):
            if i < len(original_features) and i < len(generated_features):
                orig_val = original_features[i]
                gen_val = generated_features[i]
                differences[dim_name] = {
                    'original': float(orig_val),
                    'generated': float(gen_val),
                    'difference': float(abs(orig_val - gen_val))
                }
        
        # Calculate average difference
        avg_diff = np.mean([d['difference'] for d in differences.values()])
        
        return {
            'word': word,
            'cosine_similarity': float(cosine_sim),
            'average_difference': float(avg_diff),
            'differences': differences
        }
    
    def save_integration_results(self, vocabulary: Dict[str, VectorSpace], output_path: str):
        """
        Save the generated VectorSpace objects for use with ENGRAF.
        
        Args:
            vocabulary: Generated word-to-VectorSpace mappings
            output_path: Path to save the results
        """
        # Convert VectorSpace objects to serializable format
        serializable_vocab = {}
        for word, vector_space in vocabulary.items():
            vector_dict = {}
            for dim in VECTOR_DIMENSIONS:
                vector_dict[dim] = float(vector_space[dim])
            serializable_vocab[word] = vector_dict
        
        output_data = {
            'metadata': {
                'description': 'Small-LLM to ENGRAF VectorSpace integration',
                'model_performance': 'loss 0.0020 train/0.0017 validation',
                'vector_dimensions': len(VECTOR_DIMENSIONS),
                'vocabulary_size': len(vocabulary),
                'generation_date': '2025-08-01',
                'engraf_dimensions': VECTOR_DIMENSIONS
            },
            'vocabulary_vectors': serializable_vocab
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Integration results saved to: {output_path}")


def test_engraf_integration():
    """
    Test the integration with actual ENGRAF vocabulary and vectors.
    """
    print("Testing ENGRAF Integration with Real VectorSpace Objects")
    print("=" * 60)
    
    # Show ENGRAF dimensions
    print(f"ENGRAF Vector Dimensions ({len(VECTOR_DIMENSIONS)}):")
    for i, dim in enumerate(VECTOR_DIMENSIONS):
        print(f"  {i:2d}: {dim}")
    
    # Show sample ENGRAF vocabulary
    print(f"\nSample ENGRAF Vocabulary ({len(SEMANTIC_VECTOR_SPACE)} words):")
    sample_words = list(SEMANTIC_VECTOR_SPACE.keys())[:10]
    for word in sample_words:
        vector = SEMANTIC_VECTOR_SPACE[word]
        active_dims = [dim for dim in VECTOR_DIMENSIONS if vector[dim] > 0]
        print(f"  {word}: {active_dims}")
    
    return sample_words


if __name__ == "__main__":
    # Test the integration framework with real ENGRAF data
    test_words = test_engraf_integration()
    print(f"\nReady to integrate {len(SEMANTIC_VECTOR_SPACE)} ENGRAF words with Small-LLM")
    print("Next step: Run integration with trained model")
