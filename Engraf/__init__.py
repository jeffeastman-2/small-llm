# ENGRAF package
# 1975 N-Space Model for Visual and Verbal Concepts
# Modern Python implementation

from .vector_space import VectorSpace, vector_from_features, VECTOR_DIMENSIONS, VECTOR_LENGTH
from .vocabulary import SEMANTIC_VECTOR_SPACE

__all__ = ['VectorSpace', 'vector_from_features', 'VECTOR_DIMENSIONS', 'VECTOR_LENGTH', 'SEMANTIC_VECTOR_SPACE']
