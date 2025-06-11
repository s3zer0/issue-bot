"""
Detection module for hallucination detection and keyword generation.
"""

# Import only what's needed to avoid circular imports
from .keyword_generator import generate_keywords_for_topic

__all__ = [
    'generate_keywords_for_topic'
]