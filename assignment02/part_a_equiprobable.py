"""
Part (a): Random 4-letter words with equiprobable letters
========================================================

This module generates random 4-letter words by selecting each letter independently
with equal probability (1/26 for each letter).
"""

import numpy as np
import random

def generate_equiprobable_words(num_words=100):
    """
    Generate random 4-letter words with equiprobable letters.
    
    Args:
        num_words (int): Number of words to generate (default: 100)
    
    Returns:
        list: List of generated 4-letter words
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    words = []
    
    for _ in range(num_words):
        # Generate each letter independently with equal probability
        word = ''.join(random.choice(letters) for _ in range(4))
        words.append(word)
    
    return words

def test_equiprobable_generation():
    """
    Test function to verify equiprobable letter generation.
    """
    print("Testing equiprobable letter generation...")
    words = generate_equiprobable_words(10)
    print(f"Generated words: {words}")
    
    # Check letter distribution
    all_letters = ''.join(words)
    letter_counts = {}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        letter_counts[letter] = all_letters.count(letter)
    
    print(f"Letter counts: {letter_counts}")
    print("Expected: roughly equal distribution across all letters")

if __name__ == "__main__":
    test_equiprobable_generation()