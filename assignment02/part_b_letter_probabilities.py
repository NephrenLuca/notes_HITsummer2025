"""
Part (b): Letter probability estimation and word generation
========================================================

This module estimates letter probabilities from text and generates random
4-letter words using these estimated probabilities.
"""

import numpy as np
from collections import Counter
import random

def estimate_letter_probabilities(words):
    """
    Estimate letter probabilities from a list of words.
    
    Args:
        words (list): List of words (strings of letters)
    
    Returns:
        dict: Dictionary mapping each letter to its probability
    """
    # Count all letters in all words
    all_letters = ''.join(words)
    letter_counts = Counter(all_letters)
    
    # Calculate total number of letters
    total_letters = sum(letter_counts.values())
    
    # Calculate probabilities
    letter_probs = {}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        letter_probs[letter] = letter_counts.get(letter, 0) / total_letters
    
    return letter_probs

def generate_letter_prob_words(letter_probs, num_words=100):
    """
    Generate random 4-letter words using estimated letter probabilities.
    
    Args:
        letter_probs (dict): Dictionary mapping letters to their probabilities
        num_words (int): Number of words to generate (default: 100)
    
    Returns:
        list: List of generated 4-letter words
    """
    letters = list(letter_probs.keys())
    probs = list(letter_probs.values())
    
    words = []
    for _ in range(num_words):
        # Generate each letter according to the estimated probabilities
        word = ''.join(np.random.choice(letters, p=probs) for _ in range(4))
        words.append(word)
    
    return words

def print_letter_statistics(letter_probs, words):
    """
    Print statistics about letter probabilities and generated words.
    
    Args:
        letter_probs (dict): Estimated letter probabilities
        words (list): Generated words
    """
    print("\nLetter Probability Statistics:")
    print("-" * 40)
    
    # Sort letters by probability (descending)
    sorted_letters = sorted(letter_probs.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 most frequent letters:")
    for i, (letter, prob) in enumerate(sorted_letters[:10]):
        print(f"  {letter}: {prob:.4f}")
    
    # Analyze generated words
    all_letters = ''.join(words)
    generated_counts = Counter(all_letters)
    total_generated = len(all_letters)
    
    print(f"\nGenerated word statistics:")
    print(f"  Total words: {len(words)}")
    print(f"  Total letters: {total_generated}")
    
    print("\nLetter distribution in generated words:")
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        count = generated_counts.get(letter, 0)
        prob = count / total_generated if total_generated > 0 else 0
        print(f"  {letter}: {count} ({prob:.4f})")

def test_letter_probabilities():
    """
    Test function to verify letter probability estimation and generation.
    """
    print("Testing letter probability estimation...")
    
    # Test with sample words
    test_words = ['hello', 'world', 'python', 'markov', 'chain', 'text', 'generation']
    letter_probs = estimate_letter_probabilities(test_words)
    
    print(f"Estimated letter probabilities: {letter_probs}")
    
    # Generate words using these probabilities
    words = generate_letter_prob_words(letter_probs, 10)
    print(f"Generated words: {words}")
    
    print_letter_statistics(letter_probs, words)

if __name__ == "__main__":
    test_letter_probabilities()