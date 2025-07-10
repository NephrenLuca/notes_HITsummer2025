"""
Part (c): First Order Markov Chain
=================================

This module estimates first order transition probabilities P(x_{n+1} | x_n)
and generates random 4-letter words using these probabilities.
"""

import numpy as np
from collections import defaultdict, Counter
import random

def estimate_transition_probabilities(words):
    """
    Estimate first order transition probabilities P(x_{n+1} | x_n).
    
    Args:
        words (list): List of words (strings of letters)
    
    Returns:
        dict: Dictionary mapping (current_letter, next_letter) to probability
    """
    # Count transitions within words only (not between words)
    transition_counts = defaultdict(Counter)
    
    for word in words:
        if len(word) < 2:
            continue
        
        # Count transitions within the word
        for i in range(len(word) - 1):
            current_letter = word[i]
            next_letter = word[i + 1]
            transition_counts[current_letter][next_letter] += 1
    
    # Calculate transition probabilities
    transition_probs = {}
    for current_letter in 'abcdefghijklmnopqrstuvwxyz':
        total_transitions = sum(transition_counts[current_letter].values())
        
        if total_transitions == 0:
            # If no transitions from this letter, use uniform distribution
            for next_letter in 'abcdefghijklmnopqrstuvwxyz':
                transition_probs[(current_letter, next_letter)] = 1.0 / 26
        else:
            # Calculate conditional probabilities
            for next_letter in 'abcdefghijklmnopqrstuvwxyz':
                count = transition_counts[current_letter][next_letter]
                transition_probs[(current_letter, next_letter)] = count / total_transitions
    
    return transition_probs

def generate_first_order_markov_words(letter_probs, transition_probs, num_words=100):
    """
    Generate random 4-letter words using first order Markov chain.
    
    Args:
        letter_probs (dict): Initial letter probabilities
        transition_probs (dict): Transition probabilities
        num_words (int): Number of words to generate (default: 100)
    
    Returns:
        list: List of generated 4-letter words
    """
    letters = list(letter_probs.keys())
    initial_probs = list(letter_probs.values())
    
    words = []
    for _ in range(num_words):
        word = ""
        
        # Generate first letter using initial probabilities
        first_letter = np.random.choice(letters, p=initial_probs)
        word += first_letter
        
        # Generate remaining letters using transition probabilities
        for _ in range(3):
            current_letter = word[-1]
            
            # Get transition probabilities for current letter
            next_letter_probs = []
            for next_letter in letters:
                prob = transition_probs.get((current_letter, next_letter), 0)
                next_letter_probs.append(prob)
            
            # Normalize probabilities (in case they don't sum to 1)
            total_prob = sum(next_letter_probs)
            if total_prob > 0:
                next_letter_probs = [p / total_prob for p in next_letter_probs]
            else:
                # Fallback to uniform distribution
                next_letter_probs = [1.0 / 26] * 26
            
            # Generate next letter
            next_letter = np.random.choice(letters, p=next_letter_probs)
            word += next_letter
        
        words.append(word)
    
    return words

def print_transition_statistics(transition_probs, words):
    """
    Print statistics about transition probabilities and generated words.
    
    Args:
        transition_probs (dict): Transition probabilities
        words (list): Generated words
    """
    print("\nFirst Order Markov Chain Statistics:")
    print("-" * 45)
    
    # Show some example transition probabilities
    print("Sample transition probabilities:")
    sample_letters = ['a', 'e', 'i', 'o', 'u', 't', 'h', 's']
    for letter in sample_letters:
        print(f"\nP(next_letter | '{letter}'):")
        transitions = [(next_letter, prob) for (curr, next_letter), prob in transition_probs.items() 
                      if curr == letter and prob > 0.01]
        transitions.sort(key=lambda x: x[1], reverse=True)
        for next_letter, prob in transitions[:5]:
            print(f"  '{letter}' -> '{next_letter}': {prob:.4f}")
    
    # Analyze generated words
    print(f"\nGenerated word statistics:")
    print(f"  Total words: {len(words)}")
    
    # Count letter pairs in generated words
    pair_counts = Counter()
    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += 1
    
    print(f"  Total letter pairs: {sum(pair_counts.values())}")
    
    # Show most common letter pairs
    print("\nMost common letter pairs in generated words:")
    for pair, count in pair_counts.most_common(10):
        print(f"  '{pair[0]}{pair[1]}': {count}")

def test_first_order_markov():
    """
    Test function to verify first order Markov chain implementation.
    """
    print("Testing first order Markov chain...")
    
    # Test with sample words
    test_words = ['hello', 'world', 'python', 'markov', 'chain', 'text', 'generation', 'probability']
    letter_probs = {'a': 0.1, 'b': 0.05, 'c': 0.05, 'd': 0.05, 'e': 0.15, 'f': 0.05, 'g': 0.05, 
                   'h': 0.05, 'i': 0.1, 'j': 0.01, 'k': 0.01, 'l': 0.05, 'm': 0.05, 'n': 0.1, 
                   'o': 0.1, 'p': 0.05, 'q': 0.01, 'r': 0.05, 's': 0.05, 't': 0.1, 'u': 0.05, 
                   'v': 0.01, 'w': 0.05, 'x': 0.01, 'y': 0.05, 'z': 0.01}
    
    transition_probs = estimate_transition_probabilities(test_words)
    
    print(f"Estimated transition probabilities: {len(transition_probs)} transitions")
    
    # Generate words using these probabilities
    words = generate_first_order_markov_words(letter_probs, transition_probs, 10)
    print(f"Generated words: {words}")
    
    print_transition_statistics(transition_probs, words)

if __name__ == "__main__":
    test_first_order_markov() 