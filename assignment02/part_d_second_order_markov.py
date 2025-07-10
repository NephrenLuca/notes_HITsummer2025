"""
Part (d): Second Order Markov Chain
==================================

This module estimates second order transition probabilities P(x_{n+1} | x_n, x_{n-1})
and generates random 4-letter words using these probabilities.
"""

import numpy as np
from collections import defaultdict, Counter
import random

def estimate_second_order_transitions(words):
    """
    Estimate second order transition probabilities P(x_{n+1} | x_n, x_{n-1}).
    
    Args:
        words (list): List of words (strings of letters)
    
    Returns:
        dict: Dictionary mapping (prev_letter, current_letter, next_letter) to probability
    """
    # Count second order transitions within words only
    second_order_counts = defaultdict(Counter)
    
    for word in words:
        if len(word) < 3:
            continue
        
        # Count second order transitions within the word
        for i in range(len(word) - 2):
            prev_letter = word[i]
            current_letter = word[i + 1]
            next_letter = word[i + 2]
            second_order_counts[(prev_letter, current_letter)][next_letter] += 1
    
    # Calculate second order transition probabilities
    second_order_probs = {}
    for prev_letter in 'abcdefghijklmnopqrstuvwxyz':
        for current_letter in 'abcdefghijklmnopqrstuvwxyz':
            total_transitions = sum(second_order_counts[(prev_letter, current_letter)].values())
            
            if total_transitions == 0:
                # If no transitions for this pair, use uniform distribution
                for next_letter in 'abcdefghijklmnopqrstuvwxyz':
                    second_order_probs[(prev_letter, current_letter, next_letter)] = 1.0 / 26
            else:
                # Calculate conditional probabilities
                for next_letter in 'abcdefghijklmnopqrstuvwxyz':
                    count = second_order_counts[(prev_letter, current_letter)][next_letter]
                    second_order_probs[(prev_letter, current_letter, next_letter)] = count / total_transitions
    
    return second_order_probs

def generate_second_order_markov_words(letter_probs, transition_probs, second_order_probs, num_words=100):
    """
    Generate random 4-letter words using second order Markov chain.
    
    Args:
        letter_probs (dict): Initial letter probabilities
        transition_probs (dict): First order transition probabilities
        second_order_probs (dict): Second order transition probabilities
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
        
        # Generate second letter using first order transition probabilities
        current_letter = word[-1]
        next_letter_probs = []
        for next_letter in letters:
            prob = transition_probs.get((current_letter, next_letter), 0)
            next_letter_probs.append(prob)
        
        # Normalize probabilities
        total_prob = sum(next_letter_probs)
        if total_prob > 0:
            next_letter_probs = [p / total_prob for p in next_letter_probs]
        else:
            next_letter_probs = [1.0 / 26] * 26
        
        second_letter = np.random.choice(letters, p=next_letter_probs)
        word += second_letter
        
        # Generate remaining letters using second order transition probabilities
        for _ in range(2):
            prev_letter = word[-2]
            current_letter = word[-1]
            
            # Get second order transition probabilities
            next_letter_probs = []
            for next_letter in letters:
                prob = second_order_probs.get((prev_letter, current_letter, next_letter), 0)
                next_letter_probs.append(prob)
            
            # Normalize probabilities
            total_prob = sum(next_letter_probs)
            if total_prob > 0:
                next_letter_probs = [p / total_prob for p in next_letter_probs]
            else:
                # Fallback to first order transition probabilities
                next_letter_probs = []
                for next_letter in letters:
                    prob = transition_probs.get((current_letter, next_letter), 0)
                    next_letter_probs.append(prob)
                
                total_prob = sum(next_letter_probs)
                if total_prob > 0:
                    next_letter_probs = [p / total_prob for p in next_letter_probs]
                else:
                    next_letter_probs = [1.0 / 26] * 26
            
            # Generate next letter
            next_letter = np.random.choice(letters, p=next_letter_probs)
            word += next_letter
        
        words.append(word)
    
    return words

def print_second_order_statistics(second_order_probs, words):
    """
    Print statistics about second order transition probabilities and generated words.
    
    Args:
        second_order_probs (dict): Second order transition probabilities
        words (list): Generated words
    """
    print("\nSecond Order Markov Chain Statistics:")
    print("-" * 45)
    
    # Show some example second order transition probabilities
    print("Sample second order transition probabilities:")
    sample_pairs = [('th', 'e'), ('an', 'd'), ('in', 'g'), ('er', 'e'), ('st', 'e')]
    
    for prev_curr in sample_pairs:
        prev_letter, current_letter = prev_curr
        print(f"\nP(next_letter | '{prev_letter}', '{current_letter}'):")
        transitions = [(next_letter, prob) for (prev, curr, next_letter), prob in second_order_probs.items() 
                      if prev == prev_letter and curr == current_letter and prob > 0.01]
        transitions.sort(key=lambda x: x[1], reverse=True)
        for next_letter, prob in transitions[:5]:
            print(f"  '{prev_letter}{current_letter}' -> '{next_letter}': {prob:.4f}")
    
    # Analyze generated words
    print(f"\nGenerated word statistics:")
    print(f"  Total words: {len(words)}")
    
    # Count letter triplets in generated words
    triplet_counts = Counter()
    for word in words:
        for i in range(len(word) - 2):
            triplet = (word[i], word[i + 1], word[i + 2])
            triplet_counts[triplet] += 1
    
    print(f"  Total letter triplets: {sum(triplet_counts.values())}")
    
    # Show most common letter triplets
    print("\nMost common letter triplets in generated words:")
    for triplet, count in triplet_counts.most_common(10):
        print(f"  '{triplet[0]}{triplet[1]}{triplet[2]}': {count}")

def test_second_order_markov():
    """
    Test function to verify second order Markov chain implementation.
    """
    print("Testing second order Markov chain...")
    
    # Test with sample words
    test_words = ['hello', 'world', 'python', 'markov', 'chain', 'text', 'generation', 'probability', 'statistics']
    letter_probs = {'a': 0.1, 'b': 0.05, 'c': 0.05, 'd': 0.05, 'e': 0.15, 'f': 0.05, 'g': 0.05, 
                   'h': 0.05, 'i': 0.1, 'j': 0.01, 'k': 0.01, 'l': 0.05, 'm': 0.05, 'n': 0.1, 
                   'o': 0.1, 'p': 0.05, 'q': 0.01, 'r': 0.05, 's': 0.05, 't': 0.1, 'u': 0.05, 
                   'v': 0.01, 'w': 0.05, 'x': 0.01, 'y': 0.05, 'z': 0.01}
    
    # Estimate first order transitions for fallback
    from part_c_first_order_markov import estimate_transition_probabilities
    transition_probs = estimate_transition_probabilities(test_words)
    
    second_order_probs = estimate_second_order_transitions(test_words)
    
    print(f"Estimated second order transition probabilities: {len(second_order_probs)} transitions")
    
    # Generate words using these probabilities
    words = generate_second_order_markov_words(letter_probs, transition_probs, second_order_probs, 10)
    print(f"Generated words: {words}")
    
    print_second_order_statistics(second_order_probs, words)

if __name__ == "__main__":
    test_second_order_markov() 