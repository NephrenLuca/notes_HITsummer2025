"""
Part (g): Entropy Rate Estimation (Extra Credit)
===============================================

This module estimates the entropy rate for different Markov models.
The entropy rate measures the average uncertainty per letter in the generated text.
"""

import numpy as np
import math

def estimate_entropy_rate(probabilities, order=0, letter_probs=None, transition_probs=None):
    """
    Estimate entropy rate for different order Markov models.
    
    Args:
        probabilities: For order=0: letter probabilities dict
                     For order=1: transition probabilities dict
                     For order=2: second order transition probabilities dict
        order (int): Order of the Markov model (0, 1, or 2)
        letter_probs (dict): Letter probabilities (required for order > 0)
        transition_probs (dict): First order transition probabilities (required for order=2)
    
    Returns:
        float: Entropy rate in bits per letter
    """
    if order == 0:
        # Zero-order entropy: H = -sum(p_i * log2(p_i))
        entropy = 0
        for letter, prob in probabilities.items():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    elif order == 1:
        # First-order entropy: H = -sum(p_i * sum(p_j|i * log2(p_j|i)))
        if letter_probs is None:
            raise ValueError("letter_probs required for first-order entropy")
        
        entropy = 0
        for current_letter in 'abcdefghijklmnopqrstuvwxyz':
            p_current = letter_probs.get(current_letter, 0)
            if p_current > 0:
                letter_entropy = 0
                for next_letter in 'abcdefghijklmnopqrstuvwxyz':
                    p_next_given_current = probabilities.get((current_letter, next_letter), 0)
                    if p_next_given_current > 0:
                        letter_entropy -= p_next_given_current * math.log2(p_next_given_current)
                entropy += p_current * letter_entropy
        return entropy
    
    elif order == 2:
        # Second-order entropy: H = -sum(p_ij * sum(p_k|ij * log2(p_k|ij)))
        if letter_probs is None or transition_probs is None:
            raise ValueError("letter_probs and transition_probs required for second-order entropy")
        
        entropy = 0
        for prev_letter in 'abcdefghijklmnopqrstuvwxyz':
            for current_letter in 'abcdefghijklmnopqrstuvwxyz':
                # Calculate joint probability p(prev, current)
                p_prev = letter_probs.get(prev_letter, 0)
                p_current_given_prev = transition_probs.get((prev_letter, current_letter), 0)
                p_joint = p_prev * p_current_given_prev
                
                if p_joint > 0:
                    letter_pair_entropy = 0
                    for next_letter in 'abcdefghijklmnopqrstuvwxyz':
                        p_next_given_pair = probabilities.get((prev_letter, current_letter, next_letter), 0)
                        if p_next_given_pair > 0:
                            letter_pair_entropy -= p_next_given_pair * math.log2(p_next_given_pair)
                    entropy += p_joint * letter_pair_entropy
        return entropy
    
    else:
        raise ValueError("Order must be 0, 1, or 2")

def calculate_theoretical_entropy():
    """
    Calculate theoretical entropy for equiprobable letters.
    
    Returns:
        float: Theoretical entropy in bits per letter
    """
    # For equiprobable letters, H = log2(26)
    return math.log2(26)

def compare_entropy_rates(spamiam_entropies, saki_entropies):
    """
    Compare entropy rates between different models and text sources.
    
    Args:
        spamiam_entropies (dict): Entropy rates for Spamiam text
        saki_entropies (dict): Entropy rates for Saki Story text
    """
    print("\nEntropy Rate Comparison:")
    print("-" * 30)
    
    theoretical = calculate_theoretical_entropy()
    print(f"Theoretical maximum (equiprobable): {theoretical:.4f} bits/letter")
    
    print(f"\nSpamiam.txt:")
    for order, entropy in spamiam_entropies.items():
        print(f"  Order {order}: {entropy:.4f} bits/letter")
    
    print(f"\nSaki Story.txt:")
    for order, entropy in saki_entropies.items():
        print(f"  Order {order}: {entropy:.4f} bits/letter")
    
    print("\nAnalysis:")
    print("-" * 10)
    
    # Compare across models
    print("1. Model Comparison:")
    for text_name, entropies in [("Spamiam", spamiam_entropies), ("Saki Story", saki_entropies)]:
        print(f"   {text_name}:")
        for order in [0, 1, 2]:
            if order in entropies:
                reduction = theoretical - entropies[order]
                print(f"     Order {order}: {reduction:.4f} bits reduction from theoretical")
    
    # Compare across texts
    print("\n2. Text Source Comparison:")
    for order in [0, 1, 2]:
        if order in spamiam_entropies and order in saki_entropies:
            diff = spamiam_entropies[order] - saki_entropies[order]
            print(f"   Order {order}: Spamiam has {diff:.4f} bits higher entropy than Saki Story")
    
    print("\n3. Interpretation:")
    print("   - Lower entropy indicates more predictable/structured text")
    print("   - Higher order models capture more structure, reducing entropy")
    print("   - Spamiam has lower entropy due to repetitive nature")
    print("   - Saki Story has higher entropy due to more diverse vocabulary")

def test_entropy_estimation():
    """
    Test function to verify entropy rate estimation.
    """
    print("Testing entropy rate estimation...")
    
    # Test zero-order entropy
    letter_probs = {'a': 0.1, 'b': 0.05, 'c': 0.05, 'd': 0.05, 'e': 0.15, 'f': 0.05, 'g': 0.05, 
                   'h': 0.05, 'i': 0.1, 'j': 0.01, 'k': 0.01, 'l': 0.05, 'm': 0.05, 'n': 0.1, 
                   'o': 0.1, 'p': 0.05, 'q': 0.01, 'r': 0.05, 's': 0.05, 't': 0.1, 'u': 0.05, 
                   'v': 0.01, 'w': 0.05, 'x': 0.01, 'y': 0.05, 'z': 0.01}
    
    h0 = estimate_entropy_rate(letter_probs, order=0)
    print(f"Zero-order entropy: {h0:.4f} bits/letter")
    
    # Test theoretical maximum
    theoretical = calculate_theoretical_entropy()
    print(f"Theoretical maximum: {theoretical:.4f} bits/letter")
    print(f"Reduction from theoretical: {theoretical - h0:.4f} bits")

if __name__ == "__main__":
    test_entropy_estimation()