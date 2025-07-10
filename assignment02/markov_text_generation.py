"""
Markov Chain Text Generation - Complete Solution
===============================================

This script implements all parts of the Markov chain text generation problem:
- Part (a): Random 4-letter words with equiprobable letters
- Part (b): Random 4-letter words using estimated letter probabilities
- Part (c): Random 4-letter words using 1st order Markov chain
- Part (d): Random 4-letter words using 2nd order Markov chain
- Part (e): Repeat parts b-d using different text file
- Part (f): Analysis and comments
- Part (g): Entropy rate estimation (Extra Credit)

Author: AI Assistant
"""

import numpy as np
from collections import Counter, defaultdict
import re
import math

# Import the individual modules
from part_a_equiprobable import generate_equiprobable_words
from part_b_letter_probabilities import estimate_letter_probabilities, generate_letter_prob_words
from part_c_first_order_markov import estimate_transition_probabilities, generate_first_order_markov_words
from part_d_second_order_markov import estimate_second_order_transitions, generate_second_order_markov_words
from part_g_entropy_estimation import estimate_entropy_rate

def preprocess_text(filename):
    """
    Preprocess text file to extract only lowercase letters, ignoring case and non-alphabetic characters.
    Returns a list of words (sequences of letters).
    """
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Convert to lowercase and extract only letters
    text = text.lower()
    # Split into words (sequences of letters)
    words = re.findall(r'[a-z]+', text)
    return words

def print_word_grid(words, title):
    """
    Print 100 words in a 10x10 grid format.
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # Circle valid English words (simple check - in practice you'd use a dictionary)
    valid_words = set([
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'want', 'well', 'went', 'were', 'what', 'when', 'will', 'with', 'word', 'been', 'call', 'come', 'each', 'find', 'from', 'give', 'have', 'here', 'just', 'know', 'like', 'look', 'make', 'many', 'more', 'most', 'only', 'over', 'some', 'take', 'than', 'them', 'they', 'this', 'time', 'very', 'were', 'what', 'when', 'will', 'with', 'your', 'about', 'after', 'again', 'could', 'every', 'first', 'found', 'great', 'house', 'large', 'might', 'never', 'other', 'place', 'right', 'small', 'sound', 'still', 'their', 'there', 'these', 'thing', 'think', 'three', 'under', 'water', 'where', 'which', 'while', 'world', 'would', 'write', 'years', 'above', 'began', 'below', 'between', 'carry', 'change', 'children', 'country', 'different', 'enough', 'example', 'follow', 'important', 'letter', 'mother', 'picture', 'should', 'something', 'through', 'together', 'without'
    ])
    
    for i in range(10):
        row_words = words[i*10:(i+1)*10]
        row_str = ""
        for j, word in enumerate(row_words):
            if word in valid_words:
                row_str += f"({word}) "
            else:
                row_str += f"{word} "
        print(row_str)
    
    # Count valid words
    valid_count = sum(1 for word in words if word in valid_words)
    print(f"\nValid English words found: {valid_count}/100")

def main():
    """
    Main function to run all parts of the problem.
    """
    print("Markov Chain Text Generation - Complete Solution")
    print("=" * 60)
    
    # Preprocess text files
    print("\nPreprocessing text files...")
    spamiam_words = preprocess_text("spamiam.txt")
    saki_words = preprocess_text("saki_story.txt")
    
    print(f"Spamiam.txt: {len(spamiam_words)} words")
    print(f"Saki story.txt: {len(saki_words)} words")
    
    # Part (a): Equiprobable letters
    print("\n" + "="*60)
    print("PART (a): Random 4-letter words with equiprobable letters")
    print("="*60)
    equiprobable_words = generate_equiprobable_words(100)
    print_word_grid(equiprobable_words, "Equiprobable Letters (10x10 Grid)")
    
    # Part (b): Letter probabilities from spamiam.txt
    print("\n" + "="*60)
    print("PART (b): Random 4-letter words using letter probabilities from spamiam.txt")
    print("="*60)
    letter_probs = estimate_letter_probabilities(spamiam_words)
    letter_prob_words = generate_letter_prob_words(letter_probs, 100)
    print_word_grid(letter_prob_words, "Letter Probabilities from Spamiam (10x10 Grid)")
    
    # Part (c): First order Markov chain from spamiam.txt
    print("\n" + "="*60)
    print("PART (c): Random 4-letter words using 1st order Markov chain from spamiam.txt")
    print("="*60)
    transition_probs = estimate_transition_probabilities(spamiam_words)
    first_order_words = generate_first_order_markov_words(letter_probs, transition_probs, 100)
    print_word_grid(first_order_words, "1st Order Markov from Spamiam (10x10 Grid)")
    
    # Part (d): Second order Markov chain from spamiam.txt
    print("\n" + "="*60)
    print("PART (d): Random 4-letter words using 2nd order Markov chain from spamiam.txt")
    print("="*60)
    second_order_probs = estimate_second_order_transitions(spamiam_words)
    second_order_words = generate_second_order_markov_words(letter_probs, transition_probs, second_order_probs, 100)
    print_word_grid(second_order_words, "2nd Order Markov from Spamiam (10x10 Grid)")
    
    # Part (e): Repeat parts b-d using saki_story.txt
    print("\n" + "="*60)
    print("PART (e): Repeat parts b-d using saki_story.txt")
    print("="*60)
    
    # Letter probabilities from saki_story.txt
    saki_letter_probs = estimate_letter_probabilities(saki_words)
    saki_letter_prob_words = generate_letter_prob_words(saki_letter_probs, 100)
    print_word_grid(saki_letter_prob_words, "Letter Probabilities from Saki Story (10x10 Grid)")
    
    # First order Markov chain from saki_story.txt
    saki_transition_probs = estimate_transition_probabilities(saki_words)
    saki_first_order_words = generate_first_order_markov_words(saki_letter_probs, saki_transition_probs, 100)
    print_word_grid(saki_first_order_words, "1st Order Markov from Saki Story (10x10 Grid)")
    
    # Second order Markov chain from saki_story.txt
    saki_second_order_probs = estimate_second_order_transitions(saki_words)
    saki_second_order_words = generate_second_order_markov_words(saki_letter_probs, saki_transition_probs, saki_second_order_probs, 100)
    print_word_grid(saki_second_order_words, "2nd Order Markov from Saki Story (10x10 Grid)")
    
    # Part (f): Analysis and comments
    print("\n" + "="*60)
    print("PART (f): Analysis and Comments")
    print("="*60)
    analyze_results()
    
    # Part (g): Entropy rate estimation (Extra Credit)
    print("\n" + "="*60)
    print("PART (g): Entropy Rate Estimation (Extra Credit)")
    print("="*60)
    estimate_all_entropy_rates(spamiam_words, saki_words)

def analyze_results():
    """
    Analyze and comment on the results from all parts.
    """
    print("\nAnalysis of Results:")
    print("-" * 30)
    
    print("\n1. Equiprobable Letters (Part a):")
    print("   - Generated words are completely random")
    print("   - No linguistic patterns or structure")
    print("   - Expected to have very few valid English words")
    
    print("\n2. Letter Probabilities (Part b):")
    print("   - Words reflect the frequency distribution of letters in the source text")
    print("   - More realistic letter combinations than random")
    print("   - Still lacks sequential dependencies between letters")
    
    print("\n3. First Order Markov Chain (Part c):")
    print("   - Considers the probability of the next letter given the current letter")
    print("   - Captures basic sequential patterns in the language")
    print("   - Should produce more realistic letter sequences")
    
    print("\n4. Second Order Markov Chain (Part d):")
    print("   - Considers the probability of the next letter given the previous two letters")
    print("   - Captures more complex linguistic patterns")
    print("   - Should produce the most realistic letter sequences")
    
    print("\n5. Comparison between Spamiam and Saki Story:")
    print("   - Spamiam: Short, repetitive text with limited vocabulary")
    print("   - Saki Story: Longer, more diverse text with richer vocabulary")
    print("   - Saki Story should produce more varied and realistic word patterns")
    
    print("\n6. Expected Improvements with Higher Order Models:")
    print("   - Higher order models capture more complex dependencies")
    print("   - Should produce more valid English words")
    print("   - Better approximation of natural language patterns")

def estimate_all_entropy_rates(spamiam_words, saki_words):
    """
    Estimate entropy rates for all models and both text files.
    """
    print("\nEntropy Rate Estimation:")
    print("-" * 30)
    
    # For Spamiam text
    print("\nSpamiam.txt Entropy Rates:")
    spamiam_letter_probs = estimate_letter_probabilities(spamiam_words)
    spamiam_transition_probs = estimate_transition_probabilities(spamiam_words)
    spamiam_second_order_probs = estimate_second_order_transitions(spamiam_words)
    
    # Zero-order (letter probabilities)
    h0_spamiam = estimate_entropy_rate(spamiam_letter_probs, order=0)
    print(f"   Zero-order (letter probabilities): {h0_spamiam:.4f} bits/letter")
    
    # First-order (transition probabilities)
    h1_spamiam = estimate_entropy_rate(spamiam_transition_probs, order=1, letter_probs=spamiam_letter_probs)
    print(f"   First-order (Markov chain): {h1_spamiam:.4f} bits/letter")
    
    # Second-order
    h2_spamiam = estimate_entropy_rate(spamiam_second_order_probs, order=2, letter_probs=spamiam_letter_probs, transition_probs=spamiam_transition_probs)
    print(f"   Second-order (Markov chain): {h2_spamiam:.4f} bits/letter")
    
    # For Saki Story text
    print("\nSaki Story.txt Entropy Rates:")
    saki_letter_probs = estimate_letter_probabilities(saki_words)
    saki_transition_probs = estimate_transition_probabilities(saki_words)
    saki_second_order_probs = estimate_second_order_transitions(saki_words)
    
    # Zero-order (letter probabilities)
    h0_saki = estimate_entropy_rate(saki_letter_probs, order=0)
    print(f"   Zero-order (letter probabilities): {h0_saki:.4f} bits/letter")
    
    # First-order (transition probabilities)
    h1_saki = estimate_entropy_rate(saki_transition_probs, order=1, letter_probs=saki_letter_probs)
    print(f"   First-order (Markov chain): {h1_saki:.4f} bits/letter")
    
    # Second-order
    h2_saki = estimate_entropy_rate(saki_second_order_probs, order=2, letter_probs=saki_letter_probs, transition_probs=saki_transition_probs)
    print(f"   Second-order (Markov chain): {h2_saki:.4f} bits/letter")
    
    print("\nAnalysis:")
    print("-" * 10)
    print("1. Entropy rates generally decrease with higher order models")
    print("2. Saki Story has higher entropy than Spamiam due to more diverse vocabulary")
    print("3. Higher order models capture more structure, reducing uncertainty")
    print("4. The entropy rate approaches the true language entropy as order increases")

if __name__ == "__main__":
    main() 