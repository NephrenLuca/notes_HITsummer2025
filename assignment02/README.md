# Markov Chain Text Generation

This project implements a complete solution for the Markov chain text generation problem using the files "spamiam.txt" and "saki_story.txt".

## Problem Description

The project addresses the following parts of the Markov chain text generation problem:

- **Part (a)**: Generate random 4-letter words with equiprobable letters
- **Part (b)**: Estimate letter probabilities and generate words using these probabilities
- **Part (c)**: First order Markov chain with transition probabilities P(x_{n+1} | x_n)
- **Part (d)**: Second order Markov chain with transition probabilities P(x_{n+1} | x_n, x_{n-1})
- **Part (e)**: Repeat parts b-d using different text file
- **Part (f)**: Analysis and comments on results
- **Part (g)**: Entropy rate estimation (Extra Credit)

## Files Structure

```
├── markov_text_generation.py    # Main solution file
├── part_a_equiprobable.py       # Part (a) implementation
├── part_b_letter_probabilities.py # Part (b) implementation
├── part_c_first_order_markov.py # Part (c) implementation
├── part_d_second_order_markov.py # Part (d) implementation
├── part_g_entropy_estimation.py # Part (g) implementation
├── results_and_analysis.md      # Complete results and analysis
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── spamiam.txt                 # Input text file
└── saki_story.txt             # Input text file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Solution

To run the complete solution with all parts:

```bash
python markov_text_generation.py
```

This will:
- Preprocess both text files
- Generate words for all parts (a-e)
- Display results in 10×10 grids
- Circle valid English words
- Provide analysis and entropy rate estimation

### Run Individual Parts

You can also run individual parts separately:

```bash
# Part (a): Equiprobable letters
python part_a_equiprobable.py

# Part (b): Letter probabilities
python part_b_letter_probabilities.py

# Part (c): First order Markov chain
python part_c_first_order_markov.py

# Part (d): Second order Markov chain
python part_d_second_order_markov.py

# Part (g): Entropy estimation
python part_g_entropy_estimation.py
```

## Key Features

### Text Preprocessing
- Converts all text to lowercase
- Extracts only alphabetic characters (a-z)
- Splits into words (sequences of letters)
- Ignores case distinctions and non-alphabetic characters

### Word Generation
- Generates 100 random 4-letter words for each part
- Displays results in 10×10 grid format
- Circles valid English words
- Provides statistics and analysis

### Markov Chain Models
- **Zero-order**: Independent letter selection
- **First-order**: P(x_{n+1} | x_n) - considers current letter
- **Second-order**: P(x_{n+1} | x_n, x_{n-1}) - considers previous two letters

### Entropy Rate Estimation
- Calculates entropy rates for all models
- Compares across different text sources
- Provides theoretical analysis

## Results

The complete results are available in `results_and_analysis.md`, including:

- Generated words for all parts in 10×10 grid format
- Valid English word counts
- Letter probability distributions
- Transition probability examples
- Entropy rate calculations
- Comprehensive analysis and comparison

## Analysis Highlights

1. **Model Improvement**: Higher order models produce more realistic text
2. **Text Source Impact**: Saki Story produces more diverse patterns than Spamiam
3. **Valid Word Generation**: Higher order models generate more valid English words
4. **Entropy Reduction**: Higher order models capture more linguistic structure

## Dependencies

- **numpy**: For numerical operations and random sampling
- **matplotlib**: For plotting (if needed for visualization)
- **collections**: For Counter and defaultdict data structures
- **re**: For regular expression text processing
- **math**: For mathematical operations and entropy calculations

## Notes

- The solution handles edge cases where transition probabilities might be zero
- Fallback mechanisms ensure robust word generation
- All probabilities are properly normalized
- The implementation is modular and well-documented

## Expected Output

When you run the main script, you'll see:
1. Preprocessing information
2. Generated words for each part in grid format
3. Valid English word counts
4. Statistical analysis
5. Entropy rate calculations
6. Comparative analysis between models and text sources

The results demonstrate the effectiveness of Markov chain models for text generation and show clear improvements as model complexity increases. 