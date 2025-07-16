import math
import pandas as pd
from collections import Counter

def entropy(probabilities):
    """Calculate entropy given a list of probabilities."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def calculate_entropy(data, target_column):
    """Calculate entropy of the target variable."""
    total_count = len(data)
    class_counts = Counter(data[target_column])
    
    probabilities = [count / total_count for count in class_counts.values()]
    return entropy(probabilities)

def calculate_information_gain(data, feature, target_column):
    """Calculate information gain for a given feature."""
    total_entropy = calculate_entropy(data, target_column)
    total_count = len(data)
    
    # Group by feature values
    feature_groups = data.groupby(feature)
    
    weighted_entropy = 0
    for feature_value, group in feature_groups:
        group_count = len(group)
        group_entropy = calculate_entropy(group, target_column)
        weighted_entropy += (group_count / total_count) * group_entropy
    
    return total_entropy - weighted_entropy

def id3_algorithm(data, features, target_column):
    """Build decision tree using ID3 algorithm."""
    # Base cases
    if len(data[target_column].unique()) == 1:
        return data[target_column].iloc[0]
    
    if len(features) == 0:
        return data[target_column].mode()[0]
    
    # Calculate information gain for all features
    information_gains = {}
    for feature in features:
        information_gains[feature] = calculate_information_gain(data, feature, target_column)
    
    # Select best feature
    best_feature = max(information_gains, key=information_gains.get)
    
    # Create tree
    tree = {best_feature: {}}
    
    # Remove best feature from features list
    remaining_features = [f for f in features if f != best_feature]
    
    # Create branches for each value of best feature
    for feature_value in data[best_feature].unique():
        subset = data[data[best_feature] == feature_value]
        tree[best_feature][feature_value] = id3_algorithm(subset, remaining_features, target_column)
    
    return tree

# Sample dataset (Play Tennis dataset)
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 
                'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                   'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("=== ID3 Algorithm Verification ===\n")

# Calculate initial entropy
initial_entropy = calculate_entropy(df, 'Play')
print(f"Initial Entropy: {initial_entropy:.3f}")

# Calculate information gain for each feature
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
print("\nInformation Gains:")
for feature in features:
    ig = calculate_information_gain(df, feature, 'Play')
    print(f"{feature}: {ig:.3f}")

# Build decision tree
print("\n=== Building Decision Tree ===")
tree = id3_algorithm(df, features, 'Play')

print("\nFinal Decision Tree Structure:")
print(tree)

# Verify calculations
print("\n=== Verification of Calculations ===")

# Outlook calculations
outlook_groups = df.groupby('Outlook')
print("\nOutlook breakdown:")
for outlook, group in outlook_groups:
    yes_count = len(group[group['Play'] == 'Yes'])
    no_count = len(group[group['Play'] == 'No'])
    total = len(group)
    print(f"{outlook}: {yes_count} Yes, {no_count} No (Total: {total})")

# Humidity calculations for Sunny branch
sunny_data = df[df['Outlook'] == 'Sunny']
print(f"\nSunny branch - Total instances: {len(sunny_data)}")
humidity_groups = sunny_data.groupby('Humidity')
for humidity, group in humidity_groups:
    yes_count = len(group[group['Play'] == 'Yes'])
    no_count = len(group[group['Play'] == 'No'])
    print(f"  Humidity {humidity}: {yes_count} Yes, {no_count} No")

# Wind calculations for Rainy branch
rainy_data = df[df['Outlook'] == 'Rainy']
print(f"\nRainy branch - Total instances: {len(rainy_data)}")
wind_groups = rainy_data.groupby('Wind')
for wind, group in wind_groups:
    yes_count = len(group[group['Play'] == 'Yes'])
    no_count = len(group[group['Play'] == 'No'])
    print(f"  Wind {wind}: {yes_count} Yes, {no_count} No") 