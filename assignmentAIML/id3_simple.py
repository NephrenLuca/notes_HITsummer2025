import math
from collections import Counter

def entropy(probabilities):
    """Calculate entropy given a list of probabilities."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def calculate_entropy(data, target_column):
    """Calculate entropy of the target variable."""
    total_count = len(data)
    class_counts = Counter([row[target_column] for row in data])
    
    probabilities = [count / total_count for count in class_counts.values()]
    return entropy(probabilities)

def calculate_information_gain(data, feature, target_column):
    """Calculate information gain for a given feature."""
    total_entropy = calculate_entropy(data, target_column)
    total_count = len(data)
    
    # Group by feature values
    feature_values = {}
    for row in data:
        feature_val = row[feature]
        if feature_val not in feature_values:
            feature_values[feature_val] = []
        feature_values[feature_val].append(row)
    
    weighted_entropy = 0
    for feature_value, group in feature_values.items():
        group_count = len(group)
        group_entropy = calculate_entropy(group, target_column)
        weighted_entropy += (group_count / total_count) * group_entropy
    
    return total_entropy - weighted_entropy

# Sample dataset (Play Tennis dataset)
data = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'Play': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'Play': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'Play': 'No'}
]

print("=== ID3 Algorithm Verification ===\n")

# Calculate initial entropy
initial_entropy = calculate_entropy(data, 'Play')
print(f"Initial Entropy: {initial_entropy:.3f}")

# Calculate information gain for each feature
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
print("\nInformation Gains:")
for feature in features:
    ig = calculate_information_gain(data, feature, 'Play')
    print(f"{feature}: {ig:.3f}")

# Verify calculations
print("\n=== Verification of Calculations ===")

# Outlook calculations
outlook_groups = {}
for row in data:
    outlook = row['Outlook']
    if outlook not in outlook_groups:
        outlook_groups[outlook] = []
    outlook_groups[outlook].append(row)

print("\nOutlook breakdown:")
for outlook, group in outlook_groups.items():
    yes_count = len([row for row in group if row['Play'] == 'Yes'])
    no_count = len([row for row in group if row['Play'] == 'No'])
    total = len(group)
    print(f"{outlook}: {yes_count} Yes, {no_count} No (Total: {total})")

# Humidity calculations for Sunny branch
sunny_data = [row for row in data if row['Outlook'] == 'Sunny']
print(f"\nSunny branch - Total instances: {len(sunny_data)}")
humidity_groups = {}
for row in sunny_data:
    humidity = row['Humidity']
    if humidity not in humidity_groups:
        humidity_groups[humidity] = []
    humidity_groups[humidity].append(row)

for humidity, group in humidity_groups.items():
    yes_count = len([row for row in group if row['Play'] == 'Yes'])
    no_count = len([row for row in group if row['Play'] == 'No'])
    print(f"  Humidity {humidity}: {yes_count} Yes, {no_count} No")

# Wind calculations for Rainy branch
rainy_data = [row for row in data if row['Outlook'] == 'Rainy']
print(f"\nRainy branch - Total instances: {len(rainy_data)}")
wind_groups = {}
for row in rainy_data:
    wind = row['Wind']
    if wind not in wind_groups:
        wind_groups[wind] = []
    wind_groups[wind].append(row)

for wind, group in wind_groups.items():
    yes_count = len([row for row in group if row['Play'] == 'Yes'])
    no_count = len([row for row in group if row['Play'] == 'No'])
    print(f"  Wind {wind}: {yes_count} Yes, {no_count} No")

print("\n=== Manual Entropy Calculations ===")

# Calculate entropy for each group manually
print("\nEntropy calculations:")

# Overall entropy
total_yes = len([row for row in data if row['Play'] == 'Yes'])
total_no = len([row for row in data if row['Play'] == 'No'])
total = len(data)
p_yes = total_yes / total
p_no = total_no / total
overall_entropy = -p_yes * math.log2(p_yes) - p_no * math.log2(p_no)
print(f"Overall entropy: -{p_yes:.3f} \\times \\log_2({p_yes:.3f}) - {p_no:.3f} \\times \\log_2({p_no:.3f}) = {overall_entropy:.3f}")

# Sunny entropy
sunny_yes = len([row for row in sunny_data if row['Play'] == 'Yes'])
sunny_no = len([row for row in sunny_data if row['Play'] == 'No'])
sunny_total = len(sunny_data)
if sunny_total > 0:
    p_sunny_yes = sunny_yes / sunny_total
    p_sunny_no = sunny_no / sunny_total
    sunny_entropy = -p_sunny_yes * math.log2(p_sunny_yes) - p_sunny_no * math.log2(p_sunny_no)
    print(f"Sunny entropy: -{p_sunny_yes:.3f} \\times \\log_2({p_sunny_yes:.3f}) - {p_sunny_no:.3f} \\times \\log_2({p_sunny_no:.3f}) = {sunny_entropy:.3f}")

# Overcast entropy (should be 0)
overcast_data = [row for row in data if row['Outlook'] == 'Overcast']
overcast_yes = len([row for row in overcast_data if row['Play'] == 'Yes'])
overcast_no = len([row for row in overcast_data if row['Play'] == 'No'])
overcast_total = len(overcast_data)
if overcast_total > 0:
    p_overcast_yes = overcast_yes / overcast_total
    p_overcast_no = overcast_no / overcast_total
    if p_overcast_no == 0:
        overcast_entropy = 0
    else:
        overcast_entropy = -p_overcast_yes * math.log2(p_overcast_yes) - p_overcast_no * math.log2(p_overcast_no)
    print(f"Overcast entropy: {overcast_entropy:.3f} (all same class)")

# Rainy entropy
rainy_yes = len([row for row in rainy_data if row['Play'] == 'Yes'])
rainy_no = len([row for row in rainy_data if row['Play'] == 'No'])
rainy_total = len(rainy_data)
if rainy_total > 0:
    p_rainy_yes = rainy_yes / rainy_total
    p_rainy_no = rainy_no / rainy_total
    rainy_entropy = -p_rainy_yes * math.log2(p_rainy_yes) - p_rainy_no * math.log2(p_rainy_no)
    print(f"Rainy entropy: -{p_rainy_yes:.3f} \\times \\log_2({p_rainy_yes:.3f}) - {p_rainy_no:.3f} \\times \\log_2({p_rainy_no:.3f}) = {rainy_entropy:.3f}")

# Information Gain calculation
weighted_entropy = (sunny_total/total) * sunny_entropy + (overcast_total/total) * overcast_entropy + (rainy_total/total) * rainy_entropy
information_gain = overall_entropy - weighted_entropy
print(f"\nInformation Gain for Outlook:")
print(f"IG = {overall_entropy:.3f} - (\\frac{{{sunny_total}}}{{{total}}} \\times {sunny_entropy:.3f} + \\frac{{{overcast_total}}}{{{total}}} \\times {overcast_entropy:.3f} + \\frac{{{rainy_total}}}{{{total}}} \\times {rainy_entropy:.3f})")
print(f"IG = {overall_entropy:.3f} - {weighted_entropy:.3f} = {information_gain:.3f}") 