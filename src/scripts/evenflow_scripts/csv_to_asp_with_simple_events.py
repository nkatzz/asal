import pandas as pd
import numpy as np
from sax import SAXTransformer
from sklearn.model_selection import train_test_split

"""
Same as csv_to_asp.py, but uses only positions and simple events.
"""

# Code for extracting the right columns
file_path = '../data/DemoDataset_1Robot.csv'
df = pd.read_csv(file_path)

# Extract the specified columns
selected_columns = ['px', 'py', 'pz', 'idle', 'linear', 'rotational', 'goal_status']
new_df = df[selected_columns].copy()

# Replace TRUE with 1 and FALSE with 0 in the 'idle', 'linear', and 'rotational' columns
bool_columns = ['idle', 'linear', 'rotational']
new_df[bool_columns] = new_df[bool_columns].replace({'TRUE': 1, 'FALSE': 0})

# Convert boolean columns to integer type
new_df[bool_columns] = new_df[bool_columns].astype(int)

# Save the transformed DataFrame to a new CSV file
new_csv_file_path = '../data/transformed_dataset.csv'
new_df.to_csv(new_csv_file_path, index=False)

# Print the path to the new CSV file for downloading
print(new_csv_file_path)