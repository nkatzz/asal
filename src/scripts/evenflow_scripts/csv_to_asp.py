import pandas as pd
import numpy as np
from sax import SAXTransformer
from sklearn.model_selection import train_test_split

"""
This script can be used to process CSV data, check the example here. It discretizes the dataset, generates the ASP
format of the data, generates a train/test split and saves the training and testing data in separate CSVs. 
The latter contain the data in ASP format, read to be passed to Clingo. See dfki_load_data.py for passing the data to ASAL.
"""


def generate_train_test_split(df_examples, iid=False, test_size=0.3):
    if iid:
        train_df, test_df = train_test_split(df_examples, test_size=test_size, stratify=df_examples['label'], random_state=42)
        return train_df, test_df
    else:
        labels_of_interest = ['moving to Station1', 'moving to Station2', 'moving to Station3', 'moving to Station4',
                              'moving to Station5', 'moving to Station6', 'stopped (unknown)', 'stopped at Station1',
                              'stopped at Station2', 'stopped at Station3', 'stopped at Station4', 'stopped at Station5',
                              'stopped at Station6']
        df_filtered = df_examples[df_examples['label'].isin(labels_of_interest)]

        train_data = []
        test_data = []

        for label, group in df_filtered.groupby('label'):
            total = len(group)
            train_size = int((1 - test_size) * total)

            train_group = group.iloc[:train_size]
            test_group = group.iloc[train_size:]

            train_data.append(train_group)
            test_data.append(test_group)

        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)

        return train_df, test_df


data_path = '../../../data'

data = pd.read_csv(data_path + '/avg_robot/DemoDataset_1Robot.csv')
# data = pd.read_csv(data_path + '/avg_robot/DemoDataset_1Robot_Coords_and_SimpleEvents_Only.csv')
# data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_vae_ripper.csv')
# data = pd.read_csv(data_path + '/avg_robot/PCA_Discretized_Dataset.csv')

sax_alphabet_size = 10
k = 50  # Length of the sub-series
m = 10  # Window step size

# Generate the class mapping
class_mapping = {label: idx for idx, label in enumerate(data['goal_status'].unique())}
class_mapping_reverse = {v: k for k, v in class_mapping.items()}

# Assign integer labels to 'goal_status' column
data['goal_status'] = data['goal_status'].map(class_mapping)

# Drop the 'goal_status' column and keep it separately
labels = data['goal_status'].values
data = data.drop(columns='goal_status')

sax_transformer = SAXTransformer(sax_alphabet_size)

# Discretize the entire dataset with SAX
sax_transformed_data = []
for feature in data.columns:

    # If we are using the simple event columns (idle, rotating, linear), whose values are boolean (0, 1), then skip SAX.
    if not data[feature].isin([0, 1]).all():
        sax_transformed_data.append(sax_transformer.column_sax_transform(data[feature].values))
    else:
        sax_transformed_data.append(data[feature].values)

discretize = True  # set to false when working with already discretized data from PCA, see agv_time_series_PCA_whole_dataset_then_discretized.py
sax_transformed_data = np.array(sax_transformed_data).T if discretize else np.array(data)

# Split the discretized dataset into sub-sequences of length k
sax_sub_series_data = []
sax_sub_series_labels = []

for i in range(0, len(sax_transformed_data) - k + 1, m):
    sub_series = sax_transformed_data[i:i + k]
    label = labels[i + k - 1]
    sax_sub_series_data.append(sub_series)
    sax_sub_series_labels.append(label)

sax_sub_series_data = np.array(sax_sub_series_data)
sax_sub_series_labels = np.array(sax_sub_series_labels)

asp_data = []

for example_id, (example, label) in enumerate(zip(sax_sub_series_data, sax_sub_series_labels)):
    logic_example = []
    for feature_idx, feature in enumerate(example.T):
        feature_logic = [f"seq({example_id},{data.columns[feature_idx]}({value}),{time})." for time, value in
                         enumerate(feature)]
        logic_example.append(feature_logic)
    class_atom = f"class({example_id},{label})."
    for item in logic_example:
        item.append(class_atom)
    asp_data.append(logic_example)

df_examples = pd.DataFrame({
    'label': [class_mapping_reverse[label] for label in sax_sub_series_labels],
    'discretized_example': list(sax_sub_series_data),
    'logical_form': asp_data
})

# Determine how many examples each class has
class_counts = pd.Series(sax_sub_series_labels).value_counts()
print(f'Instances per class (entire dataset):\n{class_counts}')

# Splitting the DataFrame into training and testing sets with a 70/30 split ratio
# train_df, test_df = train_test_split(df_examples, test_size=0.3, stratify=df_examples['label'], random_state=42)

train_df, test_df = generate_train_test_split(df_examples)

# Count the number of instances per class in the training and testing sets
train_class_counts = train_df['label'].value_counts()
test_class_counts = test_df['label'].value_counts()

print(f"Training/testing sets sizes: {len(train_df), len(test_df)}")
print(f'instances per class (training set):\n{train_class_counts}')
print(f'instances per class (testing set):\n{test_class_counts}')

# Write the generated DataFrame and the training/testing sets to CSV files
# df_examples_file_path = "/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/df_examples.csv"

if 'idle' in data.columns or 'linear' in data.columns or 'rotational' in data.columns:
    train_df_file_path = data_path + "/avg_robot/train_df_with_simple_events.csv"
    test_df_file_path = data_path + "/avg_robot/test_df_with_simple_events.csv"
else:
    train_df_file_path = data_path + "/avg_robot/train_df.csv"
    test_df_file_path = data_path + "/avg_robot/test_df.csv"

print(f'Writing to CSV files')

# df_examples.to_csv(df_examples_file_path, index=False)
train_df.to_csv(train_df_file_path, index=False)
test_df.to_csv(test_df_file_path, index=False)
