import pandas as pd
from sklearn.model_selection import train_test_split
import csv


# Step 1: Generate 'pathway_names' dictionary
def generate_pathway_names_dict(df):
    pathway_names = {}
    for column in df.columns[4:]:  # Assuming first four columns are not pathways
        transformed_name = column.lower().replace('-', '').replace(' ', '_')
        pathway_names[column] = transformed_name
    return pathway_names


# Step 2: Create methods to extract variable and non-variable pathways
def find_variable_pathways(df, pathway_names):
    variable_pathways = set()
    for column in df.columns[4:]:
        if len(df[column].unique()) > 1:
            variable_pathways.add(pathway_names[column])
    return variable_pathways


def find_non_variable_pathways(df, pathway_names):
    non_variable_pathways = set()
    for column in df.columns[4:]:
        if len(df[column].unique()) == 1:
            non_variable_pathways.add(pathway_names[column])
    return non_variable_pathways


# Load the datasets
ductal_df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/BSC/pathway_enrichment/ductal.csv')
lobular_df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/BSC/pathway_enrichment/lobular.csv')

# Generate pathway names dictionaries for both datasets
ductal_pathway_names = generate_pathway_names_dict(ductal_df)
lobular_pathway_names = generate_pathway_names_dict(lobular_df)

# Generating reverse dictionaries for both datasets to map transformed pathway names back to original names
reverse_pathway_names_ductal = {v: k for k, v in ductal_pathway_names.items()}
reverse_pathway_names_lobular = {v: k for k, v in lobular_pathway_names.items()}

# Extract variable and non-variable pathways for both datasets
ductal_variable_pathways = find_variable_pathways(ductal_df, ductal_pathway_names)
lobular_variable_pathways = find_variable_pathways(lobular_df, lobular_pathway_names)

ductal_non_variable_pathways = find_non_variable_pathways(ductal_df, ductal_pathway_names)
lobular_non_variable_pathways = find_non_variable_pathways(lobular_df, lobular_pathway_names)

print(ductal_pathway_names, lobular_pathway_names, ductal_variable_pathways, lobular_variable_pathways,
      ductal_non_variable_pathways, lobular_non_variable_pathways)


# Function to create dataframe with logical sequences
def create_logical_sequences_df(df, variable_pathways, reverse_pathway_names):
    """Create a dataframe of logical sequences."""
    logical_sequences = []
    current_example = [[] for _ in variable_pathways]
    seq_id = 0

    for _, row in df.iterrows():
        time_point = row["INTERPOL_POINT"]
        class_label = 0 if row["STAGE_TRANSITION"] == "I_to_II" else 1

        zipped = zip(current_example, variable_pathways)

        # For each variable pathway, create a logical atom
        for current_subseq, pathway in zipped:
            actual_pathway_name = reverse_pathway_names[pathway]
            enriched_state = 'enriched' if row[actual_pathway_name] == 1 else 'not_enriched'
            current_subseq.append(f"seq({seq_id},{enriched_state}({pathway}),{time_point}).")

        # If the time point is the last in the sequence, add the class label and reset for the next sequence
        if time_point == 49:
            for s in current_example:
                s.append(f"class({seq_id},{class_label}).")
            current_example = '\n'.join([' '.join(s) for s in current_example])
            logical_sequences.append((current_example, class_label))
            current_example = [[] for _ in variable_pathways]
            seq_id += 1

    return pd.DataFrame(logical_sequences, columns=['example', 'label'])


# Just a test...
all_ductal_pathways = ductal_variable_pathways.union(ductal_non_variable_pathways)

# Create dataframes for ductal and lobular datasets
logical_seqs_df_ductal = create_logical_sequences_df(ductal_df, all_ductal_pathways, reverse_pathway_names_ductal)
logical_seqs_df_lobular = create_logical_sequences_df(lobular_df, lobular_variable_pathways,
                                                      reverse_pathway_names_lobular)

# Splitting the data into train/test sets
train_ductal, test_ductal = train_test_split(logical_seqs_df_ductal, test_size=0.2, random_state=42,
                                             stratify=logical_seqs_df_ductal['label'])
train_lobular, test_lobular = train_test_split(logical_seqs_df_lobular, test_size=0.2, random_state=42,
                                               stratify=logical_seqs_df_lobular['label'])


def write_to_file(df, file_path):
    column_data = df['example'].astype(str)
    formatted_data = '\n'.join(column_data)
    with open(file_path, 'w') as file:
        file.write(formatted_data)


write_to_file(train_ductal, '../../../data/bsc_ductal/folds/fold_0/train.csv')
write_to_file(test_ductal, '../../../data/bsc_ductal/folds/fold_0/test.csv')
write_to_file(train_lobular, '../../../data/bsc_lobular/folds/fold_0/train.csv')
write_to_file(test_lobular, '../../../data/bsc_lobular/folds/fold_0/test.csv')
