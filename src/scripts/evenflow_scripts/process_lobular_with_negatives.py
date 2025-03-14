import pandas as pd
from sklearn.model_selection import train_test_split


def generate_pathway_names_dict(df):
    pathway_names = {}
    for column in df.columns[4:33]:  # skip the first 4, keep the following 29 (the pathways), skip the rest
        transformed_name = column.lower().replace('-', '').replace(' ', '_')
        pathway_names[column] = transformed_name
    return pathway_names


def find_variable_pathways(df, pathway_names):
    variable_pathways = set()
    for column in df.columns[4:33]:
        if len(df[column].unique()) > 1:
            variable_pathways.add(pathway_names[column])
    return variable_pathways


def find_non_variable_pathways(df, pathway_names):
    non_variable_pathways = set()
    for column in df.columns[4:33]:
        if len(df[column].unique()) == 1:
            non_variable_pathways.add(pathway_names[column])
    return non_variable_pathways


def create_logical_sequences_df(df, variable_pathways, reverse_pathway_names):
    """Create a dataframe of logical sequences."""
    logical_sequences = []
    current_example = [[] for _ in variable_pathways]
    seq_id = 0

    for _, row in df.iterrows():
        time_point = row["INTERPOL_POINT"]
        class_label = 0 if row["STAGE_TRANSITION"] == "I_to_I" else 1

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


def get_positives(data_path):
    df = pd.read_csv(data_path)
    df_pos = df[df["STAGE_TRANSITION"] == 'I_to_II']
    return df_pos


def merge_pos_neg(df_pos, df_neg):
    df_merged = pd.concat([df_pos, df_neg], ignore_index=True)
    # Reset the index in the merged df:
    # df_merged.reset_index(inplace=True, drop=False)
    # df_merged.rename(columns={'index': 'id'}, inplace=True)
    return df_merged


def write_to_file(df, file_path):
    column_data = df['example'].astype(str)
    formatted_data = '\n'.join(column_data)
    with open(file_path, 'w') as file:
        file.write(formatted_data)


negatives = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/BSC/lobular_i_negative_examples_w_rnaseq.csv')
positives = get_positives('/media/nkatz/storage/EVENFLOW-DATA/BSC/pathway_enrichment/lobular.csv')

lobular_df = merge_pos_neg(positives, negatives)

# Generate pathway names dictionary
lobular_pathway_names = generate_pathway_names_dict(lobular_df)

# Generating reverse dictionary
reverse_pathway_names_lobular = {v: k for k, v in lobular_pathway_names.items()}

# Extract variable and non-variable pathways
lobular_variable_pathways = find_variable_pathways(lobular_df, lobular_pathway_names)
lobular_non_variable_pathways = find_non_variable_pathways(lobular_df, lobular_pathway_names)

print(lobular_pathway_names, lobular_variable_pathways, lobular_non_variable_pathways)

all_lobular_pathways = lobular_variable_pathways.union(lobular_non_variable_pathways)

# Create dataframe for the lobular dataset
logical_seqs_df_lobular = create_logical_sequences_df(lobular_df, all_lobular_pathways, reverse_pathway_names_lobular)

# Train/test split
train_lobular, test_lobular = train_test_split(logical_seqs_df_lobular, test_size=0.2, random_state=42,
                                               stratify=logical_seqs_df_lobular['label'])

write_to_file(train_lobular, '../../../data/bsc_lobular/folds/fold_0/train.csv')
write_to_file(test_lobular, '../../../data/bsc_lobular/folds/fold_0/test.csv')
