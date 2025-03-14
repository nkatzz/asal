import pandas as pd
import ast
import math

"""
The methods here can be used to read the data from data frames when learning with ASAL, instead of having them 
stored in lists and fetching the data from there. This entails that the input data do not need to reside in files,
but they can be loaded from CSVs (see train.csv, test.csv). In particular, this script reads from train_df.csv into
a data frame and generates a new data frame consisting of the training mini-batches. The method that splits the data
into mini-batches follows the same logic as the 'get_train_data' method that is in use currently.
The method get_batch_data can be used to get a string representation of the data in a mini-batch from the mini-batch
data frame. This string can then be passed directly to Clingo.

However, replacing the current data fetching functionality with the above requires some (small) changes in the code
(TODO). Currently, we simply dump the ASP representation of the training and the testing data into files
to be used with the current data fetching tools.  

TODO: think how should we split things for cross-validation. Split per class, or split in a stratified manner
      for all classes?  
"""


def generate_mini_batches(dataframe, batch_size, class_label):
    # Separate the positive and negative examples for the current class
    positive_examples = dataframe[dataframe['label'] == class_label]
    negative_examples = dataframe[dataframe['label'] != class_label]

    # Compute the number of positive and negative examples
    pos_count = len(positive_examples)
    neg_count = len(negative_examples)

    # Calculate the number of mini-batches and the number of positive and negative examples per mini-batch
    mini_batch_count = (pos_count + neg_count) // batch_size if (pos_count + neg_count) // batch_size > 0 else 1
    pos_per_batch = int(math.ceil(pos_count / float(mini_batch_count)))
    neg_per_batch = int(math.ceil(neg_count / float(mini_batch_count)))

    mini_batches = []

    pos_idx, neg_idx = 0, 0  # Track indices for positive and negative examples

    for _ in range(mini_batch_count):
        # Extract a batch of examples without replacement
        if pos_idx + pos_per_batch <= pos_count:
            batch_pos = positive_examples.iloc[pos_idx:pos_idx + pos_per_batch]['logical_form'].tolist()
            pos_idx += pos_per_batch
        else:
            batch_pos = positive_examples.iloc[pos_idx:]['logical_form'].tolist()
            pos_idx = pos_count

        if neg_idx + neg_per_batch <= neg_count:
            batch_neg = negative_examples.iloc[neg_idx:neg_idx + neg_per_batch]['logical_form'].tolist()
            neg_idx += neg_per_batch
        else:
            batch_neg = negative_examples.iloc[neg_idx:]['logical_form'].tolist()
            neg_idx = neg_count

        combined_batch = batch_pos + batch_neg
        mini_batches.append(combined_batch)

    # Convert the mini_batches list into a DataFrame
    df_mini_batches = pd.DataFrame(mini_batches)

    return df_mini_batches


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return s


def convert_to_string(mini_batches_df: pd.DataFrame):
    batch_data_strings = []

    for _, row in mini_batches_df.iterrows():
        batch_data = [
            '\n'.join([' '.join(seq) for seq in cell]) for cell in row if cell is not None
        ]
        batch_data_strings.append('\n'.join(batch_data))

    return batch_data_strings


"""
for index, row in mini_batches_df.iterrows():
    batch_data = []
    for col in mini_batches_df.columns:
        example = []
        if row[col] is not None:
            for seq in row[col]:
                attribute_seq_to_str = ' '.join(seq)
                example.append(attribute_seq_to_str)
            example_to_str = '\n'.join(example)
            batch_data.append(example_to_str)
        batch_data_to_str = '\n'.join(batch_data)
"""


def get_batch_data(mini_batches_df: pd.DataFrame, batch_id: int):
    if batch_id not in mini_batches_df.index:
        raise IndexError(f"Index {batch_id} not found in the dataframe.")

    row = mini_batches_df.iloc[batch_id]
    batch_data = [
        '\n'.join([' '.join(seq) for seq in cell]) if cell is not None else '' for cell in row
    ]
    return '\n'.join(batch_data)


data_path = '../../../data'

original_data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv')
class_mapping = {label: idx for idx, label in enumerate(original_data['goal_status'].unique())}

train_df_file_path = data_path + '/avg_robot/train_df.csv'
test_df_file_path = data_path + '/avg_robot/test_df.csv'

# train_df_file_path = data_path + '/avg_robot/train_df_with_simple_events.csv'
# test_df_file_path = data_path + '/avg_robot/test_df_with_simple_events.csv'

train_df_loaded = pd.read_csv(train_df_file_path)
test_df_loaded = pd.read_csv(test_df_file_path)

train_df_loaded['label'] = train_df_loaded['label'].map(class_mapping)
test_df_loaded['label'] = test_df_loaded['label'].map(class_mapping)

mini_batch_size = 20000  # to get the entire training/testing set in one batch, in order to write it to a file
target_class = 2

for col in ['discretized_example', 'logical_form']:
    train_df_loaded[col] = train_df_loaded[col].apply(string_to_list)
    test_df_loaded[col] = test_df_loaded[col].apply(string_to_list)

mini_batches_df_train = generate_mini_batches(train_df_loaded, mini_batch_size, target_class)
mini_batches_df_test = generate_mini_batches(test_df_loaded, mini_batch_size, target_class)

train_batch_data_strings = convert_to_string(mini_batches_df_train)
test_batch_data_strings = convert_to_string(mini_batches_df_test)

with open('../../../data/avg_robot/folds/fold_0/train.csv', 'w') as file:
    file.write(train_batch_data_strings[0])

with open('../../../data/avg_robot/folds/fold_0/test.csv', 'w') as file:
    file.write(test_batch_data_strings[0])
