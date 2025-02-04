from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.metrics import dtw
from fastdtw import fastdtw

"""
Runs KNN for time series classification on the AGV robot data. Includes code for splitting the data into
sub-series and supports different distance functions for the KNN classifier: euclidian, Dynamic Time Warping (DTW)
from tslearn (which is computationally intensive) and DTW from the fastdtw package, which is supposed to be more
efficient.
"""


def split_time_series(df: pd.DataFrame, k: int, m: int) -> List[Tuple[pd.DataFrame, str]]:
    """
    Splits a multivariate time series DataFrame into sub-series of length k with a sliding window
    that moves ahead m points at a time. Each sub-series is labeled by the label of its last time point.

    Parameters:
    - df: The multivariate time series DataFrame.
    - k: The length of each sub-series.
    - m: The number of points the window moves ahead each time.

    Returns:
    - A list of tuples where the first element is a sub-series DataFrame and the second element is its label.
    """
    # Initialize a list to store the sub-series and their labels
    sub_series = []

    # Iterate over the DataFrame with steps of size m until there are less than k points remaining
    for i in range(0, len(df) - k + 1, m):
        # Extract a sub-series of length k
        sub_df = df.iloc[i:i + k]

        # Extract the label of the last point in the sub-series
        label = sub_df.iloc[-1]['goal_status']

        # Append the sub-series and its label to the list
        sub_series.append((sub_df.drop(columns=['goal_status']), label))

    return sub_series


def generate_train_test_split(sub_series, iid=False, test_size=0.3):
    # Extracting features and labels
    if isinstance(sub_series, list):
        X = [sub[0] for sub in sub_series]
        y = [sub[1] for sub in sub_series]
    elif isinstance(sub_series, tuple):
        X, y = sub_series[0], sub_series[1]
    else:
        raise ValueError('Unexpected type for the sub_series argument.')
    if iid:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        # Create a DataFrame for easier manipulation
        df_sub_series = pd.DataFrame({'features': list(X), 'label': y})
        labels_of_interest = ['moving to Station1', 'moving to Station2', 'moving to Station3', 'moving to Station4',
                              'moving to Station5', 'moving to Station6', 'stopped (unknown)', 'stopped at Station1',
                              'stopped at Station2', 'stopped at Station3', 'stopped at Station4',
                              'stopped at Station5',
                              'stopped at Station6']

        # Filter the DataFrame
        df_filtered = df_sub_series[df_sub_series['label'].isin(labels_of_interest)]

        # Split the data into training and testing sets
        train_data = []
        test_data = []
        for label, group in df_filtered.groupby('label'):
            total = len(group)
            train_size = int((1 - test_size) * total)

            train_group = group.iloc[:train_size]
            test_group = group.iloc[train_size:]

            train_data.append(train_group)
            test_data.append(test_group)

        # Extracting the final training and testing data
        X_train = [row['features'] for index, row in pd.concat(train_data, ignore_index=True).iterrows()]
        y_train = [row['label'] for index, row in pd.concat(train_data, ignore_index=True).iterrows()]

        X_test = [row['features'] for index, row in pd.concat(test_data, ignore_index=True).iterrows()]
        y_test = [row['label'] for index, row in pd.concat(test_data, ignore_index=True).iterrows()]

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    k = 50  # Sub-sequence length
    m = 10  # Sliding window step

    knn_distance_function = 'euclidean'

    neighbours = 1  # 5

    data_path = '../../data'
    df = pd.read_csv(data_path + '/avg_robot/DemoDataset_1Robot.csv')
    # df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/out0.csv')

    sub_series = split_time_series(df, k, m)

    # Extract the features and labels from the sub-series
    # X = [sub[0].values for sub in sub_series]  # Features (time series data)
    # y = [sub[1] for sub in sub_series]  # Labels

    # Encode the labels into integers for training the classifier
    le = LabelEncoder()

    # Split the data into training and testing sets with stratification
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = generate_train_test_split(sub_series, iid=False)

    print(f'Train/test set sizes: {len(X_train), len(X_test)}')

    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Flatten the dataset since k-NN with DTW does not require the time series structure
    # We are essentially converting each time series into a feature vector
    X_train_flat = np.array(X_train).reshape(len(X_train), -1)
    X_test_flat = np.array(X_test).reshape(len(X_test), -1)

    knn = KNeighborsClassifier(n_neighbors=neighbours, metric=knn_distance_function)

    # Train the classifier
    knn.fit(X_train_flat, y_train_encoded)

    # Predict the labels of the test set
    y_pred = knn.predict(X_test_flat)

    # Compute the confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    class_report = classification_report(y_test_encoded, y_pred, digits=3, target_names=le.classes_)

    print(conf_matrix)
    print(class_report)
