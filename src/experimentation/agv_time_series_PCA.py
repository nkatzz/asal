import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

data_path = '../../data'
df = pd.read_csv(data_path + '/avg_robot/DemoDataset_1Robot.csv')


# Function to split dataset into subseries
def split_time_series(df, k, m):
    sub_series = []
    labels = []
    for i in range(0, len(df) - k + 1, m):
        sub_df = df.iloc[i:i + k]
        label = sub_df.iloc[-1]['goal_status']
        sub_series.append(sub_df.drop(columns=['goal_status']).values)
        labels.append(label)
    return np.array(sub_series), np.array(labels)


# Split the dataset into subseries
k = 50
m = 10
X, y = split_time_series(df, k, m)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
# X_scaled = np.array([scaler.fit_transform(x) for x in X])
X_scaled = X

# Apply PCA
n_components = 5  # You can adjust the number of components as needed
pca = PCA(n_components=n_components)
X_pca = np.array([pca.fit_transform(x) for x in X_scaled])

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.3, stratify=y_encoded,
                                                    random_state=42)

# Now, X_train and X_test are the reduced-dimensionality versions of the original subseries
