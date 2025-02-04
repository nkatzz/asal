import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax

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
k = 50  # Length of each subseries
m = 10  # The window moves ahead m points at a time
X, y = split_time_series(df, k, m)

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# Scale the data
scaler = TimeSeriesScalerMinMax()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit k-NN classifier with DTW metric
knn_dtw = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
knn_dtw.fit(X_train_scaled, y_train)

# Predict the labels
y_pred = knn_dtw.predict(X_test_scaled)

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le.classes_)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
