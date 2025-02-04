import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from agv_time_series_KNN import generate_train_test_split

data_path = '../../data'
df = pd.read_csv(data_path + '/avg_robot/DemoDataset_1Robot.csv')


# Function to split dataset into sub-series
def split_time_series(df, k, m):
    sub_series = []
    labels = []
    for i in range(0, len(df) - k + 1, m):
        sub_df = df.iloc[i:i + k]
        label = sub_df.iloc[-1]['goal_status']
        sub_series.append(sub_df.drop(columns=['goal_status']).values)
        labels.append(label)
    return np.array(sub_series), np.array(labels)



# Split the dataset into sub-series
k = 50  # Length of each sub-series
m = 10  # The window moves ahead m points at a time
X, y = split_time_series(df, k, m)

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ------------------------
# This ruins the F1-score!
# ------------------------
# Standardize the features
# scaler = StandardScaler()
# X = np.array([scaler.fit_transform(x) for x in X])

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
X_train, X_test, y_train, y_test = generate_train_test_split((X, y), iid=False)

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Create a PyTorch dataset
class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create a DataLoader
train_data = TSDataset(X_train, y_train)
test_data = TSDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2, num_classes=len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch[{epoch + 1}/{num_epochs}], Step[{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Test the model and get predictions
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, digits=3, target_names=le.classes_)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
