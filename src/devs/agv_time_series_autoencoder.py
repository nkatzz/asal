import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

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

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = np.array([scaler.fit_transform(x) for x in X])

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, stratify=y_encoded,
                                                    random_state=42)

# Convert the data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).reshape((X_train.shape[0], -1))
X_test_tensor = torch.FloatTensor(X_test).reshape((X_test.shape[0], -1))


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Initialize the autoencoder model
input_dim = X_train_tensor.shape[1]
encoding_dim = 64  # You can adjust this value
model = Autoencoder(input_dim, encoding_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the autoencoder
epochs = 10000  # You can adjust this value
for epoch in range(epochs):
    encoded, decoded = model(X_train_tensor)
    loss = criterion(decoded, X_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Extract latent representations for training and testing sets
with torch.no_grad():
    X_train_encoded, _ = model(X_train_tensor)
    X_test_encoded, _ = model(X_test_tensor)

# Convert to numpy arrays
X_train_encoded = X_train_encoded.numpy()
X_test_encoded = X_test_encoded.numpy()

# Now X_train_encoded and X_test_encoded can be used for training downstream classifiers
