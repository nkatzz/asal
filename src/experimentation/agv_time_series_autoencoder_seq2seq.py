import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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
k = 50
m = 10
X, y = split_time_series(df, k, m)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = np.array([scaler.fit_transform(x) for x in X])

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, stratify=y_encoded,
                                                    random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)


# Define the sequence-to-sequence autoencoder
class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Seq2SeqAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return encoded, decoded


# Create an instance of the model
input_dim = X_train.shape[2]  # Number of features
latent_dim = 20  # Dimensionality of the latent representations
model = Seq2SeqAutoencoder(input_dim, latent_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 2000
for epoch in range(epochs):
    encoded, decoded = model(X_train_tensor)
    loss = criterion(decoded, X_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Extract latent sequence representations
with torch.no_grad():
    X_train_encoded, _ = model.encoder(X_train_tensor)
    X_test_encoded, _ = model.encoder(X_test_tensor)

# Convert to numpy arrays
X_train_encoded = X_train_encoded.numpy()
X_test_encoded = X_test_encoded.numpy()

# Now X_train_encoded and X_test_encoded can be used for training downstream sequence classifiers
