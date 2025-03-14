import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data_path = '../../../data'
df = pd.read_csv(data_path + '/avg_robot/DemoDataset_1Robot.csv')

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(df.drop(columns=['goal_status']))

# Convert labels to numpy array
labels = df['goal_status'].values

# Create sub-series
k = 300  # Length of each sub-series
m = 300  # Step size
sub_series = [features[i: i + k] for i in range(0, len(features) - k + 1, m)]
sub_labels = [labels[i + k - 1] for i in range(0, len(labels) - k + 1, m)]


# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Convert entire dataset to PyTorch tensor
features_tensor = torch.tensor(features, dtype=torch.float32)

# Create DataLoader for the entire dataset
dataset = TensorDataset(features_tensor)
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = Autoencoder(input_dim=features.shape[1], latent_dim=5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for the autoencoder
num_epochs = 50

for epoch in range(num_epochs):
    for batch_features, in data_loader:
        encoded, decoded = model(batch_features)
        loss = criterion(decoded, batch_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Extract latent features for the entire dataset
with torch.no_grad():
    latent_features = model.encoder(features_tensor).numpy()

# Apply clustering on the latent features
kmeans = KMeans(n_clusters=20)  # Adjust the number of clusters as needed
kmeans.fit(latent_features)
cluster_ids = kmeans.predict(latent_features)

# Map each point in each sub-series to the corresponding cluster
symbolic_sequences = [cluster_ids[i: i + k] for i in range(0, len(cluster_ids) - k + 1, m)]

# Write symbolic sequences and their labels to a file
with open('symbolic_sequences.csv', 'w') as file:
    for seq, label in zip(symbolic_sequences, sub_labels):
        seq_str = ','.join(map(str, seq))
        file.write(f"{seq_str},{label}\n")

print("Symbolic sequences and labels written to symbolic_sequences.csv")
