import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

# Use the discretizer followed by the LSTM
symbolize = True

# Covert the probability distribution over the alphabet to actual symbols.
# If false the LSTM classifier works with sequences of prob. dists. over the alphabet, instead of symbols.
# Note that in the latter case we get much better results, since the soft assignment retains more
# information from the original data and also facilitates the gradient flow during backpropagation
# (argmax, which is used to derive the most likely symbol from a distribution is not differentiable).
argmax_discretize = False

# Here are some indicative results from the Maritime dataset with the different settings tested in this script:
# - Pure LSTM: 0.98 (positive class)
# - Discretizer + LSMT with argmax:  0.86 (simple architecture)
#                                    0.56 (complex architecture)
# - Discretizer + LSMT with softmax: 0.96 (simple architecture)
# - Discretizer + LSMT with softmax, argmax during testing: 0.90

# Regardless of whether we used the argmax during training, use it during testing to convert the testing
# data into discrete sequences.
test_with_symbols = True

# The number of symbols to use for discretization
num_symbols = 10

num_epochs = 1000

file_path = '/media/nkatz/storage/asal-seqs/maritime/Maritime-time-series/Maritime.csv'
df = pd.read_csv(file_path)

# Reshape the data so that each instance (mmsi) has all features across all time points
num_features = 5  # 'longitude', 'latitude', 'speed', 'heading', 'course_over_ground'
num_time_points = 30  # Each time series has 30 time points
instances = df.shape[0] // num_features

# Reshape features
X = df.drop(columns=["label", "mmsi"]).values.reshape(instances, num_features, num_time_points)

# Get labels
y = df["label"].values[::num_features]  # We only need to take every num_features-th label because they are repeating

# Create a 70/30 stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Checking the shapes
print('Shapes:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Checking if GPU is available, and if not, using CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
if not torch.cuda.is_available():
    torch.set_num_threads(1)
"""


# Define the LSTM model architecture again
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out.squeeze(1)


class DiscretizeAndClassify(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_symbols):
        super(DiscretizeAndClassify, self).__init__()

        # Discretization layer
        self.discretize = nn.Linear(input_size, num_symbols)

        # the output of self.discretize is a tensor of shape (batch_size, sequence_length, num_symbols),
        # so applying softmax along dim=2 normalizes the probabilities of the symbols at each time step
        # across all batches.
        self.softmax = nn.Softmax(dim=2)

        # Classification layer
        self.lstm = nn.LSTM(num_symbols, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Discretize
        x = self.discretize(x)
        x = self.softmax(x)

        # pick the most likely symbol from the softmax distributions.
        if argmax_discretize or (test_with_symbols and not self.training):
            x = torch.argmax(x, dim=2)  # Convert to symbols

            # Convert the indices of the symbols into one-hot encoded vectors,
            # ensuring that the input to the LSTM has the correct feature size.
            x = nn.functional.one_hot(x, num_classes=num_symbols).float()

        # Classify
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])

        # Using the sigmoid is the standard way to go when dealing with binary classification problems.
        # We'd use softmax for a binary classification problem.
        out = self.sigmoid(out)
        return out.squeeze(1)


class ComplexDiscretizeAndClassify(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_symbols):
        super(ComplexDiscretizeAndClassify, self).__init__()

        # More complex discretization layer
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, num_symbols)
        )
        self.softmax = nn.Softmax(dim=2)

        # Classification layer
        self.lstm = nn.LSTM(num_symbols, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # More complex discretization
        x = self.encoder(x)
        x = self.softmax(x)
        x = torch.argmax(x, dim=2)
        x = nn.functional.one_hot(x, num_classes=num_symbols).float()

        # Classify
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out.squeeze(1)


# Instantiate the model and move it to the appropriate device
# Instantiate the model
input_size = num_features
hidden_size = 64
num_layers = 3

if not symbolize:
    model = LSTMClassifier(input_size, hidden_size, num_layers).to(device)
else:
    model = DiscretizeAndClassify(input_size, hidden_size, num_layers, num_symbols)
    # model = ComplexDiscretizeAndClassify(input_size, hidden_size, num_layers, num_symbols)

# Display the model architecture and the device being used
print(model, device)

# Binary Cross Entropy Loss as the loss function
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate

# verification
print(model, criterion, optimizer, device)

# Converting the dataset into PyTorch tensors and moving them to the appropriate device.
# With permute the input tensor shape is changed from (batch_size, num_features, sequence_length)
# to (batch_size, sequence_length, num_features).
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Creating TensorDatasets for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Creating DataLoaders for training and testing data
batch_size = 64  # You can adjust the batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Training loop with batches
for epoch in range(num_epochs):
    model.train()  # set to training mode.
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation with batches
model.eval()  # set to testing mode.
all_outputs = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        outputs = (outputs > 0.5).float()
        all_outputs.extend(outputs.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Print classification report
print(classification_report(all_labels, all_outputs))