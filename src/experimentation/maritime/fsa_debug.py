import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


df = pd.read_csv('/media/nkatz/storage/asal-seqs/maritime/Maritime-time-series/Maritime.csv')

# Reshape and split the data
num_features = 5
num_time_points = 30
instances = df.shape[0] // num_features

X = df.drop(columns=["label", "mmsi"]).values.reshape(instances, num_time_points, num_features)
y = df["label"].values[::num_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


class MaritimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = MaritimeDataset(X_train, y_train)
test_dataset = MaritimeDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class SoftDiscretize(nn.Module):
    def __init__(self, input_size, num_symbols):
        super(SoftDiscretize, self).__init__()
        self.fc = nn.Linear(input_size, num_symbols)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()  # Convert x to float32 to ensure dtype consistency
        x = self.fc(x)
        return self.softmax(x)


class NeuralAutomaton(nn.Module):
    def __init__(self, num_states, num_symbols, num_classes, l1_reg=0.01):
        super(NeuralAutomaton, self).__init__()
        self.num_states = num_states
        self.num_symbols = num_symbols
        self.P = nn.Parameter(torch.softmax(torch.randn(num_symbols, num_states, num_states), dim=2))
        self.start_state = nn.Parameter(torch.softmax(torch.randn(num_states), dim=0), requires_grad=False)
        self.accept_states = nn.Parameter(torch.softmax(torch.randn(num_states, num_classes), dim=1))
        self.l1_reg = l1_reg

    def forward(self, x):
        seq_len = x.size(0)
        state_probs = self.start_state.clone()

        for t in range(seq_len):
            weighted_P = torch.matmul(x[t, :], self.P)
            state_probs = torch.matmul(state_probs, weighted_P)

        class_probs = torch.matmul(state_probs, self.accept_states)
        return class_probs

    def get_l1_loss(self):
        l1_loss = torch.norm(self.P, p=1)
        return self.l1_reg * l1_loss



num_states = 10
num_symbols = 10
num_classes = 2
num_epochs = 3

discretizer = SoftDiscretize(input_size=5, num_symbols=num_symbols).to(device)
automaton = NeuralAutomaton(num_states=num_states, num_symbols=num_symbols, num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(list(discretizer.parameters()) + list(automaton.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_X, batch_Y in train_loader:
        batch_loss = 0.0

        for i in range(len(batch_X)):
            x = batch_X[i].to(device)
            y = batch_Y[i].to(device)

            symbol_probs = discretizer(x)
            class_probs = automaton(symbol_probs)

            loss = criterion(class_probs.unsqueeze(0), y.unsqueeze(0))
            loss += automaton.get_l1_loss()  # Add L1 regularization loss
            batch_loss += loss

        batch_loss /= len(batch_X)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")


# Switch to evaluation mode
discretizer.eval()
automaton.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        for i in range(len(batch_X)):
            x = batch_X[i].to(device)
            y = batch_Y[i].to(device)

            symbol_probs = discretizer(x)
            class_probs = automaton(symbol_probs)

            predicted = torch.argmax(class_probs).item()
            y_true.append(y.item())
            y_pred.append(predicted)

# Print classification report
print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))


