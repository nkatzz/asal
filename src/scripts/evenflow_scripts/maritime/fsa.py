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

file_path = '/media/nkatz/storage/asal-seqs/maritime/Maritime-time-series/Maritime.csv'
df = pd.read_csv(file_path)

# Reshape and split the data
num_features = 5
num_time_points = 30
instances = df.shape[0] // num_features

X = df.drop(columns=["label", "mmsi"]).values.reshape(instances, num_time_points, num_features)
y = df["label"].values[::num_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# Dataset Class
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

"""
class DiscretizeAndAutomatonClassify(nn.Module):
    def __init__(self, input_size, num_symbols, num_states, num_classes, l1_reg):
        super(DiscretizeAndAutomatonClassify, self).__init__()

        # Discretization layer
        self.discretize = nn.Linear(input_size, num_symbols)
        self.softmax = nn.Softmax(dim=2)

        # Transition matrices for each symbol
        self.P = nn.Parameter(torch.randn(num_symbols, num_states, num_states))
        nn.init.xavier_uniform_(self.P)

        # L1 regularization strength
        self.l1_reg = l1_reg

        # Accepting states' weights
        self.accepting_weights = nn.Parameter(torch.randn(num_states, num_classes))
        nn.init.xavier_uniform_(self.accepting_weights)

    def forward(self, x):
        # Soft discretization
        x = self.discretize(x)
        x = self.softmax(x)  # Keep it as probability distributions

        batch_size, seq_len, _ = x.size()

        # Initial state distribution (assuming start at state 0)
        state_dist = torch.zeros(batch_size, self.P.size(1)).to(x.device)
        state_dist[:, 0] = 1.0

        for t in range(seq_len):
            # Calculate the weighted transition matrix for each batch
            weighted_P = torch.einsum('ijk,bi->bjk', self.P, x[:, t, :])

            # Update state distribution
            state_dist = torch.bmm(state_dist.unsqueeze(1), weighted_P).squeeze(1)

        # Calculate the probabilities of the accepting states
        accepting_probs = F.softmax(self.accepting_weights, dim=1)
        output = torch.matmul(state_dist, accepting_probs)

        # print(output)

        # Add L1 regularization to the loss
        # l1_loss = self.l1_reg * torch.sum(torch.abs(self.P.clone()))
        # self.add_loss(l1_loss)

        return output

    def add_loss(self, val):
        if hasattr(self, 'additional_loss'):
            self.additional_loss += val
        else:
            self.additional_loss = val

    def get_additional_loss(self):
        return getattr(self, 'additional_loss', 0)
"""


# Neural Automaton
class NeuralAutomaton(nn.Module):
    def __init__(self, num_states, num_symbols, num_classes):
        super(NeuralAutomaton, self).__init__()
        self.num_states = num_states
        self.num_symbols = num_symbols

        # Initialized with softmax to ensure they are valid probabilities
        self.P = nn.Parameter(torch.softmax(torch.randn(num_symbols, num_states, num_states), dim=2))
        self.start_state = nn.Parameter(torch.softmax(torch.randn(num_states), dim=0), requires_grad=False)
        self.accept_states = nn.Parameter(torch.softmax(torch.randn(num_states, num_classes), dim=1))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialization of state probabilities with batch support
        state_probs = self.start_state.unsqueeze(0).repeat(batch_size, 1)

        # Iteratively update state probabilities
        for t in range(seq_len):
            # Compute the weighted sum of transition matrices for each symbol in the batch
            weighted_P = torch.einsum('bs,snm->bnm', x[:, t, :], self.P)

            # Update state probabilities
            state_probs = torch.einsum('bn,bnm->bm', state_probs, weighted_P)

        # Compute final class probabilities
        class_probs = torch.einsum('bn,bnc->bc', state_probs, self.accept_states)

        return class_probs


# Discretization and Classification Model
class DiscretizeAndClassify(nn.Module):
    def __init__(self, input_size, num_symbols, automaton):
        super(DiscretizeAndClassify, self).__init__()
        self.discretize = nn.Linear(input_size, num_symbols)
        self.softmax = nn.Softmax(dim=2)
        self.automaton = automaton

    def forward(self, x):
        x = self.discretize(x)
        x = self.softmax(x)
        x = self.automaton(x)
        return x


num_states = 10
num_symbols = 10
num_classes = 2

# Instantiate the Model
automaton = NeuralAutomaton(num_states, num_symbols, num_classes)
model = DiscretizeAndClassify(input_size=5, num_symbols=num_symbols, automaton=automaton)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(100):
    model.train()
    total_loss = 0.0

    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()

        # Corrected tensor construction
        outputs = model(batch_X.clone().detach().float())

        # Corrected tensor construction and added retain_graph=True
        loss = criterion(outputs, batch_Y.clone().detach().long())
        # loss += model.get_additional_loss()
        loss.backward(retain_graph=True)  # Added retain_graph=True
        # loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item() * len(batch_Y)

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataset)}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        outputs = model(torch.tensor(batch_X, dtype=torch.float32))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_Y.numpy())

# Print Classification Report
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))
