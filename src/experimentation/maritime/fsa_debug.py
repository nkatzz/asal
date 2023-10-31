import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


class MaritimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SoftDiscretize(nn.Module):
    def __init__(self, input_size, num_symbols):
        super(SoftDiscretize, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, num_symbols)
        )

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
        self.num_classes = num_classes
        self.total_states = num_states + num_classes

        # Transition probabilities for non-accepting (regular) states
        self.P_non_accepting = nn.Parameter(
            torch.softmax(torch.randn(self.num_symbols, self.num_states, self.num_states), dim=2))

        # Transition probabilities from non-accepting to accepting states
        self.P_to_accepting = nn.Parameter(
            torch.softmax(torch.randn(self.num_symbols, self.num_states, self.num_classes), dim=2))

        self.total_P = nn.Parameter(torch.cat(
            (torch.cat((self.P_non_accepting, self.P_to_accepting), dim=2),
             torch.zeros(self.num_symbols, self.num_classes, self.total_states)), dim=1), requires_grad=True)

        # for i in range(num_states, self.total_states):
        #    self.total_P[:, i, i] = 1  # Make accepting states absorbing

        # Fixed start state
        self.start_state = nn.Parameter(torch.zeros(self.total_states), requires_grad=False)
        self.start_state[0] = 1

        self.l1_reg = l1_reg

    def forward(self, x):
        seq_len = x.size(0)
        state_probs = self.start_state.clone()

        for t in range(seq_len):
            # Compute the weighted sum of transition matrices for each symbol
            weighted_P = torch.einsum('s,snm->nm', x[t, :], self.total_P)

            # Update state probabilities
            state_probs = torch.mv(weighted_P, state_probs)

        return state_probs[self.num_states:]  # Return probabilities for accepting states

    def get_l1_loss(self):
        l1_loss = torch.norm(self.total_P, p=1)
        return self.l1_reg * l1_loss


def evaluate(data_loader, discretizer, automaton, device):
    discretizer.eval()
    automaton.eval()

    y_actual, y_predicted = [], []

    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            for i in range(len(batch_X)):
                x = batch_X[i].to(device)
                y = batch_Y[i].to(device)

                symbol_probabilities = discretizer(x)
                class_probabilities = automaton(symbol_probabilities)

                predicted_symbol = torch.argmax(class_probabilities).item()
                y_actual.append(y.item())
                y_predicted.append(predicted_symbol)

    # Print classification report
    print(classification_report(y_actual, y_predicted, target_names=["Class 0", "Class 1"]))


def train(train_loader, num_epochs, discretizer, automaton, device, test_loader=None):
    optimizer = torch.optim.Adam(list(discretizer.parameters()) + list(automaton.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, num_epochs):
        discretizer.train()
        automaton.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_Y in train_loader:
            batch_loss = 0.0
            optimizer.zero_grad()  # Reset gradients for each batch

            for i in range(len(batch_X)):
                x = batch_X[i].to(device)
                y = batch_Y[i].to(device)

                symbol_probs = discretizer(x.float())
                class_probs = automaton(symbol_probs)

                # Compute the loss
                loss = criterion(class_probs, y.long())
                batch_loss += loss

            # Average the loss over the batch and add regularization loss
            batch_loss = batch_loss / len(batch_X)
            batch_loss += automaton.get_l1_loss()

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        # Average loss
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss}")

        # print("Training set:")
        # evaluate(train_loader)
        print("On testing set:")
        evaluate(test_loader, discretizer, automaton, device)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    df = pd.read_csv('/media/nkatz/storage/asal-seqs/maritime/Maritime-time-series/Maritime.csv')

    # Reshape and split the data
    num_features = 5
    num_time_points = 30
    instances = df.shape[0] // num_features

    X = df.drop(columns=["label", "mmsi"]).values.reshape(instances, num_time_points, num_features)
    y = df["label"].values[::num_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    train_dataset = MaritimeDataset(X_train, y_train)
    test_dataset = MaritimeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_states = 3  # 10
    num_symbols = 10  # 15
    num_classes = 2
    num_epochs = 1000
    l1_param = 0.0001

    discretizer = SoftDiscretize(num_features, num_symbols).to(device)
    automaton = NeuralAutomaton(num_states, num_symbols, num_classes, l1_reg=l1_param).to(device)

    train(train_loader, num_epochs, discretizer, automaton, device, test_loader)
