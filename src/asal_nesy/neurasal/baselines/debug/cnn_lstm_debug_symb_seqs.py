import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset


"""
This is just using the symbolic seqs (digit labels) to train the LSTM and works perfectly as expected.
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_symbolic_sequences(dataset: SequenceDataset, num_classes=10):
    """
    Extract digit sequences as tensors of shape (seq_len, num_classes) from the dataset.
    """
    symbolic = []
    labels = []
    for seq in dataset.sequences:
        seq_len = seq.seq_length
        dim = seq.dimensionality
        one_hot_seq = torch.zeros(seq_len, dim * num_classes)
        for t in range(seq_len):
            for d in range(dim):
                digit = list(seq.image_labels[t][d].values())[0]
                one_hot_seq[t, d * num_classes + digit] = 1
        symbolic.append(one_hot_seq)
        labels.append(seq.seq_label)
    return symbolic, labels


class SymbolicDataset(Dataset):
    def __init__(self, symbolic, labels):
        self.symbolic = symbolic
        self.labels = labels

    def __len__(self):
        return len(self.symbolic)

    def __getitem__(self, idx):
        return self.symbolic[idx], self.labels[idx]


class SymbolicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)


def train_symbolic_lstm(train_ds, test_ds, input_size, epochs=50):
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    model = SymbolicLSTM(input_size, 128).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            outputs = model(x)
            loss = criterion(outputs.view(-1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        f1 = evaluate_symbolic(model, test_loader)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.3f}, Test F1: {f1:.3f}")
    return model


def evaluate_symbolic(model, loader):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = (outputs >= 0.5).long().view(-1)
            actual.extend(y.cpu().numpy())
            predicted.extend(preds.cpu().numpy())
    return f1_score(actual, predicted, average="binary")


if __name__ == "__main__":

    train_path = '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_train.pt'
    test_path = '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_test.pt'

    train_data, test_data = get_data(train_path=train_path, test_path=test_path)
    train_symb, train_labels = extract_symbolic_sequences(train_data)
    test_symb, test_labels = extract_symbolic_sequences(test_data)
    input_size = train_symb[0].shape[1]
    train_ds = SymbolicDataset(train_symb, train_labels)
    test_ds = SymbolicDataset(test_symb, test_labels)
    train_symbolic_lstm(train_ds, test_ds, input_size)
