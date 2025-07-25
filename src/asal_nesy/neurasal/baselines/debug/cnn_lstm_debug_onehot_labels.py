import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.asal_nesy.dsfa_old.models import DigitCNN
#from src.asal_nesy.neurasal.utils import *
from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset, TensorSequence

"""
Train the LSTM with the one-hot encoding of the digit labels instead of the CNN predictions.

This is to debug why the CNN_LSTM model cannot generalize even in-distribution. This version works fine, which
mean that the problem is with the combination of the CNN and the LSTM.
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_one_hot_sequence(batch, num_classes=10, aggregation='concat'):
    """
    Converts digit labels from a batch of TensorSequences into one-hot sequences.

    :param batch: List of TensorSequence objects
    :param num_classes: Number of classes (e.g., 10 for MNIST digits)
    :param aggregation: 'concat' or 'mean'
    :return: Tensor of shape (bs, seq_len, dim*num_classes) or (bs, seq_len, num_classes)
    """
    bs = len(batch)
    seq_len = batch[0].seq_length
    dim = batch[0].dimensionality

    one_hot = torch.zeros((bs, seq_len, dim, num_classes), device=device)
    for i, seq in enumerate(batch):
        for t in range(seq.seq_length):
            for d in range(seq.dimensionality):
                label_dict = seq.image_labels[t][d]
                # Assume the digit label is stored as the first value
                digit = list(label_dict.values())[0]
                one_hot[i, t, d, digit] = 1

    if aggregation == 'concat':
        one_hot = one_hot.view(bs, seq_len, -1)
    elif aggregation == 'mean':
        one_hot = one_hot.mean(dim=2)
    else:
        raise ValueError("aggregation must be 'concat' or 'mean'")
    return one_hot


class OneHot_Sequence_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)


def train_onehot_lstm(train_loader, test_loader, num_classes=10, aggregation='concat', epochs=50):
    input_size = num_classes * train_loader.dataset.sequences[0].dimensionality if aggregation == 'concat' else num_classes
    model = OneHot_Sequence_Classifier(input_size, 128, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            labels = torch.tensor([seq.seq_label for seq in batch], device=device).float()
            x = get_one_hot_sequence(batch, num_classes, aggregation)
            outputs = model(x)
            loss = criterion(outputs.view(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        f1 = evaluate_onehot(model, test_loader, num_classes, aggregation)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.3f}, Test F1: {f1:.3f}")

    return model


def evaluate_onehot(model, loader, num_classes, aggregation):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            labels = torch.tensor([seq.seq_label for seq in batch], device=device)
            x = get_one_hot_sequence(batch, num_classes, aggregation)
            outputs = model(x)
            preds = (outputs >= 0.5).long().view(-1)
            actual.extend(labels.cpu().numpy())
            predicted.extend(preds.cpu().numpy())
    return f1_score(actual, predicted, average="binary")


if __name__ == "__main__":
    train_path = '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_train.pt'
    test_path = '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_test.pt'

    # test_path = '/home/nkatz/dev/asal_data/mnist_nesy/len_50_dim_1_pattern_sfa_1/mnist_test.pt'

    train_data, test_data = get_data(train_path=train_path, test_path=test_path)
    train_loader = get_data_loader(train_data, batch_size=32, train=True)
    test_loader = get_data_loader(test_data, batch_size=32, train=False)
    train_onehot_lstm(train_loader, test_loader)
