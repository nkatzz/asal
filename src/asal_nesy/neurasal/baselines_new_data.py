import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.neurasal.utils import *
from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset, TensorSequence

# so the pytorch can load data from pickled objects generated via ASP
sys.path.append('/home/nkatz/dev/asal/src/asal_nesy/neurasal/mnist')

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, cnn_model, lstm_hidden_size, lstm_num_layers, output_size=1, aggregation='concat'):
        """
        aggregation: one of ['concat', 'mean']

        'concat' keeps all individual CNN outputs per image and concatenates
        them (requires dim * cnn_out_features input to LSTM).

        'mean' averages the outputs of the CNN over all dim images at each sequence point (simpler, fixed input size).

        Delayed LSTM init allows dynamic support for varying dim sizes when using 'concat'.
        """
        super(CNN_LSTM, self).__init__()
        self.cnn = cnn_model  # CNN model that outputs cnn_model.out_features per image
        self.aggregation = aggregation

        if aggregation == 'concat':
            self.input_per_timestep = None  # Set later dynamically in forward
        elif aggregation == 'mean':
            self.input_per_timestep = cnn_model.out_features
        else:
            raise ValueError("aggregation must be 'concat' or 'mean'")

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = None  # Define later based on actual input size after aggregation

        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, dim, channels, height, width)
        """
        bs, seqlen, dim, c, h, w = x.shape
        x = x.view(bs * seqlen * dim, c, h, w)  # (bs * seqlen * dim, c, h, w)

        # cnn_out = self.cnn(x, apply_softmax=False)  # (bs * seqlen * dim, cnn_out_features)
        cnn_out = self.cnn(x, apply_softmax=False, return_features=True)
        cnn_out = cnn_out.view(bs, seqlen, dim, -1)  # (bs, seqlen, dim, cnn_out_features)

        if self.aggregation == 'concat':
            # Concatenate features from all dim images
            cnn_out = cnn_out.view(bs, seqlen, -1)  # (bs, seqlen, dim * cnn_out_features)
            input_size = cnn_out.size(-1)
        elif self.aggregation == 'mean':
            cnn_out = cnn_out.mean(dim=2)  # (bs, seqlen, cnn_out_features)
            input_size = cnn_out.size(-1)

        # Initialize LSTM if not already done (delayed because input size depends on dim)
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                batch_first=True,
            )

        lstm_out, _ = self.lstm(cnn_out)  # (bs, seqlen, lstm_hidden_size)
        final_out = self.fc(lstm_out[:, -1, :])  # (bs, output_size)

        # print("CNN out mean/std:", round(cnn_out.mean().item(), 3), round(cnn_out.std().item(), 3))

        # print(torch.sigmoid(final_out).mean().item())

        return torch.sigmoid(final_out)


if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    """
    seq_length, train_num, test_num = 10, 1000, 300
    OOD = True
    train_loader, test_loader = get_data_loaders_OOD(batch_size=50) if OOD else get_data_loaders(batch_size=50)
    """

    batch_size = 50

    train_data, test_data = get_data('/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_train.pt',
                                     '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_test.pt')

    from collections import Counter

    label_counts = Counter(seq.seq_label for seq in train_data)
    print("Label distribution in train data:", label_counts)

    label_counts = Counter(seq.seq_label for seq in test_data)
    print("Label distribution in test data:", label_counts)

    # Just for debugging when testing on the training set, make sure that the data are the same.
    # assert all(seq1.seq_id == seq2.seq_id for seq1, seq2 in zip(train_data, test_data))

    train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, batch_size=50, train=True)
    test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, batch_size, train=False)

    # Initialize CNN + LSTM model
    cnn_model = DigitCNN(out_features=10)  # CNN outputs 10 features (one for each digit label)
    lstm_hidden_size = 128  # Define LSTM hidden state size
    lstm_num_layers = 2  # Number of LSTM layers

    model = CNN_LSTM(cnn_model, lstm_hidden_size, lstm_num_layers).to(device)

    # extremely sensitive to the lr, loss does not improve for lr=0.0001, or lr=0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.001
    criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification

    num_epochs = 200

    print("Training CNN + LSTM baseline...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            tensors = torch.stack([seq.images for seq in batch]).to(device)  # (bs, seqlen, dim, c, h, w)
            labels = torch.tensor([seq.seq_label for seq in batch]).to(device)  # (BS,)

            # Forward pass
            outputs = model(tensors)  # Shape: (batch_size, 1)
            # loss = criterion(outputs.squeeze(), labels.float())
            loss = criterion(outputs.view(-1), labels.float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            actual, predicted = [], []
            total_loss_1 = 0.0
            for batch in test_loader:
                tensors = torch.stack([seq.images for seq in batch]).to(device)  # (bs, seqlen, dim, c, h, w)
                labels = torch.tensor([seq.seq_label for seq in batch]).to(device)  # (BS,)
                # labels = labels.view(-1)

                outputs = model(tensors)
                # predictions = (outputs.squeeze() >= 0.5).long()
                predictions = (outputs >= 0.5).long().view(-1)

                actual.extend(labels.cpu().numpy())
                predicted.extend(predictions.cpu().numpy())

                pred_probs = outputs.view(-1)
                true_labels = labels.float()
                loss_eval = nn.BCELoss(reduction='sum')(pred_probs, true_labels).item()
                total_loss_1 += loss_eval

        test_f1 = f1_score(actual, predicted, average='binary', zero_division=0)
        print("Eval BCE Loss:", total_loss_1 / len(test_data))

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.3f}, "
              f"Test F1: {test_f1:.3f}, Time: {time.time() - start_time:.3f} secs")
