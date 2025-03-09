import torch.nn as nn
import torch
import os
import sys
import time

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.dsfa_old.mnist_seqs_new import get_data_loaders, get_data_loaders_OOD
from src.asal.logger import *
from src.asal_nesy.neurasal.utils import *


class CNN_LSTM(nn.Module):
    def __init__(self, cnn_model, lstm_hidden_size, lstm_num_layers, output_size=1):
        super(CNN_LSTM, self).__init__()
        self.cnn = cnn_model  # Assume this is a digit-classification CNN, e.g., DigitCNN
        self.lstm = nn.LSTM(
            input_size=cnn_model.out_features,  # Input size to LSTM is the CNN output size
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,  # Shape of input: (batch_size, seq_len, input_size)
        )
        self.fc = nn.Linear(lstm_hidden_size, output_size)  # Fully connected layer to output binary classification

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.size()
        # Reshape to process each element in the sequence through CNN
        x = x.view(-1, channels, height, width)  # Shape: (batch_size * seq_len, channels, height, width)
        # No softmax here!
        cnn_out = self.cnn(x, apply_softmax=False)  # Shape: (batch_size * seq_len, cnn_out_features)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, cnn_out_features)

        # Pass the sequence of CNN outputs into the LSTM
        lstm_out, _ = self.lstm(cnn_out)  # Shape: (batch_size, seq_len, lstm_hidden_size)

        # Take the last hidden state for classification (binary classification per sequence)
        final_out = self.fc(lstm_out[:, -1, :])  # Shape: (batch_size, output_size)
        return torch.sigmoid(final_out)  # Binary classification, use sigmoid activation


if __name__ == "__main__":
    seq_length, train_num, test_num = 10, 1000, 300
    OOD = False
    train_loader, test_loader = get_data_loaders_OOD(batch_size=50) if OOD else get_data_loaders(batch_size=50)

    # Initialize CNN + LSTM model
    cnn_model = DigitCNN(out_features=10)  # CNN outputs 10 features (one for each digit label)
    lstm_hidden_size = 128  # Define LSTM hidden state size
    lstm_num_layers = 1  # Number of LSTM layers

    model = CNN_LSTM(cnn_model, lstm_hidden_size, lstm_num_layers).to(device)

    # extremely sensitive to the lr, loss does not improve for lr=0.0001, or lr=0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification

    num_epochs = 200

    print("Training CNN + LSTM baseline...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            tensors, labels, sequences = batch  # tensors, labels, and sequences come from the dataset
            tensors, labels = tensors.to(device), labels.to(device)

            # Forward pass
            outputs = model(tensors)  # Shape: (batch_size, 1)
            loss = criterion(outputs.squeeze(), labels.float())  # Compute binary cross-entropy loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            actual, predicted = [], []
            for batch in test_loader:
                tensors, labels, sequences = batch
                tensors, labels = tensors.to(device), labels.to(device)

                outputs = model(tensors)
                predictions = (outputs.squeeze() >= 0.5).long()

                actual.extend(labels.cpu().numpy())
                predicted.extend(predictions.cpu().numpy())

        # Compute F1 score or other metrics
        test_f1 = f1_score(actual, predicted, average="macro")

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.3f}, "
              f"Test F1: {test_f1:.3f}, Time: {int(time.time() - start_time)} secs")
