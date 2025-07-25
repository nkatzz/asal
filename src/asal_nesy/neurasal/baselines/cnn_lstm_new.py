import os
import sys
import time
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.neurasal.utils import *
from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset, TensorSequence

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, cnn_model, lstm_hidden_size, lstm_num_layers,
                 output_size=1, cnn_return_features=True, with_cnn_dropout=False, aggregation='concat'):
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
        self.return_features = cnn_return_features
        self.cnn_dropout = with_cnn_dropout

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

        # cnn_out = self.cnn(x, apply_softmax=False)  # (bs * seqlefn * dim, cnn_out_features)
        cnn_out = self.cnn(x, apply_softmax=False, return_features=self.return_features, with_dropout=self.cnn_dropout)  # Returning logits (not features) works better!
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
            ).to(device)

        lstm_out, _ = self.lstm(cnn_out)  # (bs, seqlen, lstm_hidden_size)
        final_out = self.fc(lstm_out[:, -1, :])  # (bs, output_size)

        # print("CNN out mean/std:", round(cnn_out.mean().item(), 3), round(cnn_out.std().item(), 3))
        # print(torch.sigmoid(final_out).mean().item())

        return torch.sigmoid(final_out)


def test_model(model, data_loader: DataLoader[SequenceDataset], probability_threshold=0.5):
    model.eval()
    with torch.no_grad():
        actual, predicted = [], []
        total_loss_1 = 0.0
        for batch in data_loader:
            tensors = torch.stack([seq.images for seq in batch]).to(device)  # (bs, seqlen, dim, c, h, w)
            labels = torch.tensor([seq.seq_label for seq in batch]).to(device)  # (BS,)
            # labels = labels.view(-1)

            outputs = model(tensors)
            # predictions = (outputs.squeeze() >= 0.5).long()
            predictions = (outputs >= probability_threshold).long().view(-1)

            actual.extend(labels.cpu().numpy())
            predicted.extend(predictions.cpu().numpy())

            # pred_probs = outputs.view(-1)
            # true_labels = labels.float()
            # loss_eval = nn.BCELoss(reduction='sum')(pred_probs, true_labels).item()
            # total_loss_1 += loss_eval

    f1 = f1_score(actual, predicted, average='binary', zero_division=0)
    # print("Eval BCE Loss:", total_loss_1 / len(test_data))
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    return f1, tp, fp, fn, tn

def get_symbolic_seqs(dataset: SequenceDataset, write_seqs_to):
    for seq in dataset:
        labels = torch.tensor([seq.seq_label for seq in batch]).to(device)
        with open(write_seqs_to, "w") as f:
            for seq in labels:
                f.write(seq.item())
        f.close()


if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # seq_length, train_num, test_num = 10, 1000, 300
    # OOD = True
    # train_loader, test_loader = get_data_loaders_OOD(batch_size=50) if OOD else get_data_loaders(batch_size=50)

    batch_size = 50
    num_epochs = 200
    pre_train_cnn_for = 0  # num of epochs, set to 0 for no pre-training
    lr =  0.0001  # 0.005
    cnn_return_features = True
    cnn_dropout = False

    train_data, test_data = get_data('/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_train.pt',
                                     '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_test.pt')

    from collections import Counter

    label_counts = Counter(seq.seq_label for seq in train_data)
    print("Label distribution in train data:", label_counts)

    label_counts = Counter(seq.seq_label for seq in test_data)
    print("Label distribution in test data:", label_counts)

    # Just for debugging when testing on the training set, make sure that the data are the same.
    # assert all(seq1.seq_id == seq2.seq_id for seq1, seq2 in zip(train_data, test_data))

    train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, batch_size=batch_size, train=True)
    test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, batch_size=batch_size, train=False)

    # Initialize CNN + LSTM model
    cnn_model = DigitCNN(out_features=10)  # CNN outputs 10 features (one for each digit label)
    lstm_hidden_size = 128  # Define LSTM hidden state size
    lstm_num_layers = 1  # Number of LSTM layers

    model = CNN_LSTM(cnn_model, lstm_hidden_size, lstm_num_layers,
                     cnn_return_features=cnn_return_features, with_cnn_dropout=cnn_dropout).to(device)

    # extremely sensitive to the lr, loss does not improve for lr=0.0001, or lr=0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 0.001  0.005
    criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification

    if pre_train_cnn_for > 0:
        from src.asal_nesy.neurasal.neurasal_functions import pretrain_nn
        class_attrs = ['d1']
        print(f'Pretraining CNN for {pre_train_cnn_for} epochs')
        pretrain_nn(train_data, test_data, 0,
                    cnn_model, optimizer, class_attrs, with_fully_labelled_seqs=True, num_epochs=pre_train_cnn_for)


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

        train_f1, train_tps, train_fps, train_fns, train_tns = test_model(model, train_loader, probability_threshold=0.5)
        test_f1, test_tps, test_fps, test_fns, test_tns = test_model(model, test_loader, probability_threshold=0.5)

        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader):.3f}, Time: {time.time() - start_time:.3f} secs\n"
              f"Train (F1, tps, fps, fns, tns): ({train_f1:.3f}, {train_tps}, {train_fps}, {train_fns}, {train_tns})\n"
              f"Test  (F1, tps, fps, fns, tns): ({test_f1:.3f}, {test_tps}, {test_fps}, {test_fns}, {test_tns})")

        """
        print('')
        for p in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
            test_f1, test_tps, test_fps, test_fns, test_tns = test_model(model, test_loader, probability_threshold=p)
            print (p, test_f1, test_tps, test_fps, test_fns)
        """




