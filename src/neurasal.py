import os
import sys
import torch
import torch.nn as nn
import time

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.dsfa_old.mnist_seqs_new import get_data_loaders
from src.asal.logger import *
from src.asal_nesy.neurasal.pre_train_model import pre_train
from src.asal_nesy.neurasal.utils import *
from src.asal_nesy.pre_train_cnn import SimpleCNN

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)
# print(sys.path[0])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device: {device}')

if __name__ == "__main__":
    # Learn an SFA from some initial fully labeled sequences

    asal_train_path = f'{project_root}/data/mnist_nesy/train.csv'
    max_states = 4
    target_class = 1
    sfa = induce_sfa(asal_train_path, max_states, target_class, time_lim=30)

    # Neural stuff follow
    pre_training_size = 10  # num of fully labeled seed sequences.
    num_epochs = 100
    batch_size = 1
    cnn_classes = 10  # digits num. for MNIST
    model = DigitCNN(out_features=cnn_classes)
    # model = SimpleCNN()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.1
    criterion = nn.BCELoss()
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    logger.info(f'Pre-training on images from {pre_training_size} sequences')

    pre_train_model(train_loader, test_loader, 10, model, optimizer)

    logger.info(f'Training with the SFA...')

    for epoch in range(num_epochs):
        actual, predicted = [], []
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            img_sequences, labels, symbolic_sequences = batch[0], batch[1], batch[2]
            img_sequences, labels, symbolic_sequences = (
                img_sequences.to(device), labels.to(device), symbolic_sequences.to(device))

            """Need to fix the batches here. Currently works for batch size = 1 only"""

            sequence = img_sequences
            seq_size = sequence.shape[1]  # Keep Sequence length for later
            # Make the sequence of size (batch_size * seq_len, 1, 28, 28)
            sequence = sequence.view(-1, sequence.shape[2], sequence.shape[3], sequence.shape[4])

            start_time = time.time()
            nn_outputs = model(sequence, apply_softmax=True)

            # Transpose the tensor to align predictions per digit
            output_transposed = nn_outputs.T  # Shape becomes [10, 10]

            # Create dictionary mapping each digit to its respective predictions
            weights = {sfa.symbols[i]: output_transposed[i] for i in range(len(sfa.symbols))}

            labelling_function = create_labelling_function(weights, sfa.symbols)

            acceptance_probability = sfa.forward(labelling_function)

            loss = criterion(acceptance_probability, labels.float())
            # Collect stats for training F1
            pred = (acceptance_probability >= 0.5)

            print(loss, pred, labels, acceptance_probability)

            backprop(loss, optimizer)

            actual.extend(labels)
            predicted.extend(pred)
            total_loss += loss.item()

        logger.info(
            f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.3f}, Time: {(time.time() - start_time) / 60:.2f} mins')

        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        logger.info(f'Train F1: {train_f1} ({tps}, {fps}, {fns})')

        # test_model_fixed_sfa(model, max_states, test_loader)
