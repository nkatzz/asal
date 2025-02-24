import os
import sys
import torch
import torch.nn as nn
import time

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN, NonTrainableNeuralSFA
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
    # from src.asal_nesy.neurasal.debug_mnist_even_odd import get_sfa
    # sfa = get_sfa()

    pre_training_size = 10  # num of fully labeled seed sequences.
    num_epochs = 100

    # batch size vs lr: bs=50 --> lr=0.01, bs=1 --> lr=0.001
    batch_size = 50
    cnn_output_size = 10  # digits num. for MNIST
    model = DigitCNN(out_features=cnn_output_size)
    # model = SimpleCNN()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 0.001
    criterion = nn.BCELoss()
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    logger.info(f'Pre-training on images from {pre_training_size} sequences')

    # num_samples is number of randomly selected sequences. We pre-train on every image from these seqs.
    pre_train_model(train_loader, test_loader, 10, model, optimizer, num_epochs=100)

    logger.info(f'Training with the SFA...')

    for epoch in range(num_epochs):
        actual, predicted = [], []
        actual_latent, predicted_latent = [], []
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            img_sequences, labels, symbolic_sequences = batch[0], batch[1], batch[2]
            img_sequences, labels, symbolic_sequences = (
                img_sequences.to(device), labels.to(device), symbolic_sequences.to(device))

            sequence_length = img_sequences.shape[1]

            # Make the sequence of size (batch_size * seq_len, 1, 28, 28)
            img_sequences = img_sequences.view(-1, img_sequences.shape[2], img_sequences.shape[3],
                                               img_sequences.shape[4])

            nn_outputs = model(img_sequences, apply_softmax=True)
            nn_outputs = nn_outputs.view(batch_size, sequence_length, cnn_output_size)

            # store for latent concept prediction performance
            actual_latent.append(symbolic_sequences.flatten().squeeze(0).cpu())
            predicted_latent.append(torch.argmax(nn_outputs, dim=2).flatten().cpu())

            # Transpose the tensor so that the rows are the probabilities per variable
            output_transposed = nn_outputs.transpose(1, 2)

            # Create dictionary mapping each digit to its respective predictions
            probabilities = {sfa.symbols[i]: output_transposed[:, i, :] for i in range(len(sfa.symbols))}

            labelling_function = create_labelling_function(probabilities, sfa.symbols)

            acceptance_probabilities = torch.clamp(sfa.forward(labelling_function), 0, 1)

            loss = criterion(acceptance_probabilities, labels.float())
            # Collect stats for training F1
            predictions = (acceptance_probabilities >= 0.5)

            """==========================================DEBUG=============================================="""
            # prob_bebug, loss_debug, _, _ = process_sequences_debug(batch, model_debug, criterion, max_states)
            # print(loss, pred, labels, acceptance_probability)
            # print(f'loss: {loss}|{loss_debug}, prob: {acceptance_probability}|{prob_bebug}')
            """==========================================DEBUG=============================================="""

            backprop(loss, optimizer)

            actual.extend(labels)
            predicted.extend(predictions)
            total_loss += loss.item()

        actual_latent = torch.cat(actual_latent).numpy()  # Concatenates list of tensors into one
        predicted_latent = torch.cat(predicted_latent).numpy()

        latent_f1_macro = f1_score(actual_latent, predicted_latent, average="macro")
        latent_f1_micro = f1_score(actual_latent, predicted_latent, average="micro")

        test_f1, test_latent_f1_macro, t_tps, t_fps, t_fns = test_model(model, sfa, test_loader,
                                                                        batch_size, cnn_output_size)

        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        logger.info(
            f'Epoch {epoch}\nLoss: {total_loss / len(train_loader):.3f}, Time: {int(time.time() - start_time)} secs\n'
            f'Train F1: {train_f1:.3f} ({tps}, {fps}, {fns}) | latent: {latent_f1_macro:.3f}\n'
            f'Test F1: {test_f1:.3f} ({t_tps}, {t_fps}, {t_fns}) | latent: {test_latent_f1_macro:.3f}')
