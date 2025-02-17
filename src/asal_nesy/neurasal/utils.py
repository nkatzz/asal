from torch.utils.data import Dataset, DataLoader
import random
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from src.asal.logger import *
import nnf

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class ImageDataset(Dataset):
    """Container class for individual image/label pairs"""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_indiv_data_loader(sequences):
    """Returns a data loader for individual image/label pairs"""
    # Extract image/digit pairs
    train_images = []
    train_labels = []
    for sequence, _, symbolic_sequence in sequences:
        sequence = sequence.squeeze(0)  # Remove batch dimension
        symbolic_sequence = symbolic_sequence.squeeze(0)
        for img, digit in zip(sequence, symbolic_sequence):
            train_images.append(img)  # Append individual image
            train_labels.append(digit)  # Append corresponding digit
    train_dataset = ImageDataset(train_images, train_labels)
    train_loader_individual = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader_individual


def pre_train_model(seq_train_loader, seq_test_loader, num_samples, model, optimizer, num_epochs=10):
    # un-batch the data for sampling.
    train_data_list = [list(x) for batch in seq_train_loader for x in zip(*batch)]

    # Convert train_loader to a list (efficient for small datasets)
    # train_data_list = list(seq_train_loader)
    selected_samples = random.sample(train_data_list, num_samples)

    train_loader_individual = get_indiv_data_loader(selected_samples)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader_individual:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, apply_softmax=False)  # Get logits
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader_individual):.4f}")

    # Test the model on individual images from the seq_test_loader
    # un-batch the data first.
    test_data_list = [list(x) for batch in seq_train_loader for x in zip(*batch)]
    test_loader_individual = get_indiv_data_loader(test_data_list)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader_individual:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, apply_softmax=True)  # Get probabilities

            _, predictions = torch.max(outputs, 1)  # Get class with highest probability

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

        test_f1 = f1_score(y_true, y_pred, average="micro")  # average="micro"
        logger.info(f'Pre-trained model F1-score on test set: {test_f1}')


def create_labelling_function(weights: dict, sfa_symbols: list[str]):
    def labelling_function(var: nnf.Var) -> torch.Tensor:
        if str(var.name) in sfa_symbols:
            return (
                weights[str(var.name)]
                if var.true
                else torch.ones_like(weights[str(var.name)])
            )
        return weights[str(var.name)] if var.true else 1 - weights[str(var.name)]

    return labelling_function


def backprop(batch_loss, optimizer):
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()


def get_stats(predicted, actual):
    # Ensure that predicted and actual are on the CPU and convert them to NumPy arrays
    predicted = [int(p) for p in predicted]
    actual = [int(a) for a in actual]

    # Calculate tp, fp, fn for binary classification metrics
    tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    tn = sum(p == 0 and a == 0 for p, a in zip(predicted, actual))

    # Calculate precision, recall, and F1-score for binary classification
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_binary = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate F1-macro and F1-weighted using sklearn
    f1_macro = f1_score(actual, predicted, average='macro')
    f1_weighted = f1_score(actual, predicted, average='weighted')

    return f1_binary, f1_macro, f1_weighted, tp, fp, fn, tn


def process_sequences_debug(data, model, criterion, num_states):
    sequence, label, symbolic_sequence = data[0], data[1], data[2]
    sequence, label, symbolic_sequence = sequence.to(device), label.to(device), symbolic_sequence.to(device)
    cnn_prediction, guard_prediction, final_states_distribution = model(sequence)
    acceptance_probability = final_states_distribution[:, num_states - 1]

    # print(f'Acceptance probability: {acceptance_probability}')
    acceptance_probability = torch.clamp(acceptance_probability, 0, 1)

    loss = criterion(acceptance_probability, label.float())
    # Collect stats for training F1
    predicted = (acceptance_probability >= 0.5)

    return acceptance_probability, loss, label, predicted


def process_sequence(sequence, symb_sequence, seq_label, model, sfa, criterion):
    # Make the sequence of size (batch_size * seq_len, 1, 28, 28)
    sequence = sequence.view(-1, sequence.shape[2], sequence.shape[3], sequence.shape[4])
    nn_outputs = model(sequence, apply_softmax=True)

    # Transpose the tensor to align predictions per digit
    output_transposed = nn_outputs.T  # Shape becomes [10, 10]

    # Create dictionary mapping each digit to its respective predictions
    weights = {sfa.symbols[i]: output_transposed[i] for i in range(len(sfa.symbols))}

    labelling_function = create_labelling_function(weights, sfa.symbols)

    acceptance_probability = torch.clamp(sfa.forward(labelling_function), 0, 1)
    loss = criterion(acceptance_probability, seq_label.float())
    prediction = (acceptance_probability >= 0.5)
    return loss, prediction


def test_model(model, sfa, test_loader, batch_size, cnn_output_size):
    actual, predicted = [], []
    actual_latent, predicted_latent = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
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

            acceptance_probability = torch.clamp(sfa.forward(labelling_function), 0, 1)

            # Collect stats for training F1
            pred = (acceptance_probability >= 0.5)

            actual.extend(labels)
            predicted.extend(pred)

        actual_latent = torch.cat(actual_latent).numpy()  # Concatenates list of tensors into one
        predicted_latent = torch.cat(predicted_latent).numpy()

        latent_f1_macro = f1_score(actual_latent, predicted_latent, average="macro")
        latent_f1_micro = f1_score(actual_latent, predicted_latent, average="micro")

        _, test_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        return test_f1, latent_f1_macro, tps, fps, fns
