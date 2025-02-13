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


def pre_train_model(seq_train_loader, seq_test_loader, num_samples, model, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert train_loader to a list (efficient for small datasets)
    train_data_list = list(seq_train_loader)
    selected_samples = random.sample(train_data_list, num_samples)

    train_loader_individual = get_indiv_data_loader(selected_samples)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10
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
    test_loader_individual = get_indiv_data_loader(seq_test_loader)

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

        test_f1 = f1_score(y_true, y_pred, average="weighted")
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


def test_model_fixed_sfa(model, num_states, test_loader, where='Test'):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for sequence, label, symb_seq in test_loader:
            sequence = sequence.to(device)
            label = label.to(device)
            cnn_prediction, guard_prediction, states_probs = model(sequence)

            predicted_state = torch.argmax(states_probs, dim=1)
            prediction = []
            for ps in predicted_state:
                prediction += [1] if ps == num_states - 1 else [0]

            actual.extend(label.tolist())
            predicted.extend(prediction)

    _, f1, _, tps, fps, fns, _ = get_stats(predicted, actual)
    print(f'{where}  F1: {f1} ({tps}, {fps}, {fns})')
