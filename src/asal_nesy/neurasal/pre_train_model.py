from torch.utils.data import Dataset, DataLoader
import random
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from src.asal.logger import *


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def pre_train(seq_train_loader, seq_test_loader, num_samples, model, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert train_loader to a list (efficient for small datasets)
    train_data_list = list(seq_train_loader)
    selected_samples = random.sample(train_data_list, num_samples)

    # Extract image/digit pairs
    train_images = []
    train_labels = []
    for sequence, _, symbolic_sequence in selected_samples:
        sequence = sequence.squeeze(0)  # Remove batch dimension
        symbolic_sequence = symbolic_sequence.squeeze(0)
        for img, digit in zip(sequence, symbolic_sequence):
            train_images.append(img)  # Append individual image
            train_labels.append(digit)  # Append corresponding digit

    # Create dataset
    train_dataset = ImageDataset(train_images, train_labels)
    train_loader_individual = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
    test_images = []
    test_labels = []
    for sequence, _, symbolic_sequence in seq_test_loader:
        sequence = sequence.squeeze(0)  # Remove batch dimension
        symbolic_sequence = symbolic_sequence.squeeze(0)
        for img, digit in zip(sequence, symbolic_sequence):
            test_images.append(img)
            test_labels.append(digit)

    # Create test dataset and loader
    test_dataset = ImageDataset(test_images, test_labels)
    test_loader_individual = DataLoader(test_dataset, batch_size=32, shuffle=False)

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


