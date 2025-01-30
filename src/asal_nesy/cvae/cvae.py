from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Parameters
input_dim = 784  # For MNIST
latent_dim = 2
num_classes = 10
batch_size = 128
num_epochs = 50
learning_rate = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Load MNIST test dataset
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Split the test dataset into validation (unseen) and remaining test data
unseen_indices = list(range(5000))  # First 5000 samples for unseen data
remaining_indices = list(range(5000, 10000))  # Remaining samples for actual test data

unseen_dataset = Subset(test_dataset, unseen_indices)
remaining_test_dataset = Subset(test_dataset, remaining_indices)

# Create DataLoader for unseen data
x_unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

# Extract labels for unseen data
y_unseen_labels = torch.tensor([test_dataset.targets[i] for i in unseen_indices])


# Define CVAE components
class Encoder(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + num_classes, 256)
        self.fc2_mean = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        h = F.relu(self.fc1(x))
        z_mean = self.fc2_mean(h)
        z_logvar = self.fc2_logvar(h)
        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + num_classes, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, z, c):
        z = torch.cat((z, c), dim=1)
        h = F.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))
        return x_reconstructed


class CVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, num_classes, latent_dim)
        self.decoder = Decoder(latent_dim, num_classes, input_dim)

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x, c):
        z_mean, z_logvar = self.encoder(x, c)
        z = self.reparameterize(z_mean, z_logvar)
        x_reconstructed = self.decoder(z, c)
        return x_reconstructed, z_mean, z_logvar


# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model and optimizer
cvae = CVAE(input_dim, num_classes, latent_dim)
optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)


def loss_function(reconstructed_x, x, z_mean, z_logvar):
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return BCE + KLD


# Training loop
for epoch in range(num_epochs):
    cvae.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.view(-1, input_dim)
        labels_onehot = F.one_hot(labels, num_classes).float()
        optimizer.zero_grad()
        reconstructed_x, z_mean, z_logvar = cvae(data, labels_onehot)
        loss = loss_function(reconstructed_x, data, z_mean, z_logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")

# Load MNIST test dataset for unseen data
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Split the test dataset into validation (unseen) and remaining test data
unseen_indices = list(range(5000))  # First 5000 samples for unseen data
remaining_indices = list(range(5000, 10000))  # Remaining samples for actual test data

unseen_dataset = Subset(test_dataset, unseen_indices)
remaining_test_dataset = Subset(test_dataset, remaining_indices)

# Create DataLoader for unseen data
x_unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

# Extract labels for unseen data
y_unseen_labels = torch.tensor([test_dataset.targets[i] for i in unseen_indices])


def get_latent_representations(encoder, data_loader):
    encoder.eval()
    latent_representations = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(-1, input_dim)
            z_mean, z_logvar = encoder(data, torch.zeros(data.size(0), num_classes))
            z = cvae.reparameterize(z_mean, z_logvar)
            latent_representations.append(z)
    return torch.cat(latent_representations, dim=0)


def get_latent_representations_with_label(encoder, data_loader, num_classes):
    encoder.eval()
    latent_representations = []
    labels_list = []
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.view(-1, input_dim)
            labels_onehot = F.one_hot(labels, num_classes).float()
            z_mean, z_logvar = encoder(data, labels_onehot)
            z = cvae.reparameterize(z_mean, z_logvar)
            latent_representations.append(z)
            labels_list.append(labels)
    return torch.cat(latent_representations, dim=0), torch.cat(labels_list, dim=0)


# Get latent representations for training data
# latent_train = get_latent_representations(cvae.encoder, train_loader)
latent_train, labels_train = get_latent_representations_with_label(cvae.encoder, train_loader, num_classes)

# Get latent representations for unseen data
# latent_unseen = get_latent_representations(cvae.encoder, x_unseen_loader)
latent_unseen, labels_unseen = get_latent_representations_with_label(cvae.encoder, x_unseen_loader, num_classes)

# Convert latent representations to numpy arrays
latent_train_np = latent_train.cpu().numpy()
labels_np = train_dataset.targets.numpy()
latent_unseen_np = latent_unseen.cpu().numpy()
y_unseen_labels_np = y_unseen_labels.cpu().numpy()

# Train the classifier
# classifier = RandomForestClassifier()
# classifier.fit(latent_train_np, labels_np)

# Train the XGBoost classifier
classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
classifier.fit(latent_train_np, labels_np)

# Predict classes for unseen data based on latent representations
predicted_classes = classifier.predict(latent_unseen_np)

# Evaluate the performance
accuracy = accuracy_score(y_unseen_labels_np, predicted_classes)
conf_matrix = confusion_matrix(y_unseen_labels_np, predicted_classes)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")


# Visualize latent space
def visualize_latent_space(latent_representations, labels, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()


visualize_latent_space(latent_train_np, labels_np, 'Latent Space of Training Data')
visualize_latent_space(latent_unseen_np, y_unseen_labels_np, 'Latent Space of Unseen Data')


# Define a function to sample from the latent space with class conditioning
def sample_and_reconstruct(cvae, class_label, num_samples=1):
    cvae.eval()
    with torch.no_grad():
        # Sample from the standard normal distribution
        z = torch.randn(num_samples, latent_dim)
        # Create one-hot encoded class label
        class_onehot = F.one_hot(torch.tensor([class_label] * num_samples), num_classes).float()
        # Reconstruct the image from the sampled latent vector
        reconstructed_x = cvae.decoder(z, class_onehot)
        return reconstructed_x.view(-1, 28, 28)


# Visualize the reconstructed image
def visualize_reconstructed_images(images, title):
    plt.figure(figsize=(10, 2))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


# Sample and reconstruct images for each class
for class_label in range(num_classes):
    sampled_images = sample_and_reconstruct(cvae, class_label, num_samples=5)
    visualize_reconstructed_images(sampled_images.cpu(), f'Reconstructed Images for Class {class_label}')
