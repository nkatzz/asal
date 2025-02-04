import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Load MNIST dataset
train_images = MNIST(root="./mnist_data", train=True, download=True, transform=ToTensor())
test_images = MNIST(root="./mnist_data", train=False, download=True, transform=ToTensor())

# Select 100 images for training
train_subset = torch.utils.data.Subset(train_images, range(100))
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

# Use the remaining images for evaluation
rest_train_loader = DataLoader(
    torch.utils.data.Subset(train_images, range(100, len(train_images))),
    batch_size=32,
    shuffle=False,
)
test_loader = DataLoader(test_images, batch_size=32, shuffle=False)


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    print(f'epoch: {epoch}')
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


# Evaluate the model
def evaluate_model(loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predictions = outputs.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    return f1_score(y_true, y_pred, average="weighted")


# Compute F1-scores
rest_train_f1 = evaluate_model(rest_train_loader)
test_f1 = evaluate_model(test_loader)

print("F1-score on rest of training set:", rest_train_f1)
print("F1-score on test set:", test_f1)
