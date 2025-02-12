import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.dsfa_old.mnist_seqs_new import get_data_loaders
from src.asal.logger import *
from src.asal_nesy.neurasal.pre_train_model import pre_train
from src.asal_nesy.pre_train_cnn import SimpleCNN


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)
# print(sys.path[0])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device: {device}')

if __name__ == "__main__":
    # Learn an SFA from some initial fully labeled sequences
    """
    asal_train_path = f'{project_root}/data/mnist_nesy/train.csv'
    max_states = 4
    target_class = 1
    sfa = induce_sfa(asal_train_path, max_states, target_class, time_lim=30)
    """
    # Neural stuff follow
    pre_training_size = 10  # num of fully labeled seed sequences.
    num_epochs = 100
    batch_size = 1
    cnn_classes = 10  # digits num. for MNIST
    model = DigitCNN(out_features=cnn_classes)
    # model = SimpleCNN()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.1
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    logger.info(f'Pre-training on images from {pre_training_size} sequences')

    pre_train(train_loader, test_loader, 10, model, optimizer)

    logger.info(f'Training with the SFA...')

    # for epoch in range(num_epochs):

