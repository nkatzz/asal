import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models import NonTrainableNeuralSFA # NonTrainableNeuralSFA_roadR
import os
from utils import (backprop, get_stats, test_model_fixed_sfa, process_sequences)
from mnist_seqs_new import get_data_loaders
from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.asp_programs import mnist_even_odd
import time
import logging
from src.asal_nesy.device import device

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s')
# Set log file, its level and format
file_handler = logging.FileHandler('./training_logger.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Set stream its level and format
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

print(device)


if __name__ == '__main__':

    current_dir = os.path.dirname(__file__)
    save_models_path = os.path.join(current_dir, '..', 'saved_models')
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)

    print('Building SDDs...')
    asp_program = mnist_even_odd
    sdd_builder = SDDBuilder(asp_program,
                             vars_names=['d'],
                             categorical_vars=['d'])
    sdd_builder.build_sdds()
    print(f"""SDDs:\n{sdd_builder.circuits if
    sdd_builder.circuits is not None else 'Discarded, using the polynomials'}""")

    input_domain = list(range(0, 10))  # the length of the NN output vector, for MNIST, 0..9
    num_states = 4
    num_epochs = 100

    model = NonTrainableNeuralSFA(sdd_builder, num_states, cnn_out_features=len(input_domain))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.1
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

    # Define a learning rate scheduler
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decrease LR by 10% every 5 epochs

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 0.001
    criterion = nn.BCELoss()
    save_model_to = os.path.join(current_dir, '..', 'saved_models', 'sfa.pth')
    batch_size = 1  # 128
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    for epoch in range(num_epochs):
        logger.info(f"\nStart training epoch {epoch+1}")
        actual, predicted = [], []

        total_loss = 0.0

        start_time = time.time()
        for batch in train_loader:
            loss, a, p = process_sequences(batch, model, criterion, num_states)
            backprop(loss, optimizer)
            actual.extend(a)
            predicted.extend(p)
            total_loss += loss.item()

        logger.info(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.3f}, Time: {(time.time()-start_time)/60:.2f} mins')

        # train_f1, tps, fps, fns = get_stats(predicted, actual)
        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        logger.info(f'Train F1: {train_f1} ({tps}, {fps}, {fns})')
        test_model_fixed_sfa(model, num_states, test_loader)

        # Update the learning rate scheduler
        # scheduler.step()
        # torch.save(model, save_model_to)
