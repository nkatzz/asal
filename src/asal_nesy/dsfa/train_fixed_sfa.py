import torch
import torch.nn as nn
from models import NonTrainableNeuralSFA
import os
from utils import (get_stats, test_model_fixed_sfa)
from mnist_seqs_new import get_data_loaders
from src.asal_nesy.sdds.build_sdds import SDDBuilder
from src.asal_nesy.sdds.asp_programs import mnist_even_odd
import tracemalloc


if not torch.cuda.is_available():
    torch.set_num_threads(1)  # usually faster than using multiple threads in the CPU case

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

current_dir = os.path.dirname(__file__)
save_models_path = os.path.join(current_dir, '..', 'saved_models')
if not os.path.exists(save_models_path):
    os.makedirs(save_models_path)


if __name__ == '__main__':
    print('Building SDDs...')
    asp_program = mnist_even_odd
    sdd_builder = SDDBuilder(asp_program,
                             vars_names=['d'],
                             categorical_vars=['d'])
    sdd_builder.build_sdds()
    print(f'SDDs:\n{sdd_builder.circuits}')

    input_domain = list(range(0, 10))  # the length of the NN output vector, for MNIST, 0..9
    num_states = 4
    num_epochs = 100

    model = NonTrainableNeuralSFA(sdd_builder, num_states, cnn_out_features=len(input_domain))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.01
    criterion = nn.BCELoss()
    save_model_to = os.path.join(current_dir, '..', 'saved_models', 'sfa.pth')
    train_loader, test_loader = get_data_loaders()
    batch_size = 128

    # tracemalloc.start()

    for epoch in range(num_epochs):
        losses = []
        count = 0
        batch_loss = torch.zeros(1).to(device)
        actual, predicted = [], []

        for sequence, label, symbolic_sequence in train_loader:
            sequence = [tensor.to(device) for tensor in sequence]
            label = label.to(device)
            symbolic_sequence = torch.tensor(symbolic_sequence).to(device)

            if batch_size == 1: optimizer.zero_grad()

            cnn_predictions, guards_predictions, final_states_distribution = model(sequence)

            acceptance_probability = final_states_distribution[-1].unsqueeze(0)
            acceptance_probability = torch.clamp(acceptance_probability, 0, 1)
            loss = criterion(acceptance_probability, label.float())
            batch_loss += loss

            # Collect stats for training F1
            predicted_state = torch.argmax(final_states_distribution).item()
            prediction = 1 if predicted_state == num_states - 1 else 0
            actual.append(label.item())
            predicted.append(prediction)

            if batch_size > 1 and count % batch_size == 0:
                total_loss = batch_loss / batch_size
                
                total_loss.backward()

                # for name, param in model.named_parameters():
                #    if param.requires_grad:
                #        print(f"Gradient of {name} is {param.grad}")

                optimizer.step()
                optimizer.zero_grad()
                print(total_loss.item())

                # current, peak = tracemalloc.get_traced_memory()
                # print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

                batch_loss = torch.zeros(1).to(device)
            losses.append(loss)
            count += 1

        print(f'Epoch {epoch}, Loss: {sum(losses) / len(losses)}')
        train_f1, tps, fps, fns = get_stats(predicted, actual)
        print(f'Train F1: {train_f1} ({tps}, {fps}, {fns})')
        test_model_fixed_sfa(model, num_states, test_loader)
        # torch.save(model, save_model_to)

