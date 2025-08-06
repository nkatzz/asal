import torch
import torch.nn as nn
from models import NeuralSFA, softmax_with_temperature
import os
from utils import (get_stats, test_model, set_correct_model)
from mnist_seqs_new import get_data_loaders
from src.globals import device

if not torch.cuda.is_available():
    torch.set_num_threads(1)

if __name__ == "__main__":

    input_domain = list(range(0, 10))
    # input_domain = list(range(0, 6))
    guards = [[8], [4, 6], [2, 0], [7, 9], [5], [1, 3]]

    with_prior_model = False  # if True a fixed automaton is used, set by the utils.set_correct_model method.
    show_transitions = False
    nesy_mode = True  # if False the input consists of the symbolic (label) sequences, so it's not a NeSy setting
    num_states, num_guards = 50, 6  # states are originally 4
    num_epochs = 100
    softmax_temp = 0.1  # 0.01
    temperature_discount = 0.99
    discount_temp = False

    model = NeuralSFA(
        num_states,
        num_guards,
        nesy_mode=nesy_mode,
        cnn_out_features=len(input_domain),
        with_prior_model=with_prior_model
    )

    model = model.to(device)

    if with_prior_model:
        set_correct_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 0.001
    criterion = nn.BCELoss()

    current_dir = os.path.dirname(__file__)
    save_model_to = os.path.join(current_dir, '..', 'saved_models', 'sfa.pth')

    train_loader, test_loader = get_data_loaders()

    batch_size = 10

    for epoch in range(num_epochs):
        if discount_temp:
            softmax_temp = softmax_temp * temperature_discount
        losses = []
        count = 0
        batch_loss, batch_generalization_loss = torch.zeros(1).to(device), torch.zeros(1).to(device)
        actual, predicted = [], []
        accum_cnn_predictions = torch.zeros(1, len(input_domain))
        for sequence, label, symbolic_sequence in train_loader:
            sequence = [tensor.to(device) for tensor in sequence]
            label = label.to(device)
            # symbolic_sequence = torch.tensor(symbolic_sequence).to(device)
            symbolic_sequence = symbolic_sequence.to(device)

            if batch_size == 1: optimizer.zero_grad()

            if nesy_mode:
                cnn_predictions, guards_predictions, final_states_distribution = model(sequence, softmax_temp)
            else:
                cnn_predictions, guards_predictions, final_states_distribution = model(symbolic_sequence, softmax_temp)

            # if nesy_mode:
            #    cnn_predictions = torch.stack([t.squeeze() for t in cnn_predictions])
            #    accum_cnn_predictions = torch.cat((accum_cnn_predictions, cnn_predictions), dim=0)

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

                optimizer.step()
                optimizer.zero_grad()

                batch_loss = torch.zeros(1).to(device)
                accum_cnn_predictions = torch.zeros(1, len(input_domain))

            losses.append(loss)
            count += 1

        print(f'Epoch {epoch}, Loss: {sum(losses) / len(losses)}')
        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)
        print(f'Train F1: {train_f1} ({tps}, {fps}, {fns})')
        test_model(model, num_states, test_loader, softmax_temp=softmax_temp, nesy_mode=nesy_mode)
        # torch.save(model, save_model_to)

        if show_transitions:
            for i, matrix in enumerate(model.transition_matrices):
                print(f'Transition Matrix for Rule {i}:\n{softmax_with_temperature(matrix, temperature=softmax_temp)}')
            # see_parameters(model)
