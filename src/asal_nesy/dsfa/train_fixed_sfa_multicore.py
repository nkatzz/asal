import torch
import torch.nn as nn
from models import NonTrainableNeuralSFA
import os
from utils import (process_batch, reshape_batch, backprop, get_stats, test_model_fixed_sfa)
from mnist_seqs_new import get_data_loaders
from src.asal_nesy.sdds.build_sdds import SDDBuilder
from src.asal_nesy.sdds.asp_programs import mnist_even_odd
import multiprocessing as mp
import functools
import torch.distributed as dist


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def process_sequence_multiproc(data, model_state_dict, model_init_params, criterion):
    sequence, label, symbolic_sequence = data[0], data[1], data[2]
    sdd_builder, num_states, cnn_features = model_init_params[0], model_init_params[1], model_init_params[2]
    model = NonTrainableNeuralSFA(sdd_builder, num_states, cnn_out_features=cnn_features)
    model.load_state_dict(model_state_dict)
    model.train()
    cnn_predictions, guards_predictions, final_states_distribution = model([s.unsqueeze(1) for s in sequence])
    acceptance_probability = final_states_distribution[-1].unsqueeze(0)
    acceptance_probability = torch.clamp(acceptance_probability, 0, 1)
    loss = criterion(acceptance_probability, label.unsqueeze(0).float())

    # print(f'local loss: {loss}')
    # We cannot returned the loss here, since pytorch's autograd does not support
    # sharing tensors with gradient tracking across processes. So instead we compute the
    # gradients with loss.backward() and return those, in order to be averaged and backproped.
    loss.backward()
    grads = {name: param.grad.data.clone() for name, param in model.named_parameters()}
    return loss.item(), grads


def process_batch_multicore(batch, batch_size, model, criterion, num_states):

    init_params = model.sdd_builder, model.num_states, model.cnn_out_features
    sequence_process_function = functools.partial(process_sequence_multiproc,
                                                  model_state_dict=model.state_dict(),
                                                  model_init_params=init_params,
                                                  criterion=criterion)

    seqs, labels, symb_seqs = batch[0], batch[1], batch[2]
    batch_sequences = reshape_batch(batch_size, seqs, labels, symb_seqs)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(sequence_process_function, batch_sequences)
        # result = functools.reduce(combine, p)

    total_loss = 0
    sum_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for loss, grad_dict in results:
        total_loss += loss
        for name, grad in grad_dict.items():
            sum_grads[name] += grad

    return sum_grads, len(results), total_loss


if __name__ == '__main__':

    multicore = True
    mp.set_start_method('spawn')

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

    # if multicore:
    #     model.share_memory()  # This makes the model parameters shared across processes

    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.01
    criterion = nn.BCELoss()
    save_model_to = os.path.join(current_dir, '..', 'saved_models', 'sfa.pth')
    batch_size = 128
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    for epoch in range(num_epochs):
        count = 0
        epoch_loss, actual, predicted = [], [], []

        for batch in train_loader:
            if not multicore:
                batch_loss, a, p = process_batch(batch, batch_size, model, criterion, num_states)
                backprop(batch_loss, optimizer)
                actual.extend(a)
                predicted.extend(p)
                ls = batch_loss.item()
                print(ls)
                epoch_loss.append(ls)
            else:
                sum_grads, length, batch_loss = (
                    process_batch_multicore(batch, batch_size, model, criterion, num_states)
                )

                for name, param in model.named_parameters():
                    param.grad = sum_grads[name] / length

                ls = batch_loss / batch_size  # .item()
                print(ls)
                epoch_loss.append(ls)

        print(f'Epoch {epoch}, Loss: {sum(epoch_loss) / len(epoch_loss)}')

        if not multicore:
            train_f1, tps, fps, fns = get_stats(predicted, actual)
            print(f'Train F1: {train_f1} ({tps}, {fps}, {fns})')
            test_model_fixed_sfa(model, num_states, test_loader)
            # torch.save(model, save_model_to)

    if multicore:
        test_model_fixed_sfa(model, num_states, train_loader, where='Train')
        test_model_fixed_sfa(model, num_states, test_loader)
