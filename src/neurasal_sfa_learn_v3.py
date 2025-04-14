import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.asal_nesy.neurasal.mnist.compile_multivar_nnf
from src.asal_nesy.neurasal.mnist.compile_multivar_nnf import get_sfa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset, TensorSequence
from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.args_parser import parse_args
from src.logger import logger
from src.asal_nesy.cirquits.asp_programs import mnist_even_odd_learn
from src.asal_nesy.device import device
from src.asal_nesy.neurasal.neurasal_functions import (nesy_forward_pass, get_latent_loss,
                                                       pretrain_nn, StatsCollector, eval_model, nn_forward_pass,
                                                       sequence_to_facts, induce_sfa_simple, set_all_labelled)

nn_args = argparse.Namespace(
    app_name='mnist',
    num_epochs=100,
    active_learning_frequency=5,
    points_to_label=100,
    top_N_seqs=20,
    entropy_scaling_factor=100,
    w_label_density=2.0,
    w_seq_entropy=1.0,
    w_img_entropy=0.0,
    nn_batch_size=50,
    pre_train_nn=True,
    pre_training_size=10,  # num of fully labeled seed sequences.
    pre_train_num_epochs=100,
    learn_seed_sfa_from_pretrained=False,
)

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    logger.info(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    # torch.set_printoptions(threshold=float('inf'))  # print tensors in full
    # torch.set_printoptions(threshold=1000)  # reset.

    cnn_output_size = 10
    model = DigitCNN(out_features=cnn_output_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sequence_criterion = nn.BCELoss()
    nn_criterion = nn.CrossEntropyLoss()

    app_name = nn_args.app_name
    train_data, test_data = get_data('/home/nkatz/dev/asal/data/mnist_nesy/mnist_train.pt',
                                     '/home/nkatz/dev/asal/data/mnist_nesy/mnist_test.pt')

    train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, nn_args.nn_batch_size, train=True)
    test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, nn_args.nn_batch_size, train=False)

    # for s in train_data:
    #     print(s.seq_id)

    class_attrs = ['d1', 'd2', 'd3']

    if nn_args.pre_train_nn:
        logger.info(f'Pre-training on images from {nn_args.pre_training_size} sequences')
        pretrain_nn(train_data, test_data, nn_args.pre_training_size,
                    model, optimizer, class_attrs, nn_args.pre_train_num_epochs)

    logger.info(f'Inducing initial automaton...')
    asp_comp_program = mnist_even_odd_learn

    if nn_args.learn_seed_sfa_from_pretrained and nn_args.pre_train_nn:
        nn_forward_pass(train_loader, model, cnn_output_size)
        data = [sequence_to_facts(seq) for seq in train_data.sequences]

        asal_args = argparse.Namespace(
            tclass=args.tclass,
            batch_size=20,
            test=args.test,
            train=args.train,
            domain=args.domain,
            predicates="equals",
            mcts_iters=10,
            all_opt=True,
            tlim=60,
            states=args.states,
            exp_rate=args.exp_rate,
            mcts_children=args.mcts_children,
            show=args.show,
            unsat_weight=10,  # set this to 0 to have uncertainty weights per sequence
            max_alts=args.max_alts,
            coverage_first=args.coverage_first,
            min_attrs=args.min_attrs,
            warns_off=False,
            revise=False
        )
        data_write_to = f"{project_root}/data/mnist_nesy/train_from_predictions.lp"
        with open(data_write_to, 'w') as f:
            f.write('\n'.join(data))
        asal_args.train = data_write_to
        sfa_dnnf, sfa_asal = induce_sfa_simple(asal_args, asp_comp_program)
    else:
        # sfa_dnnf, sfa_asal = induce_sfa(args, asp_comp_program)
        sfa_dnnf = get_sfa()

    logger.info(f'Starting CNN + Automaton training with interleaved active learning...')
    # sequence_loss_weight = epoch / num_epochs
    seq_loss_weight = 1.0

    for epoch in range(1, nn_args.num_epochs + 1):

        sc = StatsCollector()
        model.train()
        optimizer.zero_grad()

        for batch in train_loader:

            # for debugging
            # set_all_labelled(batch)

            acceptance_probabilities, latent_predictions, nn_outputs = (
                nesy_forward_pass(batch, model, sfa_dnnf, cnn_output_size, update_seqs_stats=True)
            )

            sequence_labels = torch.tensor([seq.seq_label for seq in batch]).to(device)  # (BS,)

            seq_loss = sequence_criterion(acceptance_probabilities, sequence_labels.float())
            latent_loss = get_latent_loss(batch, nn_outputs, nn_criterion, class_attrs)

            total_loss = seq_loss_weight * seq_loss + latent_loss
            # total_loss = seq_loss
            # total_loss = latent_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sequence_predictions = (acceptance_probabilities >= 0.5)

            sc.update_stats(batch, latent_predictions, sequence_predictions,
                            class_attrs, seq_loss.item(), latent_loss.item())

        eval_model(sc, test_loader, model, sfa_dnnf, cnn_output_size, class_attrs, epoch)

        if epoch % nn_args.active_learning_frequency == 0:
            pass
