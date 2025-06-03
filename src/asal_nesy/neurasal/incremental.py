import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset, TensorSequence
# from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.args_parser import parse_args
from src.logger import *
from src.asal_nesy.cirquits.asp_programs import mnist_even_odd_learn
from src.asal_nesy.device import device
from src.asal_nesy.neurasal.neurasal_functions import (nesy_forward_pass, pretrain_nn, induce_sfa, nesy_train,
                                                       initialize_fully_labeled_seqs, al_random_sampling,
                                                       al_expected_acceptance_loss, StatsCollector,
                                                       compute_expected_acceptance_losses)


class NeSyModel:
    def __init__(self, nn_model, sfa_model_asal, afa_model_dnnf, lr):
        self.nn_model = nn_model
        self.sfa_asal = sfa_model_asal
        self.sfa_dnnf = afa_model_dnnf
        self.optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
        self.lr = lr
        # self.nn_output_size = nn_output_size
        self.seq_loss_weight = 1.0
        self.training_history = {'seq_f1': [], 'img_f1': []}
        self.target_loss = None
        self.latent_loss = None
        self.combined_loss = None

    def update_stats(self, test_stats: dict, train_stats: StatsCollector, train_loader_length):
        self.training_history['seq_f1'].extend(test_stats['seq_f1'])
        self.training_history['img_f1'].extend(test_stats['img_f1'])
        self.target_loss = train_stats.target_loss / train_loader_length
        self.latent_loss = train_stats.latent_loss / train_loader_length
        self.combined_loss = self.target_loss + self.latent_loss


def find_next_sfa(fully_labelled_seqs: list[TensorSequence],
                  train_loader: DataLoader[SequenceDataset],
                  test_loader: DataLoader[SequenceDataset],
                  cnn_output_size,
                  nn_criterion,
                  sequence_criterion,
                  asal_args,
                  asp_comp_program,
                  class_attrs,
                  epochs,
                  current_nesy_model):
    tried_nesy_models = []
    symb_seqs = [s.get_symbolic_seq() for s in fully_labelled_seqs]

    # To revise an existing SFA instead of learning from scratch get it from
    # current_nesy_model.sfa_asal and pass it induce_sfa. Mind the --revise ASAL option in that case.
    automata = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)

    logger.info(f"Induced {len(automata)} optimal SFA. Training each for {epochs} epochs to select the best...")

    for sfa_dnnf, sfa_asal in automata:
        logger.info(f"""Training with\n{sfa_asal.show(mode="simple")}""")

        model_copy = deepcopy(current_nesy_model.nn_model)
        nesy_model = NeSyModel(model_copy, sfa_asal, sfa_dnnf, current_nesy_model.lr)
        tried_nesy_models.append(nesy_model)

        test_stats, train_stats = nesy_train(model_copy, train_loader, sfa_dnnf, cnn_output_size,
                                             nn_criterion, sequence_criterion, nesy_model.optimizer,
                                             epochs, nesy_model.seq_loss_weight, class_attrs, test_loader,
                                             update_seqs_stats=True, show_log=False)

        nesy_model.update_stats(test_stats, train_stats, len(train_loader))

    best_model = min(tried_nesy_models, key=lambda x: x.combined_loss)
    return best_model


def get_labelled_point(train_loader: DataLoader[SequenceDataset],
                       fully_labelled_seqs: list[TensorSequence],
                       current_nesy_model: NeSyModel,
                       random_query=False):
    labeled_ids = set(seq.seq_id for seq in fully_labelled_seqs)
    unlabeled_candidates = [seq for seq in train_data if seq.seq_id not in labeled_ids]
    unlabeled_ids = set(seq.seq_id for seq in unlabeled_candidates)

    if not random_query:
        losses = compute_expected_acceptance_losses(current_nesy_model.nn_model,
                                                    train_loader, current_nesy_model.sfa_dnnf, cnn_output_size)

        filtered_losses = {sid: loss for sid, loss in losses.items() if sid in unlabeled_ids}

        if not filtered_losses:
            print("No more unlabeled sequences to choose from.")
            sys.exit(-1)

        best_seq_id = max(filtered_losses, key=filtered_losses.get)
        best_seq = next(seq for seq in unlabeled_candidates if seq.seq_id == best_seq_id)
    else:
        best_seq = random.choice(unlabeled_candidates)

    return best_seq


def run_experiments(train_data, test_data, N_runs, query_budget, epochs,
                    pretrain_for, lr, num_init_fully_labelled, asp_comp_program,
                    cnn_output_size, class_attrs, asal_args, random_query=False):
    # incr_histories, baseline_histories, active_learn_histories, random_selection_histories = [], [], [], []
    for _ in range(N_runs):
        train_f1_target, train_f1_latent, test_f1_target, test_f1_latent = 0.0, 0.0, 0.0, 0.0
        # Get new loaders for each experiments to make sure the data gets shuffled.
        train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, nn_args.nn_batch_size, train=True)
        test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, nn_args.nn_batch_size, train=False)

        base_model = DigitCNN(out_features=cnn_output_size).to(device)
        optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)  # 0.01
        sequence_criterion = nn.BCELoss()
        nn_criterion = nn.CrossEntropyLoss()

        # Pick an initial set of fully labelled sequences
        fully_labeled_seq_ids = initialize_fully_labeled_seqs(train_loader, num_init_fully_labelled)
        fully_labelled_seqs = list(filter(lambda seq: seq.seq_id in fully_labeled_seq_ids, train_data))
        for seq in fully_labelled_seqs:
            # Mark them all as fully labelled
            seq.mark_seq_as_fully_labelled()

        logger.info(f"Pre-training CNN on available labels for {pretrain_for} epochs...")
        pretrain_nn(SequenceDataset(fully_labelled_seqs), test_data, 0,
                    base_model, optimizer, class_attrs, with_fully_labelled_seqs=True, num_epochs=pretrain_for)

        current_nesy_model = NeSyModel(base_model, None, None, lr)
        logger.info("Inducing initial SFA from fully labelled sequences...")

        current_nesy_model = find_next_sfa(fully_labelled_seqs, train_loader, test_loader,
                                           cnn_output_size, nn_criterion, sequence_criterion,
                                           asal_args, asp_comp_program, class_attrs, epochs, current_nesy_model)

        logger.info(green(f"""Best model:\n{current_nesy_model.sfa_asal.show(mode="simple")}\nLoss: {current_nesy_model.combined_loss} 
        (target: {current_nesy_model.target_loss}), latent: {current_nesy_model.latent_loss}\n
        Test F1: target: {current_nesy_model.training_history['seq_f1'][-1]}, latent: {current_nesy_model.training_history['img_f1'][-1]}"""))

        # Fall back to this every time a different method is tried in the experiments.
        init_nesy_model = current_nesy_model
        current_loss = current_nesy_model.combined_loss

        logger.info(yellow(f'Fully labelled: {[s.seq_id for s in fully_labelled_seqs]}'))

        msg = (
            f"Active learning with EAL for {query_budget} queries..."
            if not random_query
            else f"Active learning with random sampling for {query_budget} queries..."
        )

        logger.info(green(msg))

        for query in range(query_budget):
            best_seq = get_labelled_point(train_loader, fully_labelled_seqs,
                                          current_nesy_model, random_query=random_query)
            best_seq.mark_seq_as_fully_labelled()
            fully_labelled_seqs.append(best_seq)
            logger.info(yellow(f'Selected sequence: {best_seq.seq_id}'))
            logger.info(yellow(f'Fully labelled: {[s.seq_id for s in fully_labelled_seqs]}'))
            new_nesy_model = find_next_sfa(fully_labelled_seqs, train_loader, test_loader,
                                           cnn_output_size, nn_criterion, sequence_criterion,
                                           asal_args, asp_comp_program, class_attrs, epochs, current_nesy_model)

            if new_nesy_model.combined_loss < current_loss:
                current_nesy_model = new_nesy_model
                current_loss = current_nesy_model.combined_loss

            logger.info(green(
                f"""Best model:\n{current_nesy_model.sfa_asal.show(mode="simple")}\nLoss: {current_nesy_model.combined_loss} (target: {current_nesy_model.target_loss}), latent: {current_nesy_model.latent_loss}\nTest F1: target: {current_nesy_model.training_history['seq_f1'][-1]}, latent: {current_nesy_model.training_history['img_f1'][-1]}"""))

        # Train for a few more epochs in the end
        # nesy_train(model, train_loader, sfa_dnnf, cnn_output_size,
        #            nn_criterion, sequence_criterion, optimizer,
        #            num_epochs, seq_loss_weight, class_attrs, test_loader, show_log=show_stats)



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    asal_args = argparse.Namespace(tclass=args.tclass, batch_size=20000, test=args.test, train=args.train,
                                   domain=args.domain,
                                   predicates="equals", mcts_iters=10, all_opt=True,  # Get multiple optimal models!
                                   tlim=60, states=args.states,
                                   exp_rate=args.exp_rate, mcts_children=args.mcts_children, show=args.show,
                                   unsat_weight=10,  # Set this to 0 to have uncertainty weights per sequence
                                   max_alts=args.max_alts, coverage_first=args.coverage_first, min_attrs=args.min_attrs,
                                   warns_off=False, revise=False)

    nn_args = argparse.Namespace(app_name='mnist', num_epochs=100, active_learning_frequency=5, points_to_label=100,
                                 top_N_seqs=20, entropy_scaling_factor=100, w_label_density=2.0, w_seq_entropy=1.0,
                                 w_img_entropy=0.0, nn_batch_size=50, pre_train_nn=True,
                                 pre_training_size=10,  # num of fully labeled seed sequences.
                                 pre_train_num_epochs=100, learn_seed_sfa_from_pretrained=False, )

    num_init_fully_labelled = 4  # Number of initial fully labelled sequences
    num_queries = 10  # Total number of active learning queries
    num_epochs = 20  # Number of epochs to train after each active learning update
    cnn_output_size = 10  # for MNIST
    pre_train_for = 100
    nn_batch_size = 50
    lr = 0.01  # 0.01 initially
    N_samples = 5  # Number of samples/unlabeled sequence for ELR
    # class_attrs = ['d1', 'd2', 'd3']
    class_attrs = ['d1']
    asp_comp_program = mnist_even_odd_learn
    num_runs = 1  # Number of experiments to run

    train_data, test_data = get_data('/home/nkatz/dev/asal/data/mnist_nesy/mnist_train.pt',
                                     '/home/nkatz/dev/asal/data/mnist_nesy/mnist_test.pt')

    run_experiments(train_data, test_data, num_runs, num_queries, num_epochs, pre_train_for,
                    lr, num_init_fully_labelled, asp_comp_program, cnn_output_size, class_attrs, asal_args)
