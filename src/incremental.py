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
from src.asal_nesy.globals import device, sample_symb_seqs, num_top_k
from src.asal.structs import Automaton
from src.asal.tester import Tester
from src.asal_nesy.neurasal.neurasal_functions import (nesy_forward_pass, pretrain_nn, induce_sfa, nesy_train,
                                                       initialize_fully_labeled_seqs, al_random_sampling,
                                                       al_expected_acceptance_loss, StatsCollector,
                                                       compute_expected_acceptance_losses, compute_seq_probs,
                                                       set_asp_weights, get_sequence_stats, get_nn_predicted_seqs,
                                                       timer)


# Doesn't seem to improve things, but need to try it on a computer with actually many cores...
# if not torch.cuda.is_available():
#     torch.set_num_threads(1)

class NeSyModel:
    def __init__(self, nn_model, sfa_model_asal, sfa_model_dnnf, lr):
        self.nn_model = nn_model
        self.sfa_asal = sfa_model_asal
        self.sfa_dnnf = sfa_model_dnnf
        self.optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
        self.lr = lr
        # self.nn_output_size = nn_output_size
        self.seq_loss_weight = 1.0
        self.training_history = {'seq_f1': [], 'img_f1': []}
        self.target_loss = None
        self.latent_loss = None
        self.combined_loss = None
        self.train_f1 = None
        self.latent_f1 = None

    @staticmethod
    def get_score(ts: StatsCollector):
        from sklearn.metrics import f1_score
        latent_f1_macro = f1_score(ts.latent_actual, ts.latent_predicted, average="macro")
        f1, tps, fps, fns = get_sequence_stats(ts.seq_predicted, ts.seq_actual)
        return f1, tps, fps, fns, latent_f1_macro

    def update_stats(self, test_stats: dict, train_stats: StatsCollector, train_loader_length):
        self.training_history['seq_f1'].extend(test_stats['seq_f1'])
        self.training_history['img_f1'].extend(test_stats['img_f1'])
        self.target_loss = train_stats.target_loss / train_loader_length
        self.latent_loss = train_stats.latent_loss / train_loader_length
        self.combined_loss = self.target_loss + self.latent_loss

        train_f1, train_tps, train_fps, train_fns, train_latent_f1 = self.get_score(train_stats)
        self.train_f1 = train_f1
        self.latent_f1 = train_latent_f1


@timer
def train(args):
    sfa_asal, sfa_dnnf, current_nesy_model, train_loader, test_loader, cnn_output_size, \
        lr, nn_criterion, sequence_criterion, epochs, seq_loss_weight, class_attrs = args

    logger.info(f"""Training with\n{sfa_asal.show(mode="simple")}""")

    model_copy = deepcopy(current_nesy_model.nn_model)
    nesy_model = NeSyModel(model_copy, sfa_asal, sfa_dnnf, lr)

    test_stats, train_stats = nesy_train(model_copy, train_loader, sfa_dnnf, cnn_output_size,
                                         nn_criterion, sequence_criterion, nesy_model.optimizer,
                                         epochs, seq_loss_weight, class_attrs, test_loader,
                                         update_seqs_stats=True, show_log=False)

    nesy_model.update_stats(test_stats, train_stats, len(train_loader))
    logger.info(yellow(f'Loss: {nesy_model.combined_loss:.3f}'))
    return nesy_model


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
                  current_nesy_model,
                  revise_existing=True,
                  parallel_training=False,
                  use_partially_labeled_seqs=False):
    tried_nesy_models = []
    if not use_partially_labeled_seqs:  # use only the fully labelled sequences in this case.
        symb_seqs = [s.get_labelled_seq_asp() for s in fully_labelled_seqs]
    else:
        # Here we need to:
        # 1. calculate the sequence probabilities for the unlabeled sequences,
        # 2. compute weights for them,
        # 3. compute weights for the fully labelled seqs,
        # 4. change the asal args to allow for different weight per sequence and
        # 5. pass the symbolic sequences to ASAL.
        # It makes sense to disable the all_opt option in this case (just get one SFA).

        # 1. calculate the sequence probabilities for the unlabeled sequences
        for batch in train_loader:
            seq_probs = compute_seq_probs(current_nesy_model.nn_model, batch)
            predicted_seqs, softmx_preds = get_nn_predicted_seqs(batch, current_nesy_model.nn_model, cnn_output_size)
            for seq, prob, pred_seq, softmx_pred in zip(batch, seq_probs, predicted_seqs, softmx_preds):
                seq.sequence_probability = prob.item()
                seq.predicted_symbolic_seq = pred_seq
                seq.predicted_softmaxed_seq = softmx_pred

        #===============================================================================================================
        # Try to implement the abductive query point selection here, factor it out into a method later,
        # provide options ofr which version we are using (labelling entire sequences, or individual points across seqs).
        all_seqs = [seq for batch in train_loader for seq in batch]

        # First, get the misclassified - given the current SFA - sequences
        misclassified_seqs = [
            seq for seq in all_seqs
            if (seq.seq_label == 1 & seq.acceptance_probability < 0.5)
            or (seq.seq_label == 0 & seq.acceptance_probability >= 0.5)
        ]

        for seq in misclassified_seqs:
            asp_atoms = []
            labels_grouped = [list(group) for group in zip(*seq.image_labels)]
            attributes = sorted({k for g in labels_grouped for d in g for k in d})
            for t in range(seq.seq_length):
                for d in range(seq.dimensionality):
                    distribution = seq.predicted_softmaxed_seq[t][d]
                    for i, prob in enumerate(distribution):
                        atom = f'pred({seq.seq_id},{attributes[d],})'

        # ==============================================================================================================

        # 2. Select the K-most probable sequences.
        all_unlabelled = [s for batch in train_loader for s in batch if s not in fully_labelled_seqs]
        k = num_top_k
        top_k = sorted(all_unlabelled, key=lambda s: s.sequence_probability, reverse=True)[:k]
        set_asp_weights(top_k, fully_labelled_seqs, max_unlabelled_weight=100)
        fully_labelled_symb_seqs = [s.get_labelled_seq_asp(with_custom_weights=True) for s in fully_labelled_seqs]
        predicted_symb_seqs = [s.get_predicted_seq_asp(with_custom_weights=True) for s in top_k]
        symb_seqs = fully_labelled_symb_seqs + predicted_symb_seqs

        asal_args.unsat_weight = 0  # Set this to allow ASAL to optimize the per-sequence weights.

    # To revise an existing SFA instead of learning from scratch get it from
    # current_nesy_model.sfa_asal and pass it induce_sfa. Mind the --revise ASAL option in that case.
    if revise_existing:
        automata = induce_sfa(symb_seqs, asal_args, asp_comp_program,
                              class_attrs, existing_sfa=current_nesy_model.sfa_asal)
    else:
        automata = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)

    logger.info(f"Induced {len(automata)} optimal SFA. Training each for {epochs} epochs to select the best...")

    args_list = [
        (sfa_asal, sfa_dnnf, current_nesy_model, train_loader, test_loader,
         cnn_output_size, current_nesy_model.lr, nn_criterion, sequence_criterion,
         epochs, current_nesy_model.seq_loss_weight, class_attrs)
        for (sfa_dnnf, sfa_asal) in automata
    ]

    if not parallel_training:
        for args in args_list:
            nesy_model = train(args)
            tried_nesy_models.append(nesy_model)
        best_model = min(tried_nesy_models, key=lambda x: x.combined_loss)
        return best_model

    else:
        # Does not work on GPU. It is extremely slow on CPU... so, a bad idea overall.
        from multiprocessing import Pool
        import torch.multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
            # with Pool(processes=min(mp.cpu_count(), len(automata))) as pool:
            trained_models = pool.map(train, args_list)
            best_model = min(trained_models, key=lambda m: m.combined_loss)
            return best_model


def get_labelled_seq(train_loader: DataLoader[SequenceDataset],
                     fully_labelled_seqs: list[TensorSequence],
                     current_nesy_model: NeSyModel,
                     random_query=False):
    labeled_ids = set(seq.seq_id for seq in fully_labelled_seqs)
    unlabeled_candidates = [seq for seq in train_data if seq.seq_id not in labeled_ids]
    unlabeled_ids = set(seq.seq_id for seq in unlabeled_candidates)

    is_misclassified = (
        lambda seq: (seq.seq_label == 1 and seq.acceptance_probability < 0.5) or
                    (seq.seq_label == 0 and seq.acceptance_probability >= 0.5)
    )

    if not random_query:
        losses = compute_expected_acceptance_losses(current_nesy_model.nn_model,
                                                    train_loader, current_nesy_model.sfa_dnnf, cnn_output_size)

        filtered_losses = {sid: loss for sid, loss in losses.items() if sid in unlabeled_ids}

        if not filtered_losses:
            print("No more unlabeled sequences to choose from.")
            sys.exit(-1)

        """
        sorted_losses = dict(sorted(filtered_losses.items(), key=lambda x: x[1], reverse=True))
        for sid, loss in sorted_losses.items():
            print(f"{sid}: {loss}")
        """

        best_seq_id = max(filtered_losses, key=filtered_losses.get)
        best_seq = next(seq for seq in unlabeled_candidates if seq.seq_id == best_seq_id)

        best_seq_unsat_samples = 0
        if sample_symb_seqs:
            sorted_losses = dict(sorted(filtered_losses.items(), key=lambda x: x[1], reverse=True))
            num_samples = 10
            current_sfa = current_nesy_model.sfa_asal.show(mode="reasoning")

            for sid, loss in sorted_losses.items():

                num_unsatisfied_samples = 0
                seq = next(seq for seq in unlabeled_candidates if seq.seq_id == sid)

                if is_misclassified(seq):  # don't bother checking for seqs that are already OK.
                    for i in range(num_samples):  # num of samples
                        symb_seq = seq.sample_symbolic_sequence(current_nesy_model.nn_model)
                        tester = Tester(symb_seq, args, current_sfa)
                        tester.test_model()
                        if len(tester.fp_seq_ids) == 1 | len(tester.fn_seq_ids) == 1:
                            num_unsatisfied_samples += 1
                    # print(f'seq {sid}, loss: {loss}, num of unsat samples: {num_unsatisfied_samples}')
                    if num_unsatisfied_samples / num_samples >= 0.8:
                        best_seq_id = sid
                        best_seq_unsat_samples = num_unsatisfied_samples
                        best_seq = seq
                        break

        if is_misclassified(best_seq):
            extra_msg = f'| unsat samples: {best_seq_unsat_samples}' if sample_symb_seqs else ''

            logger.info(yellow(f'Selected: {best_seq.seq_id} | Acceptance probability: '
                               f'{best_seq.acceptance_probability:.3f} | Label: {best_seq.seq_label} | '
                               f'BCE loss: {filtered_losses[best_seq.seq_id]:.3f} {extra_msg}'))
        else:
            logger.info('No incorrectly classified sequences!')

        """
        # Debugging to check if the sampled symbolic seqs from the CNN yields changes to the SFA,
        # despite the fact the actual selected seq-to-label does not, i.e. it is not informative...
        # This code induces a new SFA from the already fully labelled seqs plus the new one to see
        # if the SFA differs from the previous one.
        symb_seqs = [s.get_symbolic_seq() for s in fully_labelled_seqs]
        for i in range(10):
            sample_symb_seq = best_seq.sample_symbolic_sequence(current_nesy_model.nn_model)
            fls = symb_seqs + [sample_symb_seq]
            automata = induce_sfa(fls, asal_args, asp_comp_program, class_attrs)
            logger.info(f'From SAMPLE {i}:')
            for a, b in automata:
                print(f"{b.show(mode='simple')}")
        """
    else:
        best_seq = random.choice(unlabeled_candidates)
        logger.info(yellow(f'Selected: {best_seq.seq_id} | Acceptance probability: '
                           f'{best_seq.acceptance_probability} | Label: {best_seq.seq_label}'))

    best_seq = best_seq if is_misclassified(best_seq) else None

    return best_seq


def run_experiments(train_data, test_data, N_runs, query_budget, epochs,
                    pretrain_for, lr, num_init_fully_labelled, asp_comp_program,
                    cnn_output_size, class_attrs, asal_args, random_query=False, use_partially_labeled=True):
    # incr_histories, baseline_histories, active_learn_histories, random_selection_histories = [], [], [], []
    for exp_num in range(N_runs):
        logger.info(f'\n\nSTARTING EXPERIMENT {exp_num}\n')
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
        # fully_labeled_seq_ids = [714, 515, 1151, 2002]
        # fully_labeled_seq_ids = [823, 1748, 1021, 276]  # This is a good seed
        # fully_labeled_seq_ids = [433, 746, 1590, 1344]  # This is a hard seed, no convergence even with 20 queries

        logger.info(yellow(f'Fully labelled: {[s for s in fully_labeled_seq_ids]}'))

        fully_labelled_seqs = list(filter(lambda seq: seq.seq_id in fully_labeled_seq_ids, train_data))
        for seq in fully_labelled_seqs:
            # Mark them all as fully labelled
            seq.mark_seq_as_fully_labelled()

        logger.info(f"Pre-training CNN on available labels for {pretrain_for} epochs...")
        pretrain_nn(SequenceDataset(fully_labelled_seqs), test_data, 0,
                    base_model, optimizer, class_attrs, with_fully_labelled_seqs=True, num_epochs=pretrain_for)

        current_nesy_model = NeSyModel(base_model, Automaton(), None, lr)

        logger.info("Inducing initial SFA from fully labelled sequences...")

        # Call this with use_partially_labeled_seqs=True to use all the data in the induction of the initial SFA.
        current_nesy_model = find_next_sfa(fully_labelled_seqs, train_loader, test_loader,
                                           cnn_output_size, nn_criterion, sequence_criterion,
                                           asal_args, asp_comp_program, class_attrs, epochs, current_nesy_model,
                                           use_partially_labeled_seqs=use_partially_labeled)

        show_log_msg(current_nesy_model)

        # Fall back to this every time a different method is tried in the experiments.
        init_nesy_model = current_nesy_model
        current_loss = current_nesy_model.combined_loss
        current_f1 = current_nesy_model.train_f1

        logger.info(yellow(f'Fully labelled: {[s.seq_id for s in fully_labelled_seqs]}'))

        msg = (
            f"Active learning with EAL for {query_budget} queries..."
            if not random_query
            else f"Active learning with random sampling for {query_budget} queries..."
        )

        logger.info(green(msg))

        # Active learning step
        for query in range(query_budget):
            best_seq = get_labelled_seq(train_loader, fully_labelled_seqs,
                                        current_nesy_model, random_query=random_query)

            if best_seq is not None:

                best_seq.mark_seq_as_fully_labelled()
                fully_labelled_seqs.append(best_seq)
                logger.info(yellow(f'Selected sequence: {best_seq.seq_id}'))
                logger.info(yellow(f'Fully labelled: {[s.seq_id for s in fully_labelled_seqs]}'))

                new_nesy_model = find_next_sfa(fully_labelled_seqs, train_loader, test_loader,
                                               cnn_output_size, nn_criterion, sequence_criterion,
                                               asal_args, asp_comp_program, class_attrs, epochs,
                                               current_nesy_model, use_partially_labeled_seqs=use_partially_labeled)

                logger.info(yellow(f'min/current losses: {new_nesy_model.combined_loss}/{current_loss}'))
                logger.info(yellow(f'min/current train F1s: {new_nesy_model.train_f1}/{current_f1}'))

                # if new_nesy_model.combined_loss < current_loss:
                #     current_nesy_model = new_nesy_model
                #     current_loss = current_nesy_model.combined_loss

                # if new_nesy_model.train_f1 > current_f1:
                if True:
                    current_nesy_model = new_nesy_model
                    current_loss = current_nesy_model.combined_loss
                    current_f1 = new_nesy_model.train_f1
                """
                else:
                    # Train a bit more to change the landscape of sat/unsat seqs.
                    logger.info(yellow(f'Current model has not changed, training for additional {epochs} epochs...'))
                    test_stats, train_stats = nesy_train(current_nesy_model.nn_model, train_loader,
                                                         current_nesy_model.sfa_dnnf, cnn_output_size,
                                                         nn_criterion, sequence_criterion, current_nesy_model.optimizer,
                                                         epochs, current_nesy_model.seq_loss_weight, class_attrs,
                                                         test_loader, update_seqs_stats=True, show_log=False)
    
                    current_nesy_model.update_stats(test_stats, train_stats, len(train_loader))
                """

            else:  # No misclassified seqs
                logger.info(yellow(f'Current model has not changed, training for additional {epochs} epochs...'))
                test_stats, train_stats = nesy_train(current_nesy_model.nn_model, train_loader,
                                                     current_nesy_model.sfa_dnnf, cnn_output_size,
                                                     nn_criterion, sequence_criterion, current_nesy_model.optimizer,
                                                     epochs, current_nesy_model.seq_loss_weight, class_attrs,
                                                     test_loader, update_seqs_stats=True, show_log=False)

            show_log_msg(current_nesy_model)

        # Train for a few more epochs in the end
        # nesy_train(model, train_loader, sfa_dnnf, cnn_output_size,
        #            nn_criterion, sequence_criterion, optimizer,
        #            num_epochs, seq_loss_weight, class_attrs, test_loader, show_log=show_stats)


def show_log_msg(current_nesy_model):
    logger.info(green(
        f"Best model:\n{current_nesy_model.sfa_asal.show(mode='simple')}\n"
        f"Loss (combined|target|latent): {current_nesy_model.combined_loss:.3f}|{current_nesy_model.target_loss:.3f}|{current_nesy_model.latent_loss:.3f}\n"
        f"Train F1 (target|latent): {current_nesy_model.train_f1:.3f}|{current_nesy_model.latent_f1:.3f}\n"
        f"Test F1 (target|latent): {current_nesy_model.training_history['seq_f1'][-1]:.3f}|{current_nesy_model.training_history['img_f1'][-1]:.3f}"))


if __name__ == "__main__":
    """
    parser = parse_args()
    args = parser.parse_args()

    asal_args = argparse.Namespace(tclass=args.tclass, batch_size=20000, test=args.test, train=args.train,
                                   domain=args.domain,
                                   predicates="equals",
                                   mcts_iters=10,
                                   all_opt=False,  # Get multiple optimal models!
                                   tlim=120,
                                   states=args.states,
                                   exp_rate=args.exp_rate,
                                   mcts_children=1,
                                   show=args.show,
                                   unsat_weight=10,  # Set this to 0 to have uncertainty weights per sequence
                                   max_alts=args.max_alts,
                                   coverage_first=False,  # args.coverage_first,
                                   min_attrs=args.min_attrs,
                                   warns_off=False,
                                   revise=False,
                                   max_rule_length=100)
    """

    asal_args = argparse.Namespace(tclass=1,
                                   batch_size=20000,
                                   test=None,
                                   train=None,
                                   domain="/home/nkatz/dev/asal/src/asal/asp/domains/mnist_multivar.lp",
                                   predicates="equals",
                                   mcts_iters=10,
                                   all_opt=False,  # Get multiple optimal models!
                                   tlim=120,
                                   states=4,
                                   exp_rate=0.005,
                                   mcts_children=1,
                                   show='s',
                                   unsat_weight=10,  # Set this to 0 to have uncertainty weights per sequence
                                   max_alts=3,
                                   coverage_first=False,  # args.coverage_first,
                                   min_attrs=False,
                                   warns_off=False,
                                   revise=False,
                                   max_rule_length=100,
                                   with_reject_states=False)  # this might need to be changed.

    """"---------------"""
    args = asal_args
    """"---------------"""

    """
    Things to try:
    1. Add more examples than the fully labelled ones during the revision process, properly weighted.
    2. Compute all optimal, select the best.
    3. Labelled the best K seqs instead of just one.
    4. Properly revise the current model
    """

    nn_args = argparse.Namespace(app_name='mnist', num_epochs=100, active_learning_frequency=5, points_to_label=100,
                                 top_N_seqs=20, entropy_scaling_factor=100, w_label_density=2.0, w_seq_entropy=1.0,
                                 w_img_entropy=0.0, nn_batch_size=50, pre_train_nn=True,
                                 pre_training_size=10,  # num of fully labeled seed sequences.
                                 pre_train_num_epochs=100, learn_seed_sfa_from_pretrained=False, )

    num_init_fully_labelled = 2  # Number of initial fully labelled sequences
    num_queries = 10  # Total number of active learning queries
    num_epochs = 20  # 20  # Number of epochs to train after each active learning update
    cnn_output_size = 10  # for MNIST
    pre_train_for = 100  # 10  # 100
    nn_batch_size = 50
    lr = 0.01  # 0.01 initially
    N_samples = 5  # Number of samples/unlabeled sequence for ELR

    """
        BE CAREFUL: When you change from single-variate to two-variate and so on, these need to change:
        1. class_attrs below.
        2. src.asal.asp.mnist_even_odd_learn (comment/uncomment 1 {value(d2,0..9)} 1.)
        3. The domain, e.g. mnist_multivar (comment/uncomment digit(d1; d2). ect)
    """

    # class_attrs = ['d1', 'd2', 'd3']
    class_attrs = ['d1']  # single-digit
    # class_attrs = ['d1', 'd2']  # double-digit

    asp_comp_program = mnist_even_odd_learn
    num_runs = 5  # Number of experiments to run
    random_query = False

    #"""
    train_data, test_data = get_data('/home/nkatz/dev/asal/data/mnist_nesy/single_digit/mnist_train.pt',
                                     '/home/nkatz/dev/asal/data/mnist_nesy/single_digit/mnist_test.pt')
    #"""

    """
    train_data, test_data = get_data('/home/nkatz/dev/asal/data/mnist_nesy/double_digit/mnist_train.pt',
                                     '/home/nkatz/dev/asal/data/mnist_nesy/double_digit/mnist_test.pt')
    """

    run_experiments(train_data, test_data, num_runs,
                    num_queries, num_epochs, pre_train_for, lr,
                    num_init_fully_labelled, asp_comp_program,
                    cnn_output_size, class_attrs, asal_args,
                    random_query=random_query, use_partially_labeled=True)
