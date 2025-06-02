import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
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
                                                       al_expected_acceptance_loss)

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


def save_experiment_results(rs_vals, eal_vals, metric_name, query_budget, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    # Save raw F1 data
    np.save(os.path.join(out_dir, f"{metric_name}_rs_queries={query_budget}.npy"), rs_vals)
    np.save(os.path.join(out_dir, f"{metric_name}_eal_queries={query_budget}.npy"), eal_vals)

    # Save metadata (optional, but helpful)
    meta = {
        "metric": metric_name,
        "runs": rs_vals.shape[0],
        "steps": rs_vals.shape[1]
    }
    with open(os.path.join(out_dir, f"{metric_name}_meta.json"), "w") as f:
        json.dump(meta, f)

    # Save the plot
    # plt.savefig(os.path.join(out_dir, f"{metric_name}_rs_vs_eal.png"), dpi=300)


def plot_saved_results(metric_name, query_budget, out_dir="results"):
    import numpy as np
    import matplotlib.pyplot as plt
    import json

    rs_vals = np.load(os.path.join(out_dir, f"{metric_name}_rs_queries={query_budget}.npy"))
    eal_vals = np.load(os.path.join(out_dir, f"{metric_name}_eal_queries={query_budget}.npy"))

    with open(os.path.join(out_dir, f"{metric_name}_meta.json")) as f:
        meta = json.load(f)

    x = np.arange(1, rs_vals.shape[1] + 1)

    for name, vals, color in zip(['Random Sampling', 'Expected Acceptance Loss'],
                                 [rs_vals, eal_vals], ['blue', 'green']):
        mean_vals = np.mean(vals, axis=0)
        std_vals = np.std(vals, axis=0)
        plt.plot(x, mean_vals, label=name, color=color)
        plt.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, color=color)

    plt.xlabel("Training Epochs")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} Comparison: RS vs EAL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_current_cumulative_loss(model, train_loader, sfa_dnnf, cnn_output_size, sequence_criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in train_loader:
            acceptance_probs, _, _ = nesy_forward_pass(batch, model, sfa_dnnf, cnn_output_size)
            labels = torch.tensor([seq.seq_label for seq in batch]).to(device)
            # Multiply with the batch loss with the size of the batch because nn.BCELoss()
            # the average loss over the batch, but for ELR we need the cumulative loss for all sequences
            total_loss += sequence_criterion(acceptance_probs, labels.float()).item() * len(batch)
    return total_loss

"""
def run_random_sampling_experiments(N_runs, train_loader):
    all_histories = []
    for _ in range(N_runs):
        fully_labeled_seq_ids = initialize_fully_labeled_seqs(train_loader, num_init_fully_labelled)
        fully_labelled_seqs = list(filter(lambda seq: seq.seq_id in fully_labeled_seq_ids, train_data))
        for seq in fully_labelled_seqs:
            seq.mark_seq_as_fully_labelled()

        symb_seqs = [s.get_symbolic_seq() for s in fully_labelled_seqs]
        sfa_dnnf, sfa_asal = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)
        model = DigitCNN(out_features=cnn_output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        sequence_criterion = nn.BCELoss()
        nn_criterion = nn.CrossEntropyLoss()

        pretrain_nn(SequenceDataset(fully_labelled_seqs), test_data, 0,
                    model, optimizer, class_attrs, with_fully_labelled_seqs=True, num_epochs=10)

        nesy_train(model, train_loader, sfa_dnnf, cnn_output_size, nn_criterion, sequence_criterion,
                   optimizer, epochs, seq_loss_weight)

        hist = al_random_sampling(model, al_queries, epochs, fully_labelled_seqs,
                                  nn_criterion, sequence_criterion, optimizer, seq_loss_weight)

        all_histories.append(hist)

    def plot_metric(metric_name):
        all_vals = np.array([h[metric_name] for h in all_histories])  # shape: (N_runs, al_queries)
        mean_vals = np.mean(all_vals, axis=0)
        std_vals = np.std(all_vals, axis=0)

        x = np.arange(1, len(mean_vals) + 1)
        plt.figure()
        plt.errorbar(x, mean_vals, yerr=std_vals, label=metric_name.upper())
        plt.xlabel("Active Learning Queries")
        plt.ylabel(metric_name.upper())
        plt.title(f"{metric_name.upper()} vs Queries (Random Sampling)")
        plt.legend()
        plt.grid(True)
        plt.show()
        # save_experiment_results(rs_vals, eal_vals, metric_name, out_dir="results")

    plot_metric('seq_f1')
    plot_metric('img_f1')
"""

def run_comparative_experiments_rs_vs_eal(train_data, test_data, N_runs, query_budget, epochs,
                                          pretrain_for, lr, num_init_fully_labelled, asp_comp_program,
                                          cnn_output_size, seq_loss_weight, class_attrs, asal_args):
    rs_histories, eal_histories = [], []
    for _ in range(N_runs):
        # Get new loaders for each experiments to make sure the data gets shuffled.
        train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, nn_args.nn_batch_size, train=True)
        test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, nn_args.nn_batch_size, train=False)

        fully_labeled_seq_ids = initialize_fully_labeled_seqs(train_loader, num_init_fully_labelled)
        seed_seqs = list(filter(lambda seq: seq.seq_id in fully_labeled_seq_ids, train_data))
        for seq in seed_seqs:
            seq.mark_seq_as_fully_labelled()

        logger.info("Inducing initial SFA from fully labelled sequences...")

        symb_seqs = [s.get_symbolic_seq() for s in seed_seqs]
        init_sfa_dnnf, init_sfa_asal = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)[-1]

        base_model = DigitCNN(out_features=cnn_output_size).to(device)
        optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)  # 0.01
        sequence_criterion = nn.BCELoss()
        nn_criterion = nn.CrossEntropyLoss()

        logger.info(f"Pre-training CNN on available labels for {pretrain_for} epochs...")

        pretrain_nn(SequenceDataset(seed_seqs), test_data, 0,
                    base_model, optimizer, class_attrs, with_fully_labelled_seqs=True, num_epochs=pretrain_for)

        logger.info("Initial NeSy CNN/SFA training...")

        nesy_train(base_model, train_loader, init_sfa_dnnf, cnn_output_size,
                   nn_criterion, sequence_criterion, optimizer, epochs, seq_loss_weight, class_attrs, test_loader)

        rs_model = deepcopy(base_model)
        eal_model = deepcopy(base_model)
        rs_optimizer = torch.optim.Adam(rs_model.parameters(), lr=lr)
        eal_optimizer = torch.optim.Adam(eal_model.parameters(), lr=lr)

        logger.info(green(f"Active learning with random sampling for {query_budget} queries..."))

        # misclassified_seqs = get_misclassified_seqs(model, train_loader, sfa_dnnf, cnn_output_size)

        rs_history = al_random_sampling(train_data, train_loader, test_loader, rs_model, query_budget,
                                        epochs, list(seed_seqs), cnn_output_size, asal_args, asp_comp_program,
                                        class_attrs, nn_criterion, sequence_criterion, rs_optimizer, seq_loss_weight)

        logger.info(green(f"Active learning with EAL for {query_budget} queries..."))

        eal_history = al_expected_acceptance_loss(eal_model, train_loader, test_loader, train_data, init_sfa_dnnf,
                                                  query_budget, epochs, list(seed_seqs), nn_criterion,
                                                  sequence_criterion, eal_optimizer, seq_loss_weight,
                                                  cnn_output_size, asal_args, asp_comp_program, class_attrs)

        rs_histories.append(rs_history)
        eal_histories.append(eal_history)

    def plot_comparison(metric_name, title_name):
        def extract_vals(histories):
            return np.array([h[metric_name] for h in histories])

        rs_vals = extract_vals(rs_histories)
        # x = np.arange(1, al_queries + 1)
        x = np.arange(1, rs_vals.shape[1] + 1)
        eal_vals = extract_vals(eal_histories)

        plt.figure()
        for name, vals, color in zip(['Random Sampling', 'Expected Acceptance Loss'],
                                     [rs_vals, eal_vals], ['blue', 'green']):
            mean_vals = np.mean(vals, axis=0)
            std_vals = np.std(vals, axis=0)
            plt.plot(x, mean_vals, label=name, color=color)

            """
            # This  to show variability across runs, i.e., how much the performance fluctuates with 
            # random seeds or sequence choices. But to show statistical confidence in the mean F1, 
            # especially for comparing methods, we should instead plot the 95% confidence interval (or similar), 
            # which reflects uncertainty in the estimated mean, not the full variation across runs.
            
            lower = np.clip(mean_vals - std_vals, 0.0, 1.0)
            upper = np.clip(mean_vals + std_vals, 0.0, 1.0)
            plt.fill_between(x, lower, upper, alpha=0.3, color=color)
            """
            # 95% confidence intervals:
            n_runs = rs_vals.shape[0]  # or len(histories)
            stderr = std_vals / np.sqrt(n_runs)
            ci = 1.96 * stderr

            lower = np.clip(mean_vals - ci, 0.0, 1.0)
            upper = np.clip(mean_vals + ci, 0.0, 1.0)
            plt.fill_between(x, lower, upper, alpha=0.3, color=color)

        for q in range(1, query_budget):
            plt.axvline(q * epochs, linestyle='--', color='gray', alpha=0.3)

        for q in range(1, query_budget + 1):
            x_pos = (q - 0.5) * epochs
            plt.text(x_pos, plt.ylim()[1] * 0.95, f"Q{q}", ha='center', fontsize=8, color='gray')

        plt.xlabel("Epochs after Active Learning Queries")
        plt.ylabel("Test F1-Score")
        plt.title(title_name, fontsize=13)
        plt.legend()
        plt.grid(True)
        plt.show()
        save_experiment_results(rs_vals, eal_vals, metric_name, query_budget, out_dir="/home/nkatz/dev/asal_plots")

    plot_comparison('seq_f1', "Sequence classification")
    plot_comparison('img_f1', "Image classification")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    asal_args = argparse.Namespace(tclass=args.tclass, batch_size=20000, test=args.test, train=args.train,
                                   domain=args.domain,
                                   predicates="equals", mcts_iters=10, all_opt=False, tlim=60, states=args.states,
                                   exp_rate=args.exp_rate, mcts_children=args.mcts_children, show=args.show,
                                   unsat_weight=10,  # set this to 0 to have uncertainty weights per sequence
                                   max_alts=args.max_alts, coverage_first=args.coverage_first, min_attrs=args.min_attrs,
                                   warns_off=False, revise=False)

    num_init_fully_labelled = 20  # Number of initial fully labelled sequences
    num_queries = 3  # Total number of active learning queries
    num_epochs = 20  # Number of epochs to train after each active learning update
    cnn_output_size = 10  # for MNIST
    pre_train_for = 100
    nn_batch_size = 50
    lr = 0.01  # 0.01 initially
    seq_loss_weight = 1.0
    N_samples = 5  # Number of samples/unlabeled sequence for ELR
    # class_attrs = ['d1', 'd2', 'd3']
    class_attrs = ['d1']
    asp_comp_program = mnist_even_odd_learn
    num_runs = 1  # Number of experiments to run

    logger.info(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    """
    train_data, test_data = get_data('/home/nkatz/dev/asal/data/mnist_nesy/mnist_train.pt',
                                     '/home/nkatz/dev/asal/data/mnist_nesy/mnist_test.pt')

    train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, nn_args.nn_batch_size, train=True)
    test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, nn_args.nn_batch_size, train=False)
   """

    """======================================================================="""
    # plot_saved_results('seq_f1', 20, out_dir="/home/nkatz/dev/asal_plots")
    # run_random_sampling_experiments(5)

    train_data, test_data = get_data('/home/nkatz/dev/asal/data/mnist_nesy/mnist_train.pt',
                                     '/home/nkatz/dev/asal/data/mnist_nesy/mnist_test.pt')
    run_comparative_experiments_rs_vs_eal(train_data, test_data, num_runs, num_queries,
                                          num_epochs, pre_train_for, lr, num_init_fully_labelled, asp_comp_program,
                                          cnn_output_size, seq_loss_weight, class_attrs, asal_args)
    """======================================================================="""

    #----------------------------------------------------------------------------------------------
    """
    logger.info("Initializing model")
    fully_labeled_seq_ids = initialize_fully_labeled_seqs(train_loader, num_init_fully_labelled)
    fully_labelled_seqs = list(filter(lambda seq: seq.seq_id in fully_labeled_seq_ids, train_data))
    for seq in fully_labelled_seqs:
        seq.mark_seq_as_fully_labelled()

    symb_seqs = [s.get_symbolic_seq() for s in fully_labelled_seqs]

    sfa_dnnf, sfa_asal = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)
    #-----------------------------------------------------------------------------------------------

    model = DigitCNN(out_features=cnn_output_size).to(device)
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sequence_criterion = nn.BCELoss()
    nn_criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------------------------------------
    logger.info("Pre-train the NN...")
    pretrain_nn(SequenceDataset(fully_labelled_seqs), test_data, 0,
                model, optimizer, class_attrs, with_fully_labelled_seqs=True, num_epochs=10)
    logger.info("Initial NeSy training...")
    nesy_train(model, train_loader, sfa_dnnf, cnn_output_size, nn_criterion, sequence_criterion,
               optimizer, epochs, seq_loss_weight)
    #----------------------------------------------------------------------------------------------

    logger.info("Computing loss and identifying misclassified examples with current model")
    current_loss = get_current_cumulative_loss(model, train_loader, sfa_dnnf, cnn_output_size, sequence_criterion)
    misclassified_seqs = get_misclassified_seqs(model, train_loader, sfa_dnnf, cnn_output_size)
    logger.info(
        f"Misclassified: {len(misclassified_seqs)}, current cumulative loss examples with current model: {current_loss}")
    """

    """This is the ELR thing, it should be included in a method. But it is extremely slow (and still throws
    out of memory errors - I think - need to check this. It is not realistic to have something like this)
    
    for i, candidate_seq in enumerate(misclassified_seqs):
        logger.info(f'{i}/{len(misclassified_seqs)} Computing ELR score for sequence {candidate_seq.seq_id}')
        compute_elr_score(candidate_seq, model, cnn_output_size, train_loader, nn_criterion,
                          sequence_criterion, lr, class_attrs, seq_loss_weight, N_samples, symb_seqs, current_loss)
        logger.info(f'ELR score for sequence {candidate_seq.seq_id}: {candidate_seq.elr_score}')
    """
