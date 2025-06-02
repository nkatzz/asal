from collections import defaultdict
import torch
import sys
from torch.utils.data import DataLoader
from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset
from src.asal_nesy.neurasal.neurasal_functions import (nesy_forward_pass, pretrain_nn, induce_sfa, nesy_train,
                                                       initialize_fully_labeled_seqs, al_random_sampling,
                                                       al_expected_acceptance_loss)
import os
import argparse
from src.args_parser import parse_args
from src.asal_nesy.cirquits.asp_programs import mnist_even_odd_learn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    asal_args = argparse.Namespace(tclass=args.tclass, batch_size=20000, test=args.test, train=args.train,
                                   domain=args.domain,
                                   predicates="equals", mcts_iters=10, all_opt=args.all_opt, tlim=60, states=args.states,
                                   exp_rate=args.exp_rate, mcts_children=args.mcts_children, show=args.show,
                                   unsat_weight=10,  # set this to 0 to have uncertainty weights per sequence
                                   max_alts=args.max_alts, coverage_first=args.coverage_first, min_attrs=args.min_attrs,
                                   warns_off=False, revise=False)

    class_attrs = ['d1']
    asp_comp_program = mnist_even_odd_learn

    train_data, test_data = get_data('/home/nkatz/dev/asal/data/mnist_nesy/mnist_train.pt',
                                     '/home/nkatz/dev/asal/data/mnist_nesy/mnist_test.pt')
    train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, batch_size=50, train=True)
    test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, batch_size=50, train=False)

    # seq_ids = [1445, 790, 1299, 1182, 80, 893, 1440, 1259, 1661, 36, 194, 292, 1623, 1448, 731, 159, 1690, 466, 325,
    #           1551]

    seq_ids = [1445, 790, 1299, 1182, 80, 893]
    symb_seqs = [s.get_symbolic_seq() for s in train_data if s.seq_id in seq_ids]
    sfa_dnnf, sfa_asal = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)[-1]
    print(sfa_asal.show())
