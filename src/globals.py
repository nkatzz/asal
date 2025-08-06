import torch
import random

def fix_seed(seed):
    # Reproducibility
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    random.seed(seed)

    torch.backends.cudnn.deterministic = True  # ensure that CUDA algorithms use deterministic implementations.
    torch.backends.cudnn.benchmark = False  # prevent cuDNN from choosing the fastest (but nondeterministic) algorithm
    # np.random.seed(seed)  # not used

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weight_sequence_loss = False  # Based on the SFA's performance on the training set

sample_symb_seqs = True  # Sample symbolic seqs from the NN and test them against the (crisp) SFA for deciding AL query

num_top_k = 50  # 50  # Number of top-k highest probability examples to use in neurasal induction.

pick_by_edit_cost = True  # pick sequence to label for active learning by edit cost.

show_log_during_training = False  # Show train/test performance after each epoch.

# allow to include a #true self loop on the accepting state.
# This is forced if the constraint :- state(S), not transition(S,_,S). is included, see Template.py
is_accepting_absorbing = True
