import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weight_sequence_loss = False  # Based on the SFA's performance on the training set

sample_symb_seqs = True  # Sample symbolic seqs from the NN and test them against the (crisp) SFA for deciding AL query

num_top_k = 100  # Number of top-k highest probability examples to use in neurasal induction.
