import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset, TensorSequence
from src.globals import device, fix_seed
from src.logger import *
from src.asal_nesy.neurasal.neurasal_functions import initialize_fully_labeled_seqs, pretrain_nn, sequence_to_facts


# ========= DigitCNN =========
class DigitCNN(nn.Module):
    def __init__(self, dropout_rate=0.3, out_features=10, log_softmax: bool = False):
        super().__init__()
        self.out_features = out_features
        self.conv1 = nn.Conv2d(1, 8, (3, 3))
        self.conv2 = nn.Conv2d(8, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.dense = nn.Linear(in_features=32, out_features=out_features)
        self.log_softmax = log_softmax
        self.softmax = nn.LogSoftmax(dim=1) if log_softmax else nn.Softmax(dim=1)
        self.last_logits = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_image, apply_softmax=True, store_output=False, return_features=False, with_dropout=False):
        x = self.avg_pool(self.relu(self.conv1(input_image)))
        x = self.avg_pool(self.relu(self.conv2(x)))
        x = self.avg_pool(self.relu(self.conv3(x)))
        if with_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        if return_features:
            return x
        logits = self.dense(x)
        if store_output:
            self.last_logits = logits
        return self.softmax(logits) if apply_softmax else logits


# ========= Rule Mapping =========
def digit_probs_to_rule_probs_batch(digit_probs):
    p = digit_probs
    rule_1 = p[:, 8]                    # even, gt_6
    rule_2 = p[:, 4] + p[:, 6]         # even, leq_6, gt_3
    rule_3 = p[:, 0] + p[:, 2]         # even, leq_3
    rule_4 = p[:, 7] + p[:, 9]         # odd, gt_6
    rule_5 = p[:, 5]                   # odd, leq_6, gt_3
    rule_6 = p[:, 1] + p[:, 3]         # odd, leq_3
    return torch.stack([rule_1, rule_2, rule_3, rule_4, rule_5, rule_6], dim=1)


def softmax_with_temperature(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)

"""
# ========= NeuralSFA =========
class NeuralSFA(nn.Module):
    def __init__(self, num_states, num_guards, learn_sfa=False, all_trainable=False, cnn_out_features=10):
        # If learn_sfa=True it learns an SFA based on predefined guards. Else it relies on propositionalization
        # to learn a DFA. In this case num_guards = alphabet_size = num_models (one symbol per model).

        super().__init__()

        self.num_states = num_states
        self.num_rules = num_guards
        self.all_trainable = all_trainable
        self.learn_sfa = learn_sfa
        self.cnn = DigitCNN(out_features=cnn_out_features)
        self.device = device

        print(f"NeuralSFA initialized with {self.num_states} states and {self.num_rules} rules")

        if not all_trainable:
            self.transition_matrices = nn.ParameterList([
                nn.Parameter(torch.softmax(torch.randn(self.num_states - 1, self.num_states), dim=1))
                for _ in range(num_guards)
            ])
            self.accepting_state_row = torch.zeros(self.num_states).to(device)
            self.accepting_state_row[-1] = 1.0
        else:
            self.transition_matrices = nn.ParameterList([
                nn.Parameter(torch.softmax(torch.randn(self.num_states, self.num_states), dim=1))
                for _ in range(num_guards)
            ])

    def get_effective_transition_matrix(self, rule_probs, softmax_temp):
        A = torch.zeros(self.num_states, self.num_states, device=rule_probs.device)
        for i, prob in enumerate(rule_probs):
            matrix = softmax_with_temperature(self.transition_matrices[i], softmax_temp)
            if not self.all_trainable:
                matrix = torch.cat((matrix, self.accepting_state_row.unsqueeze(0)), dim=0)
            A += prob * matrix
        return A

    def forward(self, sequences, softmax_temp=0.1):
        
        # sequences: Tensor of shape (B, T, D, C, H, W)
        
        B, T, D, C, H, W = sequences.shape
        state_dists = torch.zeros(B, self.num_states, device=sequences.device)
        state_dists[:, 0] = 1.0

        all_cnn_preds, all_guard_preds = [], []
        for t in range(T):
            for d in range(D):
                images = sequences[:, t, d]               # (B, C, H, W)
                digit_probs = self.cnn(images)            # (B, 10)

                #------------------------------------------------------------------------------------------
                # Needs to be fixed, we need a method that computes model probabilities (products) from the
                # latent concept probabilities. This is just for univariate MNIST, for testing purposes.
                if self.learn_sfa:
                    guard_probs = digit_probs_to_rule_probs_batch(digit_probs)  # (B, num_rules)
                else:
                    guard_probs = digit_probs
                # ------------------------------------------------------------------------------------------

                all_cnn_preds.append(digit_probs)
                all_guard_preds.append(guard_probs)

                updated_states = []
                for b in range(B):
                    A = self.get_effective_transition_matrix(guard_probs[b], softmax_temp)
                    new_state = torch.matmul(state_dists[b], A)
                    updated_states.append(new_state)
                state_dists = torch.stack(updated_states, dim=0)

        return all_cnn_preds, all_guard_preds, state_dists  # final state distribution per sequence
"""

#===================================================================================================
# This is a more efficient version of the above for GPU-parallelism in automaton transitions:
# 1. Has a batched version of get_effective_transition_matrix.
# 2. Replaces the per-sequence for loop in forward() with batched matrix multiplication (torch.bmm)
# ==================================================================================================
class NeuralSFA(nn.Module):
    def __init__(self, num_states, num_guards, learn_sfa=False, all_trainable=False, cnn_out_features=10):
        super().__init__()

        self.num_states = num_states
        self.num_rules = num_guards
        self.all_trainable = all_trainable
        self.learn_sfa = learn_sfa
        self.cnn = DigitCNN(out_features=cnn_out_features)
        self.device = device

        print(f"NeuralSFA initialized with {self.num_states} states and {self.num_rules} rules")

        if not all_trainable:
            self.transition_matrices = nn.ParameterList([
                nn.Parameter(torch.softmax(torch.randn(self.num_states - 1, self.num_states), dim=1))
                for _ in range(num_guards)
            ])
            self.accepting_state_row = torch.zeros(self.num_states).to(device)
            self.accepting_state_row[-1] = 1.0
        else:
            self.transition_matrices = nn.ParameterList([
                nn.Parameter(torch.softmax(torch.randn(self.num_states, self.num_states), dim=1))
                for _ in range(num_guards)
            ])

    def get_effective_transition_matrix_batch(self, guard_probs_batch, softmax_temp):
        """
        guard_probs_batch: (B, num_rules)
        returns: (B, num_states, num_states)
        """
        B = guard_probs_batch.size(0)
        S = self.num_states
        device = guard_probs_batch.device
        A_batch = torch.zeros(B, S, S, device=device)

        for r in range(self.num_rules):
            matrix = softmax_with_temperature(self.transition_matrices[r], softmax_temp)  # (S-1, S) or (S, S)
            if not self.all_trainable:
                matrix = torch.cat((matrix, self.accepting_state_row.unsqueeze(0)), dim=0)  # (S, S)
            A_batch += guard_probs_batch[:, r].unsqueeze(-1).unsqueeze(-1) * matrix.unsqueeze(0)  # (B, S, S)

        return A_batch  # (B, S, S)

    def forward(self, sequences, softmax_temp=0.1):
        """
        sequences: Tensor of shape (B, T, D, C, H, W)
        """
        B, T, D, C, H, W = sequences.shape
        state_dists = torch.zeros(B, self.num_states, device=sequences.device)
        state_dists[:, 0] = 1.0  # Start state

        all_cnn_preds, all_guard_preds = [], []

        for t in range(T):
            for d in range(D):
                images = sequences[:, t, d]               # (B, C, H, W)
                digit_probs = self.cnn(images)            # (B, 10)

                # --------------------
                if self.learn_sfa:
                    guard_probs = digit_probs_to_rule_probs_batch(digit_probs)  # (B, num_rules)
                else:
                    guard_probs = digit_probs
                # --------------------

                all_cnn_preds.append(digit_probs)
                all_guard_preds.append(guard_probs)

                # 🧠 Parallelized automaton step
                A_batch = self.get_effective_transition_matrix_batch(guard_probs, softmax_temp)  # (B, S, S)
                state_dists = torch.bmm(state_dists.unsqueeze(1), A_batch).squeeze(1)  # (B, S)

        return all_cnn_preds, all_guard_preds, state_dists  # final state distribution per sequence



# ========= Utilities =========
def tensor_sequences_to_tensor(batch):
    B = len(batch)
    T, D, C, H, W = batch[0].images.shape
    x = torch.stack([seq.images for seq in batch]).to(device)
    y = torch.tensor([seq.seq_label for seq in batch], dtype=torch.float32).to(device)
    return x, y


def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = tensor_sequences_to_tensor(batch)
            cnn_preds, _, final_state = model(x)
            preds = (final_state[:, -1] > 0.5).int().cpu().tolist()
            labels = y.int().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return f1_score(all_labels, all_preds, zero_division=0)


# ========= Main Training Script =========
if __name__ == "__main__":
    seed = 1
    fix_seed(seed)

    batch_size = 32
    states = 50

    #-----------------------------------
    learn_sfa = False
    num_guards = 6 if learn_sfa else 10
    # -----------------------------------

    lr = 0.001
    num_epochs = 100
    pretrain_cnn_for = 100
    num_fully_labelled_seqs = 2000
    class_attrs = ['d1']

    # Load data
    train_data, test_data = get_data(
        '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_train.pt',
        '/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_test.pt'
    )
    train_loader = get_data_loader(train_data, batch_size=batch_size, train=True)
    test_loader = get_data_loader(test_data, batch_size=batch_size, train=False)

    model = NeuralSFA(num_states=states, num_guards=num_guards).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sequence_criterion = nn.BCELoss()

    base_model = model.cnn.to(device)
    nn_criterion = nn.CrossEntropyLoss()
    fully_labeled_seq_ids = initialize_fully_labeled_seqs(train_loader, num_fully_labelled_seqs)
    fully_labelled_seqs = list(filter(lambda seq: seq.seq_id in fully_labeled_seq_ids, train_data))
    logger.info(yellow(f'Fully labelled: {[f"{s.seq_id}:{s.seq_label}" for s in fully_labelled_seqs]}'))

    for seq in fully_labelled_seqs:  # Mark them all as fully labelled
        seq.mark_seq_as_fully_labelled()

    logger.info(f"Pre-training CNN on available labels for {pretrain_cnn_for} epochs...")
    pretrain_nn(SequenceDataset(fully_labelled_seqs), test_data, 0,
                base_model, optimizer, class_attrs, with_fully_labelled_seqs=True, num_epochs=pretrain_cnn_for)

    """
    for epoch in range(1, 100):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = tensor_sequences_to_tensor(batch)
            _, _, final_state = model(x)
            accept_probs = final_state[:, -1]

            # print("Accept probs:", accept_probs.detach().cpu().numpy())

            loss = criterion(accept_probs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_f1 = evaluate_model(model, train_loader)
        test_f1 = evaluate_model(model, test_loader)
        print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")
    """

    for epoch in range(1, num_epochs):
        model.train()
        sequence_loss_sum = 0.0
        latent_loss_sum = 0.0
        total_loss_sum = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = tensor_sequences_to_tensor(batch)  # (B, T, D, C, H, W), (B,)
            cnn_preds, _, final_state = model(x)

            # Sequence-level loss
            accept_probs = final_state[:, -1]
            seq_loss = sequence_criterion(accept_probs, y)

            # Latent loss
            B, T, D, C, H, W = x.shape
            cnn_outputs = torch.stack(cnn_preds).view(T, D, B, -1).permute(2, 0, 1, 3)  # (B, T, D, 10)

            latent_loss = 0.0 * cnn_outputs.sum()
            labeled_preds, labeled_labels = [], []

            for i, seq in enumerate(batch):
                for (t, d) in seq.get_labeled_indices():
                    pred = cnn_outputs[i, t, d]
                    label_dict = seq.get_image_label(t, d, class_attrs)

                    # print(f"Label dict: {label_dict}")

                    label = torch.tensor(label_dict[0]).long().to(device)  ### <-- This is what I change
                    labeled_preds.append(pred)
                    labeled_labels.append(label)

            if labeled_preds:
                labeled_preds_tensor = torch.stack(labeled_preds)  # (N, 10)
                labeled_labels_tensor = torch.cat(labeled_labels)  # (N,)
                latent_loss = nn_criterion(labeled_preds_tensor, labeled_labels_tensor)

            # Combine losses
            """
            if weight_sequence_loss:
                total_batch_loss = sequence_loss_weight * seq_loss + latent_loss
            else:
                total_batch_loss = seq_loss + latent_loss
            """
            total_batch_loss = seq_loss + latent_loss

            # Backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            # Accumulate
            sequence_loss_sum += seq_loss.item()
            latent_loss_sum += latent_loss.item()
            total_loss_sum += total_batch_loss.item()
            num_batches += 1

        # Evaluation
        train_f1 = evaluate_model(model, train_loader)
        test_f1 = evaluate_model(model, test_loader)

        # Print final averaged losses and F1
        avg_seq_loss = sequence_loss_sum / num_batches
        avg_latent_loss = latent_loss_sum / num_batches
        avg_total_loss = total_loss_sum / num_batches

        print(f"Epoch {epoch} | Total: {avg_total_loss:.4f} | Seq: {avg_seq_loss:.4f} | Latent: {avg_latent_loss:.4f} "
              f"| Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")

