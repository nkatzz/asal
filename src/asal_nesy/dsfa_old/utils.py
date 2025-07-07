import torch
import torch.nn.functional as F
# from mnist_seqs_new import get_data_loaders
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.asal_nesy.globals import device


def see_gradients(model):
    for i, matrix in enumerate(model.transition_matrices):
        if matrix.requires_grad:
            print(f'Gradients for Rule {i}:\n{matrix.grad}')


def see_parameters(model):
    for i, matrix in enumerate(model.transition_matrices):
        print(f'Transition Matrix for Rule {i}:\n{matrix}')


def is_stochastic(matrix, tol=1e-5):
    """
    Check if a matrix is stochastic (square matrix with non-negative entries where each row sums to 1.0).
    Typically used to describe the transitions of a Markov chain.

    Args:
    - matrix (torch.Tensor): A square matrix to be checked.
    - tol (float): A tolerance level for comparing sums to 1.0, to account for numerical precision issues.
    """
    if not matrix.ndim == 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    # Check for non-negative entries
    if torch.any(matrix < 0):
        return False

    # Check if each row sums to 1 (within a tolerance)
    row_sums = matrix.sum(dim=1)
    if torch.all(torch.abs(row_sums - 1.0) <= tol):
        return True
    else:
        return False


def custom_cross_entropy(states_probs, labels):
    # Example usage:
    # state_prob = torch.tensor([[0.0908, 0.0788, 0.0491, 0.7813]])
    # label = torch.tensor([1])
    # loss = custom_cross_entropy(state_prob, label)
    epsilon = 1e-8
    states_probs = torch.clamp(states_probs, epsilon, 1 - epsilon)
    accepting_prob = states_probs[-1]  # Assuming the last state is the accepting state
    # Calculate the cross-entropy loss manually
    loss = -labels * torch.log(accepting_prob + epsilon) - (1 - labels) * torch.log(1 - accepting_prob + epsilon)
    return loss  # loss.mean()


def get_stats(predicted, actual):
    # Ensure that predicted and actual are on the CPU and convert them to NumPy arrays
    predicted = [int(p) for p in predicted]
    actual = [int(a) for a in actual]

    # Calculate tp, fp, fn for binary classification metrics
    tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    tn = sum(p == 0 and a == 0 for p, a in zip(predicted, actual))

    # Calculate precision, recall, and F1-score for binary classification
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_binary = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate F1-macro and F1-weighted using sklearn
    f1_macro = f1_score(actual, predicted, average='macro')
    f1_weighted = f1_score(actual, predicted, average='weighted')

    return f1_binary, f1_macro, f1_weighted, tp, fp, fn, tn


def test_model(model, num_states, test_loader, softmax_temp=0.1, nesy_mode=True):
    debug = False
    input_domain = list(range(0, 10))
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for sequence, label, symb_seq in test_loader:
            sequence = [tensor.to(device) for tensor in sequence]
            label = label.to(device)
            # symb_seq = torch.tensor(symb_seq).to(device)
            symb_seq = symb_seq.to(device)

            if nesy_mode:
                cnn_prediction, guard_prediction, states_probs = model(sequence, softmax_temp)
            else:
                cnn_prediction, guard_prediction, states_probs = model(symb_seq, softmax_temp)

            if debug:
                input_predictions = [input_domain[torch.argmax(x).item()] for x in cnn_prediction]
                print(f'{input_predictions}\n{symb_seq[0]}\n')

            predicted_state = torch.argmax(states_probs).item()
            prediction = 1 if predicted_state == num_states - 1 else 0

            actual.append(label.item())
            predicted.append(prediction)

    _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)
    print(f'Test  F1: {train_f1} ({tps}, {fps}, {fns})')


def test_model_fixed_sfa(model, num_states, test_loader, where='Test'):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for sequence, label, symb_seq in test_loader:
            sequence = sequence.to(device)
            label = label.to(device)
            cnn_prediction, guard_prediction, states_probs = model(sequence)

            predicted_state = torch.argmax(states_probs, dim=1)
            prediction = []
            for ps in predicted_state:
                prediction += [1] if ps == num_states - 1 else [0]

            actual.extend(label.tolist())
            predicted.extend(prediction)

    _, f1, _, tps, fps, fns, _ = get_stats(predicted, actual)
    print(f'{where}  F1: {f1} ({tps}, {fps}, {fns})')


def digit_probs_to_rule_probs(rules_num, digit_prob_dist):
    p_0 = digit_prob_dist[0, 0]
    p_1 = digit_prob_dist[0, 1]
    p_2 = digit_prob_dist[0, 2]
    p_3 = digit_prob_dist[0, 3]
    p_4 = digit_prob_dist[0, 4]
    p_5 = digit_prob_dist[0, 5]
    p_6 = digit_prob_dist[0, 6]
    p_7 = digit_prob_dist[0, 7]
    p_8 = digit_prob_dist[0, 8]
    p_9 = digit_prob_dist[0, 9]

    rule_1 = p_8  # even, gt_6
    rule_2 = p_4 + p_6  # even, leq_6, gt_3
    rule_3 = p_0 + p_2  # even, leq_3

    rule_4 = p_7 + p_9  # odd, gt_6
    rule_5 = p_5  # odd, leq_6, gt_3
    rule_6 = p_1 + p_3  # odd, leq_3

    # digit_probabilities = [p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9]
    # as_tensor = torch.stack(digit_probabilities).to(device)

    rule_probabilities = [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]
    as_tensor = torch.stack(rule_probabilities).to(device)
    return as_tensor


def set_correct_model(model):
    """
    Transition matrices for the following mutually exclusive guards:

    rule(7) :- even(X), larger_than_6(X).
    rule(8) :- even(X), larger_than_3(X), less_eq_6(X).
    rule(9) :- even(X), less_eq_3(X).

    rule(10) :- odd(X), larger_than_6(X).
    rule(11) :- odd(X), less_eq_6(X), larger_than_3(X).
    rule(12) :- odd(X), less_eq_3(X).
    """
    model.transition_matrices[0][0] = torch.tensor([0., 1., 0., 0.])  # 1 -> 1, 2, 3, 4
    model.transition_matrices[0][1] = torch.tensor([0., 1., 0., 0.])  # 2 -> 1, 2, 3, 4
    model.transition_matrices[0][2] = torch.tensor([0., 0., 1., 0.])  # 3 -> 1, 2, 3, 4

    model.transition_matrices[1][0] = torch.tensor([1., 0., 0., 0.])
    model.transition_matrices[1][1] = torch.tensor([0., 1., 0., 0.])
    model.transition_matrices[1][2] = torch.tensor([0., 0., 1., 0.])

    model.transition_matrices[2][0] = torch.tensor([1., 0., 0., 0.])
    model.transition_matrices[2][1] = torch.tensor([0., 1., 0., 0.])
    model.transition_matrices[2][2] = torch.tensor([0., 0., 0., 1.])

    model.transition_matrices[3][0] = torch.tensor([1., 0., 0., 0.])
    model.transition_matrices[3][1] = torch.tensor([0., 1., 0., 0.])
    model.transition_matrices[3][2] = torch.tensor([0., 0., 1., 0.])

    model.transition_matrices[4][0] = torch.tensor([1., 0., 0., 0.])
    model.transition_matrices[4][1] = torch.tensor([0., 0., 1., 0.])
    model.transition_matrices[4][2] = torch.tensor([0., 0., 1., 0.])

    model.transition_matrices[5][0] = torch.tensor([1., 0., 0., 0.])
    model.transition_matrices[5][1] = torch.tensor([0., 0., 1., 0.])
    model.transition_matrices[5][2] = torch.tensor([0., 0., 0., 1.])


def digit_probs_to_rule_probs_no_nesy(digit):
    rule_1 = torch.tensor(1.0) if digit == 8 else torch.tensor(0.0)  # even, gt_6
    rule_2 = torch.tensor(1.0) if (digit == 4 or digit == 6) else torch.tensor(0.0)  # even, leq_6, gt_3
    rule_3 = torch.tensor(1.0) if (digit == 0 or digit == 2) else torch.tensor(0.0)  # even, leq_3

    rule_4 = torch.tensor(1.0) if (digit == 7 or digit == 9) else torch.tensor(0.0)  # odd, gt_6
    rule_5 = torch.tensor(1.0) if digit == 5 else torch.tensor(0.0)  # odd, leq_6, gt_3
    rule_6 = torch.tensor(1.0) if (digit == 1 or digit == 3) else torch.tensor(0.0)  # odd, leq_3

    # digit_probabilities = [p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9]
    # as_tensor = torch.stack(digit_probabilities).to(device)

    rule_probabilities = [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]
    as_tensor = torch.stack(rule_probabilities).to(device)
    return as_tensor


def show_debug_stms(cnn_predictions, guards_predictions,
                    symbolic_sequence, label, input_domain,
                    guards, states_probs, num_states):
    input_predictions = [input_domain[torch.argmax(x).item()] for x in cnn_predictions]
    guards_predictions = [guards[torch.argmax(x).item()] for x in guards_predictions]
    predicted_state = torch.argmax(states_probs).item()
    prediction = 1 if predicted_state == num_states - 1 else 0
    show = f'{input_predictions}\n{symbolic_sequence.tolist()}\n{guards_predictions}\n{prediction} {label.item()}'
    print(show + '\n')


def pairwise_distance_penalty(predictions):
    # Predictions should be of shape [sample_size, num_classes]
    # Compute pairwise cosine similarity and penalize minimal distances
    similarity_matrix = torch.matmul(predictions, predictions.T)
    # Mask out self-similarities by setting the diagonal to zero
    sample_size = predictions.shape[0]
    mask = torch.eye(sample_size, device=predictions.device)
    similarity_matrix = similarity_matrix.masked_fill_(mask == 1, 0)

    # Maximize the minimal pairwise similarity (or minimize negative value)
    penalty = -similarity_matrix.sum() / (sample_size * (sample_size - 1))
    return penalty


def diversity_penalty(predictions):
    # Predictions should be of shape [sample_size, num_classes]
    # Compute the standard deviation across the batch for each class
    diversity_score = predictions.std(dim=0).mean()
    # Return the negative diversity score as the penalty
    return -diversity_score


def to_asal():
    def process_dataset(loader: DataLoader, file):
        seqs = []
        for seq_id, (_, label, symbolic_sequence) in enumerate(loader):
            symbolic_sequence = [x.item() for x in symbolic_sequence]
            label = label.item()
            asp_seq = seq_to_asp(seq_id, symbolic_sequence, label)
            seqs.append(asp_seq)
            with open(file, "w") as f:
                for s in seqs:
                    f.write(s)
                    f.write('\n')

    def seq_to_asp(seq_id: int, seq: list, label: str):
        x = [f'seq({seq_id},d({j}),{i}).' for (i, j) in enumerate(seq)]
        x.append(f'class({seq_id},{label}).')
        return ' '.join(x)

    path = '/home/nkatz/Downloads/asal/data/mnist/folds/fold_0'
    train_loader, test_loader = get_data_loaders()
    process_dataset(train_loader, f'{path}/train.csv')
    process_dataset(test_loader, f'{path}/test.csv')


def reshape_batch(batch_size, images, labels, symb_seqs):
    data_triplets = []
    for i in range(batch_size):
        image_seq = [tensor[i] for tensor in images]
        symbolic_seq = [torch.tensor([tensor[i]]) for tensor in symb_seqs]
        label = labels[i]
        data_triplets.append((image_seq, label, symbolic_seq))
    return data_triplets


def process_sequence(data, model, criterion, num_states):
    sequence, label, symbolic_sequence = data[0], data[1], data[2]
    cnn_predictions, guards_predictions, final_states_distribution = model([s.unsqueeze(1) for s in sequence])
    acceptance_probability = final_states_distribution[-1].unsqueeze(0)
    acceptance_probability = torch.clamp(acceptance_probability, 0, 1)
    loss = criterion(acceptance_probability, label.unsqueeze(0).float())
    # Collect stats for training F1
    predicted_state = torch.argmax(final_states_distribution).item()
    prediction = 1 if predicted_state == num_states - 1 else 0
    actual, predicted = label.item(), prediction
    return loss, actual, predicted


def process_sequences(data, model, criterion, num_states):
    sequence, label, symbolic_sequence = data[0], data[1], data[2]
    sequence, label, symbolic_sequence = sequence.to(device), label.to(device), symbolic_sequence.to(device)
    cnn_prediction, guard_prediction, final_states_distribution = model(sequence)
    acceptance_probability = final_states_distribution[:, num_states-1]

    # print(f'Acceptance probability: {acceptance_probability}')
    acceptance_probability = torch.clamp(acceptance_probability, 0, 1)

    loss = criterion(acceptance_probability, label.float())
    # Collect stats for training F1
    predicted = (acceptance_probability >= 0.5)

    return loss, label, predicted


def process_batch(batch, batch_size, model, criterion, num_states):
    seqs, labels, symb_seqs = batch[0], batch[1], batch[2]
    batch_sequences = reshape_batch(batch_size, seqs, labels, symb_seqs)
    batch_loss, actual, predicted = torch.zeros(1), [], []
    for s in batch_sequences:
        loss, a, p = process_sequence(s, model, criterion, num_states)
        batch_loss += loss
        actual.append(a)
        predicted.append(p)
    return batch_loss / batch_size, actual, predicted


def backprop(batch_loss, optimizer):
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # count_instances()
    to_asal()
