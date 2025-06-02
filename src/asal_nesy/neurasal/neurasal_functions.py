import random
import torch
from torch import nn as nn
import time
import src.asal_nesy.deepfa.automaton
from src.asal_nesy.device import device
import nnf
from itertools import chain
from src.asal_nesy.neurasal.data_structs import TensorSequence, IndividualImageDataset, SequenceDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from src.asal.template import Template
from src.logger import *
from src.asal.asal import Asal
from src.asal.auxils import get_train_data
from src.asal.asp import get_induction_program
from typing import Dict, List
from collections import defaultdict
from copy import deepcopy
import tempfile


def create_labelling_function(weights: dict, sfa_symbols: list[str]):
    def labelling_function(var: nnf.Var) -> torch.Tensor:
        if str(var.name) in sfa_symbols:
            return (
                weights[str(var.name)]
                if var.true
                else torch.ones_like(weights[str(var.name)])
            )
        return weights[str(var.name)] if var.true else 1 - weights[str(var.name)]

    return labelling_function


def nn_forward_pass(train_loader, model, cnn_output_size):
    for batch in train_loader:
        training_tensors = torch.stack([
            seq.get_sequence_tensor() for seq in batch.sequences
        ]).to(device)  # (BS, SeqLen, C, H, W)

        bs, seqlen, c, w, h = training_tensors.shape
        training_tensors = training_tensors.view(-1, c, w, h)  # Flatten for CNN into (BS * SeqLen, C, W, H)

        # Forward pass through CNN
        nn_outputs = model(training_tensors, apply_softmax=True)  # (BS * SeqLen, NumClasses)
        nn_outputs = nn_outputs.view(bs, seqlen, cnn_output_size)  # (BS, SeqLen, NumClasses)
        latent_predictions = torch.argmax(nn_outputs, dim=2).cpu()  # (BS, SeqLen)
        for i, seq in enumerate(batch.sequences):
            seq.predicted_trace = latent_predictions[i].tolist()


"""
def nesy_forward_pass(batch, model, sfa, cnn_output_size, with_decay=False):
    training_tensors = torch.stack([
        seq.get_sequence_tensor() for seq in batch
    ]).to(device)  # (BS, SeqLen, C, H, W)

    bs, seqlen, c, w, h = training_tensors.shape
    training_tensors = training_tensors.view(-1, c, w, h)  # Flatten for CNN into (BS * SeqLen, C, W, H)

    # No! (much slower than view())
    # img_sequences = torch.stack([
    #    seq.get_image(i)
    #    for seq in batch.sequences
    #    for i in range(len(seq.image_tensors))
    # ])

    # Forward pass through CNN
    nn_outputs = model(training_tensors, apply_softmax=True)  # (BS * SeqLen, NumClasses)
    nn_outputs = nn_outputs.view(bs, seqlen, cnn_output_size)  # (BS, SeqLen, NumClasses)

    # dim=2 because nn_outputs is a (BS, SeqLen, num_classes)-tensor and we need to reduce over num_classes
    latent_predictions = torch.argmax(nn_outputs, dim=2).flatten().cpu()  # (BS * SeqLen,)

    output_transposed = nn_outputs.transpose(1, 2)  # Transpose output to pass to the SFA
    probabilities = {sfa.symbols[i]: output_transposed[:, i, :] for i in range(len(sfa.symbols))}  # Prob. dict. for SFA
    labelling_function = create_labelling_function(probabilities, sfa.symbols)
    acceptance_probabilities = torch.clamp(sfa.forward(labelling_function), 0, 1)

    if with_decay:
        decay_factor = 0.989
        seq_lengths = torch.full((bs,), seqlen, dtype=torch.float32)
        # Apply decay to acceptance probabilities based on sequence length
        decay_weights = torch.pow(decay_factor, seq_lengths).to(device)
        acceptance_probabilities = acceptance_probabilities * decay_weights

    # Entropy Calculation
    eps = 1e-10  # Avoid log(0) issues
    entropy_per_img = -torch.sum(nn_outputs * torch.log(nn_outputs + eps), dim=2)  # (BS, SeqLen)

    return acceptance_probabilities, latent_predictions, nn_outputs
"""


def nesy_forward_pass(batch, model, sfa, cnn_output_size, with_decay=False, update_seqs_stats=False):
    training_tensors = torch.stack([seq.images for seq in batch]).to(device)  # (bs, seqlen, dim, c, h, w)
    bs, seqlen, dim, c, w, h = training_tensors.shape
    training_tensors = training_tensors.view(-1, c, w, h)  # Flatten for CNN into (BS * SeqLen * dim, C, W, H)
    logits = model(training_tensors, apply_softmax=False)  # (BS * SeqLen * Dim, NumClasses)
    logits = logits.view(bs, seqlen, dim, cnn_output_size)  # (BS, SeqLen, Dim, NumClasses)

    # dim=-1 refers to the last dimension of the tensor, more generic and orbust than explicitly
    # giving the dimension, especially if tensor shapes slightly change at some point. This will
    # work as long as the class predictions are the last dimension.
    nn_probs = torch.softmax(logits, dim=-1)

    probabilities = get_probs_dict(nn_probs, sfa.symbols)
    labelling_function = create_labelling_function(probabilities, sfa.symbols)
    acceptance_probabilities = torch.clamp(sfa.forward(labelling_function), 0, 1)

    """
    if with_decay:
        decay_factor = 0.989
        seq_lengths = torch.full((bs,), seqlen, dtype=torch.float32)
        # Apply decay to acceptance probabilities based on sequence length
        decay_weights = torch.pow(decay_factor, seq_lengths).to(device)
        acceptance_probabilities = acceptance_probabilities * decay_weights

    # Entropy Calculation
    eps = 1e-10  # Avoid log(0) issues
    entropy_per_img = -torch.sum(nn_outputs * torch.log(nn_outputs + eps), dim=2)  # (BS, SeqLen)
    """

    # dim=3 because nn_outputs is a (BS, SeqLen, Dim, num_classes)-tensor and we need to reduce over num_classes
    latent_predictions = torch.argmax(nn_probs, dim=3).flatten().cpu()  # (BS * SeqLen * Dim,)

    if update_seqs_stats:
        for sequence, p in zip(batch, acceptance_probabilities):
            sequence.acceptance_probability = p

    return acceptance_probabilities, latent_predictions, logits


def get_probs_dict(nn_outputs: torch.Tensor, symbols: List[str]) -> Dict[str, torch.Tensor]:
    """
        Converts CNN output into a dictionary mapping each symbol to a (BatchSize, SeqLen) tensor of probabilities.

        Args:
            nn_outputs (torch.Tensor): Tensor of shape (BatchSize, SeqLen, Dim, NumClasses)
            symbols (List[str]): List of D * C symbols in order (e.g., ['d1_0', ..., 'd3_9'])

        Returns:
            Dict[str, torch.Tensor]: Mapping from symbol to tensor of shape (BatchSize, SeqLen)
        """
    B, T, D, C = nn_outputs.shape
    output_transposed = nn_outputs.permute(2, 3, 0, 1)  # (D, C, B, T)
    return {
        symbols[d * C + c]: output_transposed[d, c]
        for d in range(D) for c in range(C)
    }


def get_latent_loss(batch, nn_outputs, nn_criterion, class_attributes):
    labeled_points_predictions, labeled_points_labels = [], []
    latent_loss = torch.tensor(0.0)
    for i, seq in enumerate(batch):
        # seq.set_image_label(1, 2)
        labeled_indices = seq.get_labeled_indices()
        # print(labeled_indices)
        labeled_points_predictions.extend([nn_outputs[i][j][k] for j, k in labeled_indices])
        labeled_points_labels.extend([
            torch.tensor(seq.get_image_label(j, k, class_attributes)).long() for j, k in labeled_indices
        ])

    if labeled_points_labels:
        labeled_points_predictions = torch.stack(labeled_points_predictions).to(device)
        labels_tensor = torch.cat(labeled_points_labels).to(device)
        latent_loss = nn_criterion(labeled_points_predictions, labels_tensor)

    return latent_loss


class StatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.target_loss = 0.0
        self.latent_loss = 0.0
        self.seq_actual = []
        self.seq_predicted = []
        self.latent_actual = []
        self.latent_predicted = []

    def update_stats(self, batch, latent_predictions, seq_predictions,
                     class_attributes, target_loss=0.0, latent_loss=0.0):
        self.seq_predicted.extend([s.item() for s in seq_predictions])
        self.seq_actual.extend([seq.seq_label for seq in batch])
        self.latent_predicted.extend([s.item() for s in latent_predictions])
        self.latent_actual.extend(list(chain(*[seq.get_image_labels(class_attributes) for seq in batch])))
        self.target_loss += target_loss
        self.latent_loss += latent_loss


def eval_model(ts: StatsCollector,
               train_loader: DataLoader[SequenceDataset],
               test_loader: DataLoader[SequenceDataset],
               model: torch.nn.Module,
               sfa_dnnf: src.asal_nesy.deepfa.automaton.DeepFA,
               cnn_output_size,
               class_attributes,
               epoch_num=None,
               show_log=True):
    def get_score(ts: StatsCollector):
        latent_f1_macro = f1_score(ts.latent_actual, ts.latent_predicted, average="macro")
        f1, tps, fps, fns = get_sequence_stats(ts.seq_predicted, ts.seq_actual)
        return f1, tps, fps, fns, latent_f1_macro

    end_time = time.time()

    # Training stats:
    train_f1, train_tps, train_fps, train_fns, train_latent_f1 = get_score(ts)

    # Evaluate on the test set:
    sc = StatsCollector()
    for batch in test_loader:
        acceptance_probabilities, latent_predictions, nn_outputs = (
            nesy_forward_pass(batch, model, sfa_dnnf, cnn_output_size)
        )
        sequence_predictions = (acceptance_probabilities >= 0.5)
        sc.update_stats(batch, latent_predictions, sequence_predictions, class_attributes)

    # Testing stats:
    test_f1, test_tps, test_fps, test_fns, test_latent_f1 = get_score(sc)

    epoch = f'Epoch {epoch_num}' if epoch_num is not None else ''
    seq_loss = ts.target_loss / len(train_loader)
    latent_loss = ts.latent_loss / len(train_loader)

    if show_log:
        loss_msg = f'{(seq_loss + latent_loss):.3f} (seq: {seq_loss:.3f} | latent: {latent_loss:.3f})' if latent_loss > 0 else f'{seq_loss:.3f}'
        logger.info(
            f'{epoch}\nLoss: {loss_msg}, Time: {end_time - ts.start_time:.3f} secs\n'
            f'Train F1: {train_f1:.3f} ({train_tps}, {train_fps}, {train_fns}) | latent: {train_latent_f1:.3f}\n'
            f'Test F1: {test_f1:.3f} ({test_tps}, {test_fps}, {test_fns}) | latent: {test_latent_f1:.3f}')
        # f'Labeled images so far: {len(labeled_images)}')

    model.train()
    return test_f1, test_latent_f1


def get_sequence_stats(predicted, actual):
    predicted = [int(p) for p in predicted]  # Ensure tensors or non-lists are converted to flat integer lists
    actual = [int(a) for a in actual]

    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1_binary = f1_score(actual, predicted, average='binary', zero_division=0)  # standard F1 for binary classification

    f1_macro = f1_score(actual, predicted, average='macro')  # 1/2(F1_pos_class + F1_neg_class)
    f1_weighted = f1_score(actual, predicted, average='weighted')

    # Confusion matrix: [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()

    return f1_binary, tp, fp, fn


def pretrain_nn(train_data: SequenceDataset,
                test_data: SequenceDataset,
                num_samples,
                model,
                optimizer,
                class_attributes,
                with_fully_labelled_seqs=False,
                num_epochs=10):
    def get_image_dataset(seq_list: list[TensorSequence],
                          batch_size, shuffle=True) -> DataLoader[IndividualImageDataset]:
        train_images, train_labels = [], []
        for seq in seq_list:
            images = [seq.get_image(i, j) for i, j in seq.get_image_indices()]
            labels = [seq.get_image_label(i, j, class_attributes) for i, j in seq.get_image_indices()]
            labels_flat = list(chain(*labels))
            labels = torch.tensor(labels_flat).long()

            train_images.extend(images)
            train_labels.extend(labels)

        train_dataset = IndividualImageDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader

    sample_sequences = random.sample(train_data.sequences, num_samples) if not with_fully_labelled_seqs else train_data
    for seq in sample_sequences:
        for idx in seq.get_image_indices():
            seq.set_image_label(idx[0], idx[1])  # use these labels downstream

    train_loader = get_image_dataset(sample_sequences, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, apply_softmax=False)  # Get logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    test_loader = get_image_dataset(test_data.sequences, batch_size=32, shuffle=False)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, apply_softmax=True)  # Get probabilities

            _, predictions = torch.max(outputs, 1)  # Get class with highest probability

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

        test_f1 = f1_score(y_true, y_pred, average="micro")  # average="micro"
        logger.info(f'Pre-trained model F1-score on test set: {test_f1:.3f}')

    model.train()  # just to be on the safe side.


def sequence_to_facts(sequence: TensorSequence):
    facts = []
    labeled_points = sequence.get_labeled_indices()
    for t, digit in enumerate(sequence.predicted_trace):
        if t in labeled_points:
            facts.append(f'seq({sequence.seq_id},d({sequence.image_labels[t]}),{t}).')
        else:
            facts.append(f'seq({sequence.seq_id},d({digit}),{t}).')
    facts.append(f'class({sequence.seq_id},{sequence.seq_label}).')
    # facts.append(f'weight({seq_id},{seq_entropy}).')
    return ' '.join(facts)


def induce_sfa_simple(args, asp_compilation_program, vars, data=None, existing_sfa=None):
    shuffle = False
    template = Template(args.states, args.tclass)
    train_data = get_train_data(args.train, str(args.tclass), args.batch_size, shuffle=shuffle)

    logger.debug(f'The induction program is:\n{get_induction_program(args, template)}')

    """
    mcts = Asal(args, train_data, template)
    mcts.run_mcts()

    logger.info(blue(f'New SFA:\n{mcts.best_model.show(mode="simple")}\n'
                     f'training F1-score: {mcts.best_model.global_performance} '
                     f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})'))

    logger.info('Compiling guards into NNF...')
    sfa = mcts.best_model
    """

    asal = Asal(args, train_data, template, initialize_only=True)
    models = asal.run(train_data)
    # sfa = asal.run(train_data)
    logger.info(f'Compiling guards into NNF for {len(models)} SFA...')
    from src.asal_nesy.neurasal.sfa import compile_sfa
    result = []
    for sfa in models:
        compiled = compile_sfa(sfa, asp_compilation_program, vars)
        result.append((compiled, sfa))

    return result


def set_all_labelled(batch: list[TensorSequence]):
    """Set all training sequences as labelled. Used for debugging."""
    for s in batch:
        for t in range(s.seq_length):
            for d in range(s.dimensionality):
                if not s.is_labeled(t, d):
                    s.set_image_label(t, d)


""" 
--------------------------------------------------------------------------------------------------
----------------------------- Active Learning Functions: -----------------------------------------
--------------------------------------------------------------------------------------------------
"""


def initialize_fully_labeled_seqs(train_loader, n):
    pos_seqs = []
    neg_seqs = []
    for batch in train_loader:
        for seq in batch:
            if seq.seq_label == 1 and len(pos_seqs) < n // 2:
                pos_seqs.append(seq.seq_id)
            elif seq.seq_label == 0 and len(neg_seqs) < n // 2:
                neg_seqs.append(seq.seq_id)
        if len(pos_seqs) >= n // 2 and len(neg_seqs) >= n // 2:
            break
    return pos_seqs + neg_seqs


def get_misclassified_seqs(model, train_loader, sfa_dnnf, cnn_output_size):
    model.eval()
    misclassified_seqs = []
    with torch.no_grad():
        for batch in train_loader:
            acceptance_probs, _, _ = nesy_forward_pass(batch, model, sfa_dnnf, cnn_output_size)
            predicted = [1 if p >= 0.5 else 0 for p in acceptance_probs]
            actual = [seq.seq_label for seq in batch]
            misclassified = [s for i, s in enumerate(batch) if predicted[i] != actual[i]]
            misclassified_seqs.extend(misclassified)
    return misclassified_seqs


def compute_expected_acceptance_losses(model, data_loader, sfa_dnnf, cnn_output_size):
    """
    Compute the expected acceptance loss (BCE between acceptance prob and seq label)
    for each sequence in the data loader.

    Returns
    -------
    Dict mapping sequence ID to BCE loss
    """
    model.eval()
    loss_dict = {}
    with torch.no_grad():
        for batch in data_loader:
            acceptance_probs, _, _ = nesy_forward_pass(batch, model, sfa_dnnf, cnn_output_size)
            sequence_labels = torch.tensor([seq.seq_label for seq in batch]).to(device)
            losses = nn.functional.binary_cross_entropy(acceptance_probs, sequence_labels.float(), reduction='none')
            for seq, loss in zip(batch, losses):
                loss_dict[seq.seq_id] = loss.item()
    return loss_dict


def compute_elr_score(candidate_seq, model, cnn_output_size, train_loader,
                      cnn_criterion, sequence_criterion, asal_args, asp_comp_program,
                      lr, symb_vars, seq_loss_weight, N_samples,
                      fully_labelled_seqs, current_loss, num_epochs=3):
    # Monte Carlo approximation of expected loss after labeling the candidate sequence
    mc_losses = []
    for _ in range(N_samples):
        model_copy = deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Sample a symbolic sequence from CNN output
        symb_seq = candidate_seq.sample_symbolic_sequence(model_copy)
        symb_seqs = fully_labelled_seqs + [symb_seq]

        sfa_dnnf_new, sfa_asal_new = induce_sfa(symb_seqs, asal_args, asp_comp_program, symb_vars)[-1]

        nesy_train(model_copy, train_loader, sfa_dnnf_new, cnn_output_size,
                   cnn_criterion, sequence_criterion, optimizer, num_epochs,
                   seq_loss_weight, update_seqs_stats=False)

        loss_after = 0.0
        with torch.no_grad():
            for batch in train_loader:
                acceptance_probs, _, _ = nesy_forward_pass(batch, model_copy, sfa_dnnf_new, cnn_output_size)
                labels = torch.tensor([seq.seq_label for seq in batch]).to(device)
                loss_after += sequence_criterion(acceptance_probs, labels.float()).item() * len(batch)
        mc_losses.append(loss_after)

    expected_loss = sum(mc_losses) / len(mc_losses)
    candidate_seq.elr_score = current_loss - expected_loss
    return current_loss - expected_loss


def induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs):
    input_data = '\n'.join(symb_seqs)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(input_data)
    try:
        asal_args.train = tmp_path
        results = induce_sfa_simple(asal_args, asp_comp_program, class_attrs)
    finally:
        os.remove(tmp_path)  # Always delete the file
    return results


def nesy_train(model, train_loader, sfa_dnnf, cnn_output_size,
               nn_criterion, sequence_criterion, optimizer, num_epochs,
               seq_loss_weight, class_attrs, test_loader, update_seqs_stats=True, show_log=True):
    test_stats = defaultdict(list)
    model.train()
    optimizer.zero_grad()
    training_stats = []
    for epoch in range(num_epochs):
        stats_collector = StatsCollector()
        for batch in train_loader:
            acceptance_probabilities, latent_predictions, nn_outputs = (
                nesy_forward_pass(batch, model, sfa_dnnf,
                                  cnn_output_size, update_seqs_stats=update_seqs_stats)
            )
            sequence_labels = torch.tensor([seq.seq_label for seq in batch]).to(device)  # (BS,)
            seq_loss = sequence_criterion(acceptance_probabilities, sequence_labels.float())
            latent_loss = get_latent_loss(batch, nn_outputs, nn_criterion, class_attrs)
            total_loss = (seq_loss_weight * seq_loss) + (latent_loss * seq_loss_weight)
            # total_loss = seq_loss
            # total_loss = latent_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sequence_predictions = (acceptance_probabilities >= 0.5)
            stats_collector.update_stats(batch, latent_predictions, sequence_predictions,
                                         class_attrs, seq_loss.item(), latent_loss.item())
        training_stats.append(stats_collector)
        seq_f1, img_f1 = eval_model(stats_collector, train_loader, test_loader, model, sfa_dnnf,
                                    cnn_output_size, class_attrs, epoch, show_log=show_log)
        test_stats['seq_f1'].append(seq_f1)
        test_stats['img_f1'].append(img_f1)
    return test_stats, training_stats[-1]  # get the stats for the last epoch


def al_expected_acceptance_loss(model, train_loader, test_loader, train_data, sfa_dnnf, al_queries,
                                num_epochs, fully_labelled_seqs, nn_criterion, sequence_criterion,
                                optimizer, seq_loss_weight, cnn_output_size, asal_args, asp_comp_program,
                                class_attrs, show_stats=False):
    labeled_ids = set(seq.seq_id for seq in fully_labelled_seqs)
    history = defaultdict(list)

    for query in range(al_queries):
        misclassified_seqs = get_misclassified_seqs(model, train_loader, sfa_dnnf, cnn_output_size)
        unlabeled_candidates = [seq for seq in train_data if seq.seq_id not in labeled_ids]
        #unlabeled_candidates = [seq for seq in misclassified_seqs if seq.seq_id not in labeled_ids]
        unlabeled_ids = set(seq.seq_id for seq in unlabeled_candidates)

        losses = compute_expected_acceptance_losses(model, train_loader, sfa_dnnf, cnn_output_size)

        filtered_losses = {sid: loss for sid, loss in losses.items() if sid in unlabeled_ids}

        if not filtered_losses:
            print("No more unlabeled sequences to choose from.")
            sys.exit(-1)

        best_seq_id = max(filtered_losses, key=filtered_losses.get)
        best_seq = next(seq for seq in unlabeled_candidates if seq.seq_id == best_seq_id)

        best_seq.mark_seq_as_fully_labelled()
        fully_labelled_seqs.append(best_seq)
        labeled_ids.add(best_seq.seq_id)

        print([s.seq_id for s in fully_labelled_seqs])

        symb_seqs = [s.get_symbolic_seq() for s in fully_labelled_seqs]
        sfa_dnnf, sfa_asal = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)[-1]

        stats, _ = nesy_train(model, train_loader, sfa_dnnf, cnn_output_size,
                              nn_criterion, sequence_criterion, optimizer,
                              num_epochs, seq_loss_weight, class_attrs, test_loader, show_log=show_stats)
        # history['seq_f1'].append(stats['seq_f1'][-1])
        # history['img_f1'].append(stats['img_f1'][-1])

        history['seq_f1'].extend(stats['seq_f1'])
        history['img_f1'].extend(stats['img_f1'])

    return history


def al_random_sampling(train_data, train_loader, test_loader, init_cnn, al_queries, num_epochs, fully_labelled_seqs,
                       cnn_output_size, asal_args, asp_comp_program, class_attrs,
                       nn_criterion, sequence_criterion, optimizer, seq_loss_weight,
                       misclassified_seqs=None, show_stats=False):
    """
    Parameters
    ----------
    misclassified_seqs :
    seq_loss_weight :
    optimizer :
    num_epochs :
    sequence_criterion :
    nn_criterion :
    init_sfa : tuple (sfa_dnnf, sfa_asal) - initial SFA from seed sequences
    init_cnn : pre-trained DigitCNN model
    al_queries : int - number of active learning queries (random selection)
    fully_labelled_seqs : list of initially fully labeled sequences

    Returns
    -------
    history : dict with sequence and image F1s per query
    """
    import random

    # model = deepcopy(init_cnn)
    model = init_cnn
    labeled_ids = set(seq.seq_id for seq in fully_labelled_seqs)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # sequence_criterion = nn.BCELoss()
    # nn_criterion = nn.CrossEntropyLoss()

    history = defaultdict(list)

    for query in range(al_queries):
        # Randomly sample an unlabeled sequence
        candidates = (
            [seq for seq in train_data if seq.seq_id not in labeled_ids]
            if misclassified_seqs is None
            else misclassified_seqs
        )
        new_seq = random.choice(candidates)
        new_seq.mark_seq_as_fully_labelled()
        fully_labelled_seqs.append(new_seq)
        labeled_ids.add(new_seq.seq_id)

        print([s.seq_id for s in fully_labelled_seqs])

        symb_seqs = [s.get_symbolic_seq() for s in fully_labelled_seqs]
        sfa_dnnf, sfa_asal = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)[-1]

        stats, _ = nesy_train(model, train_loader, sfa_dnnf, cnn_output_size,
                              nn_criterion, sequence_criterion, optimizer, num_epochs,
                              seq_loss_weight, class_attrs, test_loader, show_log=show_stats)

        history['seq_f1'].extend(stats['seq_f1'])
        history['img_f1'].extend(stats['img_f1'])

    return history
