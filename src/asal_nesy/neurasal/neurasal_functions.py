import random

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import time
import src.asal_nesy.deepfa.automaton
from src.globals import device, weight_sequence_loss
import nnf
from itertools import chain
from src.asal_nesy.neurasal.data_structs import TensorSequence, IndividualImageDataset, SequenceDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from src.asal.template import Template
from src.logger import *
from src.asal.asal import Asal
from src.asal.auxils import get_train_data
from src.asal.asp import get_induction_program, get_interpreter, get_domain
from typing import Dict, List
from collections import defaultdict
from copy import deepcopy
import tempfile
from src.asal.structs import Automaton
import clingo
from clingo.script import enable_python
import multiprocessing


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if func.__name__ == "train":
            logger.info(yellow(f"NeSy {func.__name__} took {end - start:.4f} seconds"))
        return result

    return wrapper


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

"""
def get_nn_predicted_seqs(batch, model, cnn_output_size):
    # This uses the argmax indexes as the predicted classes, only works for MNIST...
    predictions_dict = batch[0].class_prediction_dict  # same for all seqs
    training_tensors = torch.stack([seq.images for seq in batch]).to(device)  # (bs, seqlen, dim, c, h, w)
    bs, seqlen, dim, c, w, h = training_tensors.shape
    training_tensors = training_tensors.view(-1, c, w, h)  # Flatten for CNN into (BS * SeqLen * dim, C, W, H)

    model.eval()
    with torch.no_grad():
        logits = model(training_tensors, apply_softmax=False)  # (BS * SeqLen * Dim, NumClasses)
        logits = logits.view(bs, seqlen, dim, cnn_output_size)  # (BS, SeqLen, Dim, NumClasses)
        nn_probs = torch.softmax(logits, dim=-1).cpu()
        latent_predictions = torch.argmax(nn_probs, dim=3).cpu()  # shape: (BS, SeqLen, Dim)

    predicted_symbolic_seqs = [
        latent_predictions[i].numpy()  # shape: (SeqLen, Dim)
        for i in range(bs)
    ]

    predicted_softmwxed_seqs = [
        nn_probs[i].numpy()
        for i in range(bs)
    ]

    model.train()
    return predicted_symbolic_seqs, predicted_softmwxed_seqs
"""

#"""
def get_nn_predicted_seqs(batch, model, cnn_output_size):

    """This uses a fixed class prediction order (consistent between the circuits and the NN) to match the
    argmax indexes with actual class labels."""

    predictions_dict = batch[0].class_prediction_dict  # same for whole batch

    training_tensors = torch.stack([seq.images for seq in batch]).to(device)  # (BS, SeqLen, Dim, C, H, W)
    bs, seqlen, dim, c, w, h = training_tensors.shape
    training_tensors = training_tensors.view(-1, c, w, h)  # (BS * SeqLen * Dim, C, W, H)

    model.eval()
    with torch.no_grad():
        logits = model(training_tensors, apply_softmax=False)  # (BS * SeqLen * Dim, NumClasses)
        logits = logits.view(bs, seqlen, dim, cnn_output_size)
        nn_probs = torch.softmax(logits, dim=-1).cpu()
        latent_predictions = torch.argmax(nn_probs, dim=3).cpu()  # (BS, SeqLen, Dim)

    predicted_symbolic_seqs = []
    for i in range(bs):
        seq_preds = []
        for t in range(seqlen):
            point_preds = {}
            for d in range(dim):
                # Determine the attribute name for this dimension
                attr = list(predictions_dict.keys())[d]
                class_order = predictions_dict[attr]

                # Get the argmax index
                idx = latent_predictions[i, t, d].item()

                # Map to the class name, split and extract actual symbol
                class_name = class_order[idx]  # e.g., 'd1_7'
                symbol = class_name.split('_')[1]  # e.g., '7'

                point_preds[attr] = int(symbol)
            seq_preds.append(point_preds)
        predicted_symbolic_seqs.append(seq_preds)

    predicted_symbolic_seqs = [
        [
            [point[attr] for attr in list(seq_points[0].keys())]
            for point in seq_points
        ]
        for seq_points in predicted_symbolic_seqs
    ]

    predicted_softmaxed_seqs = [
        nn_probs[i].numpy()
        for i in range(bs)
    ]

    model.train()
    return predicted_symbolic_seqs, predicted_softmaxed_seqs
#"""

def nesy_forward_pass(batch, model, sfa, cnn_output_size, with_decay=False, update_seqs_stats=True, max_propagation=False):
    training_tensors = torch.stack([seq.images for seq in batch]).to(device)  # (bs, seqlen, dim, c, h, w)
    bs, seqlen, dim, c, w, h = training_tensors.shape
    training_tensors = training_tensors.view(-1, c, w, h)  # Flatten for CNN into (BS * SeqLen * dim, C, W, H)
    logits = model(training_tensors, apply_softmax=False)  # (BS * SeqLen * Dim, NumClasses)
    logits = logits.view(bs, seqlen, dim, cnn_output_size)  # (BS, SeqLen, Dim, NumClasses)

    # dim=-1 refers to the last dimension of the tensor, more generic and robust than explicitly
    # giving the dimension, especially if tensor shapes slightly change at some point. This will
    # work as long as the class predictions are the last dimension.
    nn_probs = torch.softmax(logits/5, dim=-1)

    probabilities = get_probs_dict(nn_probs, sfa.symbols)
    labelling_function = create_labelling_function(probabilities, sfa.symbols)

    if not max_propagation:
        acceptance_probabilities = torch.clamp(sfa.forward(labelling_function), 0, 1)
    else:
        acceptance_probabilities = torch.clamp(
            sfa.forward(labelling_function, max_propagation=True, return_accepting=True), 0, 1)


    #"""
    if with_decay:
        decay_factor = 0.978
        seq_lengths = torch.full((bs,), seqlen, dtype=torch.float32)
        # Apply decay to acceptance probabilities based on sequence length
        decay_weights = torch.pow(decay_factor, seq_lengths).to(device)
        acceptance_probabilities = acceptance_probabilities * decay_weights
    #"""

    """
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


def compute_seq_probs(model, batch_images):
    B = len(batch_images)
    T, D, C, H, W = batch_images[0].images.shape

    # Reshape to (B * T * D, C, H, W)
    flat_images = torch.cat(
        [seq.images.view(-1, C, H, W) for seq in batch_images], dim=0
    ).to(next(model.parameters()).device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(flat_images)  # (B * T * D, num_classes)
        probs = F.softmax(logits, dim=1)  # (B * T * D, num_classes)
        argmax_probs = probs.max(dim=1).values  # (B * T * D,)

    # Reshape to (B, T, D) and compute product over T and D
    argmax_probs = argmax_probs.view(B, T, D)
    # seq_probs = argmax_probs.prod(dim=(1, 2))  # (B,)
    seq_probs = argmax_probs.prod(dim=2).prod(dim=1)

    """
    seq_prob_dict = {
        seq.seq_id: prob.item()
        for seq, prob in zip(batch_images, seq_probs)
    }
    """

    model.train()
    # return seq_prob_dict
    return seq_probs


def set_asp_weights(unlabelled_seqs, fully_labelled_seqs, max_unlabelled_weight=100):
    import numpy as np
    hard_weight = 10000000000
    probs = [seq.sequence_probability for seq in unlabelled_seqs]

    # To prevent tiny floating point values from becoming indistinguishably small when cast to integers,
    # apply log transformation (which spreads out small numbers):
    log_probs = [np.log(p + 1e-12) for p in probs]

    # We want larger weights for more confident sequences (i.e., higher original probability → higher weight), so we:
    # Normalize to [0, 1]:
    min_lp, max_lp = min(log_probs), max(log_probs)
    norm_weights = [(lp - min_lp) / (max_lp - min_lp + 1e-12) for lp in log_probs]

    # Inverting the log_probs will give higher weight to most uncertain seqs. Try it to see that we're getting
    # out huge, crappy, overfitted SFAs...
    # inv_log_probs = [-lp for lp in log_probs]
    # min_lp, max_lp = min(inv_log_probs), max(inv_log_probs)

    # scale the normalized soft weights into [1, max_unlabelled_weight]
    int_weights = [int(w * max_unlabelled_weight) + 1 for w in norm_weights]

    """
    return {
        seq.seq_id: weight
        for seq, weight in zip(unlabelled_seqs, int_weights)
    }, hard_weight
    """

    for s, w in zip(unlabelled_seqs, int_weights):
        s.asp_weight = w

    for s in fully_labelled_seqs:
        s.asp_weight = hard_weight


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
    # latent_loss = torch.tensor(0.0)
    latent_loss = 0.0 * nn_outputs.sum()  # use this to avoid issues with points without requires_grad=True
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


def test_model_max_prop(ts: StatsCollector,
                        train_loader: DataLoader[SequenceDataset],
                        test_loader: DataLoader[SequenceDataset],
                        model: torch.nn.Module,
                        sfa: src.asal_nesy.deepfa.automaton.DeepFA,
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
    model.eval()
    sc = StatsCollector()
    with torch.no_grad():
        for batch in test_loader:
            testing_tensors = torch.stack([seq.images for seq in batch]).to(device)  # (bs, seqlen, dim, c, h, w)
            bs, seqlen, dim, c, w, h = testing_tensors.shape
            testing_tensors = testing_tensors.view(-1, c, w, h)  # Flatten for CNN into (BS * SeqLen * dim, C, W, H)
            logits = model(testing_tensors, apply_softmax=False)  # (BS * SeqLen * Dim, NumClasses)
            logits = logits.view(bs, seqlen, dim, cnn_output_size)  # (BS, SeqLen, Dim, NumClasses)

            nn_probs = torch.softmax(logits, dim=-1)

            probabilities = get_probs_dict(nn_probs, sfa.symbols)
            labelling_function = create_labelling_function(probabilities, sfa.symbols)

            acceptance_probabilities = torch.clamp(
                sfa.forward(labelling_function, max_propagation=True, return_accepting=True), 0, 1)

            rejection_probabilities = torch.clamp(
                sfa.forward(labelling_function, max_propagation=True, return_accepting=False), 0, 1)

            sequence_predictions = (acceptance_probabilities - rejection_probabilities > 0.0)

            latent_predictions = torch.argmax(nn_probs, dim=3).flatten().cpu()  # (BS * SeqLen * Dim,)
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
    model.eval()
    sc = StatsCollector()
    with torch.no_grad():
        for batch in test_loader:
            acceptance_probabilities, latent_predictions, nn_outputs = (
                nesy_forward_pass(batch, model, sfa_dnnf, cnn_output_size, update_seqs_stats=False, with_decay=False)  # Set this to False in needed!
            )
            sequence_predictions = (acceptance_probabilities >= 0.99)
            sc.update_stats(batch, latent_predictions, sequence_predictions, class_attributes)

            # for seq, p in zip(batch, acceptance_probabilities):
            #     print('\n')
            #     print(p, seq.seq_label)
            #     print('\n')

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
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

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


def induce_sfa_simple(args, asp_compilation_program, vars, data=None, existing_sfa=Automaton()):
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
    models = asal.run(train_data, existing_sfa=existing_sfa)
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


class EditPoint:
    def __init__(self, attribute, value, time_point):
        self.edit_attribute = attribute
        self.value = value
        self.time_point = time_point

class CounterFact:
    def __init__(self, asp_program):
        self.edit_points = []
        self.cost = 0
        cores_num = multiprocessing.cpu_count()
        self.cores = f'-t{cores_num if cores_num <= 64 else 64}'  # 64 threads is the limit for Clingo
        self.asp_program = asp_program
        self.num_edits = 0
        self.ctl = clingo.Control([self.cores])
        self.found_model = False

    def on_model(self, model: clingo.solving.Model):
        # print(model)
        self.found_model = True
        atoms = [atom for atom in model.symbols(shown=True)]
        edit_points = []
        for atom in atoms:
            if atom.match("edit", 4):
                att = str(atom.arguments[1])
                val = str(atom.arguments[2])
                time = str(atom.arguments[3])
                edit_points.append(EditPoint(att, val, time))  # we only dee the last one
            elif atom.match("edit_count", 1):
                self.num_edits = int(str(atom.arguments[0]))

        self.edit_points = edit_points
        self.cost = model.cost[0]

    def solve(self):
        enable_python()
        self.ctl.configuration.solve.models = '0'  # all models
        self.ctl.add("base", [], self.asp_program)
        self.ctl.ground([("base", [])])
        self.ctl.solve(on_model=self.on_model)



def get_query_points_by_edit_cost(train_loader, fully_labelled_seqs, current_nesy_model, asal_args):
    if not current_nesy_model.sfa_asal.is_empty:

        labeled_ids = set(seq.seq_id for seq in fully_labelled_seqs)
        all_seqs = [seq for batch in train_loader for seq in batch]
        unlabeled_candidates = [seq for seq in all_seqs if seq.seq_id not in labeled_ids]

        # Get the misclassified - given the current SFA - sequences (these are misclassified based on acceptance prob.)
        misclassified_seqs = [
            seq for seq in unlabeled_candidates
            if (seq.seq_label == 1 and seq.acceptance_probability < 0.5)
               or (seq.seq_label == 0 and seq.acceptance_probability >= 0.5)
        ]

        counterfactuals_rules = """\n
        seqId(S) :- argmax_prediction(S,_,_).
        attribute(A) :- argmax_prediction(_,obs(A,_),_).
        attribute(A) :- prediction(_,obs(A,_),_).
        val(V) :- argmax_prediction(_,obs(_,V),_).
        val(V) :- prediction(_,obs(_,V),_).
        timeStep(T) :- argmax_prediction(_,obs(_,_),T).
        timeStep(T) :- prediction(_,obs(_,_),T).

        seq(S,obs(A,V),T) :- not_edit(S,A,V,T).
        seq(S,obs(A,V),T) :- edit(S,A,V,T). 

        {edit(S,A,V,T) : prediction(S,obs(A,V),T)}.
        {not_edit(S,A,V,T) : argmax_prediction(S,obs(A,V),T)}.

        :- edit(S,A,V,T), not_edit(S,A,V,T).
        :- edit(S,A,V1,T), edit(S,A,V2,T), V1 != V2.
        :- not seq(_,obs(A,_),T), timeStep(T), attribute(A).

        edit_count(X) :- X = #count{T,Att,Val: edit(SeqId,Att,Val,T)}.

        #minimize{W2 - W1@1: edit(SeqId,Att,Val,T), prediction_weight(SeqId,obs(Att,Val,T),W1), argmax_weight(SeqId,obs(Att,Val_1,T),W2)}.

        #show edit/4.
        #show edit_count/1.
        """

        for seq in misclassified_seqs:
            asp_facts = seq.generate_asp_prediction_facts()
            acceptance_constraint = f':- not accepted({seq.seq_id}).' if (seq.seq_label == 1 and seq.acceptance_probability < 0.5) else f':- accepted({seq.seq_id}).'
            program  = (current_nesy_model.sfa_asal.show('reasoning') + '\n' +
                        get_interpreter(asal_args) + '\n' +
                        get_domain(asal_args.domain) + '\n' +
                        acceptance_constraint + '\n' +
                        counterfactuals_rules + '\n' + asp_facts)

            counterfact_instance = CounterFact(program)
            counterfact_instance.solve()

            found_model = counterfact_instance.found_model

            if not found_model:
                logger.error(f'UNSATISFIABLE program during counterfactuals generation for sequence {seq.seq_id}. The ASP program is:\n\n{program}')
                sys.exit(1)

            seq.edit_points = counterfact_instance.edit_points
            seq.edit_cost = counterfact_instance.cost

            # This might not be very reasonable...
            seq.edit_score = seq.bce_loss * len(seq.edit_points) / (1 + seq.edit_cost) if len(seq.edit_points) > 0 else 0

            if True:  # seq.edit_points:
                print(f'Seq: {seq.seq_id} | edit points: {len(seq.edit_points)}, '
                      f'edit cost: {seq.edit_cost}, edit score: {seq.edit_score}, '
                      f'accept. prob: {seq.acceptance_probability}, label: {seq.seq_label}, BCE loss: {seq.bce_loss}')

        best_seq = max(misclassified_seqs, key=lambda x: x.edit_score)
        # best_seq = max(misclassified_seqs, key=lambda x: len(x.edit_points))
        # best_seq = max(misclassified_seqs, key=lambda x: x.edit_cost)

        logger.info(yellow(f'Best sequence: {best_seq.seq_id}, edit points: {len(best_seq.edit_points)}, '
                           f'edit cost: {best_seq.edit_cost}, edit score: {best_seq.edit_score:.3f}, '
                           f'accept. prob.: {best_seq.acceptance_probability:.3f}, label: {best_seq.seq_label}, '
                           f'BCE loss: {best_seq.bce_loss}\nedit points: {len(best_seq.edit_points)}'))

        return best_seq

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
                l = loss.item()
                loss_dict[seq.seq_id] = l
                seq.bce_loss = l
    model.train()
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


def induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs, existing_sfa=Automaton()):
    input_data = '\n'.join(symb_seqs)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(input_data)
    try:
        asal_args.train = tmp_path
        results = induce_sfa_simple(asal_args, asp_comp_program, class_attrs, existing_sfa=existing_sfa)
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
    sequence_loss_weight = 1.0
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

            if weight_sequence_loss:
                total_loss = (sequence_loss_weight * seq_loss) + latent_loss
            else:
                total_loss = seq_loss + latent_loss
                # total_loss = latent_loss  # disable the sequence loss (for experiments only)

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sequence_predictions = (acceptance_probabilities >= 0.5)
            stats_collector.update_stats(batch, latent_predictions, sequence_predictions,
                                         class_attrs, seq_loss.item(), latent_loss.item())

        training_stats.append(stats_collector)

        #"""
        seq_f1, img_f1 = eval_model(stats_collector, train_loader, test_loader, model, sfa_dnnf,
                                    cnn_output_size, class_attrs, epoch, show_log=show_log)
        #"""


        # seq_f1, img_f1 = test_model_max_prop(stats_collector, train_loader, test_loader, model,
        #                                     sfa_dnnf, cnn_output_size, class_attrs, epoch, show_log=show_log)

        test_stats['seq_f1'].append(seq_f1)
        test_stats['img_f1'].append(img_f1)

        if weight_sequence_loss:
            epoch_loss = training_stats[-1].target_loss / len(train_loader)
            sequence_loss_weight = np.exp(-epoch_loss)

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

        symb_seqs = [s.get_labelled_seq_asp() for s in fully_labelled_seqs]
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

        symb_seqs = [s.get_labelled_seq_asp() for s in fully_labelled_seqs]
        sfa_dnnf, sfa_asal = induce_sfa(symb_seqs, asal_args, asp_comp_program, class_attrs)[-1]

        stats, _ = nesy_train(model, train_loader, sfa_dnnf, cnn_output_size,
                              nn_criterion, sequence_criterion, optimizer, num_epochs,
                              seq_loss_weight, class_attrs, test_loader, show_log=show_stats)

        history['seq_f1'].extend(stats['seq_f1'])
        history['img_f1'].extend(stats['img_f1'])

    return history
