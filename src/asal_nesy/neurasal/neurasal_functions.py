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
import argparse
from src.logger import *
from src.asal.asal import Asal
from src.asal.auxils import get_train_data
from src.asal.asp import get_induction_program


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


def nesy_forward_pass(batch, model, sfa, cnn_output_size, with_decay=False):
    training_tensors = torch.stack([
        seq.get_sequence_tensor() for seq in batch.sequences
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


def get_latent_loss(batch, nn_outputs, nn_criterion):
    labeled_points_predictions, labeled_points_labels = [], []
    latent_loss = torch.tensor(0.0)
    for i, seq in enumerate(batch.sequences):
        # seq.set_label(2, 4)  debugging
        labeled_indices = seq.get_labeled_indices()
        labeled_points_predictions.extend([nn_outputs[i][j] for j in labeled_indices])
        labeled_points_labels.extend([torch.tensor(seq.get_image_label(j)).long() for j in labeled_indices])

    if labeled_points_labels:
        labeled_points_predictions = torch.stack(labeled_points_predictions).to(device)
        labeled_points_labels = torch.stack(labeled_points_labels).to(device)
        latent_loss = nn_criterion(labeled_points_predictions, labeled_points_labels)

    return latent_loss


class StatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.target_loss = 0.0
        self.latent_loss = 0.0
        self.seq_labels_actual = []
        self.seq_labels_predicted = []
        self.latent_labels_actual = []
        self.latent_labels_predicted = []

    def update_stats(self, batch, latent_predictions, seq_predictions, target_loss=0.0, latent_loss=0.0):
        self.seq_labels_predicted.extend([s.item() for s in seq_predictions])
        self.seq_labels_actual.extend([seq.get_sequence_label().item() for seq in batch.sequences])
        self.latent_labels_predicted.extend([s.item() for s in latent_predictions])
        self.latent_labels_actual.extend(list(chain.from_iterable([seq.label_trace for seq in batch.sequences])))
        self.target_loss += target_loss
        self.latent_loss += latent_loss


def eval_model(ts: StatsCollector,
               test_loader: DataLoader[SequenceDataset],
               model: torch.nn.Module,
               sfa_dnnf: src.asal_nesy.deepfa.automaton.DeepFA,
               cnn_output_size,
               train_loader_length,
               epoch_num=None):
    def get_score(ts: StatsCollector):
        latent_f1_macro = f1_score(ts.latent_labels_actual, ts.latent_labels_predicted, average="macro")
        f1, tps, fps, fns = get_sequence_stats(ts.seq_labels_predicted, ts.seq_labels_actual)
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
        sc.update_stats(batch, latent_predictions, sequence_predictions)

    # Testing stats:
    test_f1, test_tps, test_fps, test_fns, test_latent_f1 = get_score(sc)

    epoch = f'Epoch {epoch_num}' if epoch_num is not None else ''
    seq_loss = ts.target_loss / len(test_loader)
    latent_loss = ts.latent_loss / len(test_loader)
    loss_msg = f'{(seq_loss + latent_loss):.3f} (seq: {seq_loss:.3f} | latent: {latent_loss:.3f})' if latent_loss > 0 else f'{seq_loss:.3f}'
    logger.info(
        f'{epoch}\nLoss: {loss_msg}, Time: {end_time - ts.start_time:.3f} secs\n'
        f'Train F1: {train_f1:.3f} ({train_tps}, {train_fps}, {train_fns}) | latent: {train_latent_f1:.3f}\n'
        f'Test F1: {test_f1:.3f} ({test_tps}, {test_fps}, {test_fns}) | latent: {test_latent_f1:.3f}')
    # f'Labeled images so far: {len(labeled_images)}')

    model.train()


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
                num_epochs=10):
    def get_image_dataset(seq_list: list[TensorSequence],
                          batch_size, shuffle=True) -> DataLoader[IndividualImageDataset]:
        train_images, train_labels = [], []
        for seq in seq_list:
            images = [seq.get_image(i, j) for i, j in seq.get_image_indices()]
            labels = [seq.get_image_label(i, j) for i, j in seq.get_image_indices()]

            labels = torch.tensor([x for d in labels for x in d.values() ])

            train_images.extend(images)
            train_labels.extend(labels)

        train_dataset = IndividualImageDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader

    sample_sequences = random.sample(train_data.sequences, num_samples)
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


def induce_sfa_simple(args, asp_compilation_program, data=None, existing_sfa=None):
    shuffle = False
    template = Template(args.states, args.tclass)
    train_data = get_train_data(args.train, str(args.tclass), args.batch_size, shuffle=shuffle)

    logger.debug(f'The induction program is:\n{get_induction_program(args, template)}')

    mcts = Asal(args, train_data, template)
    mcts.run_mcts()

    logger.info(blue(f'New SFA:\n{mcts.best_model.show(mode="""simple""")}\n'
                     f'training F1-score: {mcts.best_model.global_performance} '
                     f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})'))

    # logger.info('Compiling guards into NNF...')
    sfa = mcts.best_model

    from src.asal_nesy.neurasal.sfa import compile_sfa

    compiled = compile_sfa(mcts.best_model, asp_compilation_program)
    return compiled, sfa
