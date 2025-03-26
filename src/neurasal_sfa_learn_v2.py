import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data._utils.collate import default_collate

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.neurasal.dev_version.mnist_seqs import get_data_loaders
from src.asal_nesy.neurasal.dev_version.utils import *
from src.args_parser import parse_args

from sklearn.metrics import f1_score
import time

# === Utility Functions ===

# Prevent transposing img_ids in DataLoader
def custom_collate_fn(batch):
    tensor_seqs, labels, symb_seqs, seq_ids, image_ids = zip(*batch)
    return (default_collate(tensor_seqs),
            default_collate(labels),
            default_collate(symb_seqs),
            default_collate(seq_ids),
            list(image_ids))

# Convert sequence into logical facts
def sequence_to_facts(seq_id, digit_preds, known_labels, seq_label, seq_entropy):
    facts = []
    for t, digit in enumerate(digit_preds):
        if t in known_labels:
            facts.append(f'seq({seq_id},d({known_labels[t]}),{t}).')
        else:
            facts.append(f'seq({seq_id},d({digit}),{t}).')
    facts.append(f'class({seq_id},{seq_label}).')
    facts.append(f'weight({seq_id},{seq_entropy}).')
    return ' '.join(facts)

# Inverse entropy weight (high weight for low entropy sequences)
def compute_inverse_entropy_weight(entropy_val, cnn_output_size, scaling_factor):
    max_entropy = math.log(cnn_output_size)
    normalized_entropy = entropy_val / max_entropy
    inv_weight = 1 - normalized_entropy
    scaled_weight = int(inv_weight * scaling_factor)
    return scaled_weight

# Entropy based on acceptance probability
def compute_sequence_entropy(acceptance_prob):
    return -acceptance_prob * math.log(acceptance_prob + 1e-8) - (1 - acceptance_prob) * math.log(1 - acceptance_prob + 1e-8)

# === Main Script ===
if __name__ == "__main__":

    parser = parse_args()
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Device: {device}')

    # Hyperparameters & toggles
    num_epochs = 500
    active_learning_frequency = 5
    # With num_images_to_label = 10 and num_sequences_to_select = 50 I keep re-learning the correct SFA,
    # * starting from the correct one *
    # When starting form an incorrect one:
    num_images_to_label = 100
    num_sequences_to_select = 20
    entropy_scaling_factor = 100

    # Weighted sequence ranking
    w_label_density = 2.0
    w_seq_entropy = 1.0
    w_img_entropy = 0.0

    batch_size = 50
    cnn_output_size = 10  # MNIST digits

    pre_train_nn = True  # Pre-train the CNN on a few labeled images.
    pre_training_size = 10  # num of fully labeled seed sequences.

    model = DigitCNN(out_features=cnn_output_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sequence_criterion = nn.BCELoss()
    cnn_criterion = nn.CrossEntropyLoss()

    logger.info('Generating training/testing data')
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    train_loader.collate_fn = custom_collate_fn

    if pre_train_nn:
        logger.info(f'Pre-training on images from {pre_training_size} sequences')
        pre_train_model(train_loader, test_loader, 10, model, optimizer, num_epochs=100)

    logger.info(f'Inducing initial automaton...')
    asp_comp_program = mnist_even_odd_learn
    sfa_dnnf, sfa_asal = induce_sfa(args, asp_comp_program)

    partially_labeled_sequences = {}
    labeled_images = set()

    logger.info(f'Starting CNN + Automaton training with interleaved active learning...')

    for epoch in range(1, num_epochs + 1):
        logger.info(f'Epoch {epoch}/{num_epochs}')

        actual, predicted, actual_latent, predicted_latent = [], [], [], []
        total_seq_loss = 0.0
        start_time = time.time()

        all_seq_scores = []
        all_img_scores = []

        model.train()
        # optimizer.zero_grad()

        for batch in train_loader:
            labels = batch[1].to(device)
            symbolic_sequences = batch[2].to(device)

            (acceptance_probabilities, act_latent,
             pred_latent, seq_ids, img_ids,
             seq_entropy_scores, entropy_per_img, nn_outputs) = process_batch(batch, model, sfa_dnnf, cnn_output_size)

            batch_size_current = len(img_ids)
            cnn_outputs = []
            cnn_targets = []

            for i in range(batch_size_current):
                seq_id = seq_ids[i].item()
                seq_img_ids = img_ids[i]
                seq_len = len(seq_img_ids)
                digit_preds = torch.argmax(nn_outputs[i], dim=1).cpu().tolist()
                if seq_id not in partially_labeled_sequences:
                    inv_weight = compute_inverse_entropy_weight(seq_entropy_scores[i].item(), cnn_output_size,
                                                                entropy_scaling_factor)
                    partially_labeled_sequences[seq_id] = {
                        'digit_preds': digit_preds,
                        'known_labels': {},
                        'seq_label': labels[i].item(),
                        'entropy_weight': inv_weight
                    }
                known_labels = partially_labeled_sequences[seq_id]['known_labels']

                for j in range(seq_len):
                    img_id = seq_img_ids[j]
                    entropy_val = entropy_per_img[i][j]
                    all_img_scores.append((img_id, entropy_val.item(), symbolic_sequences[i][j].item(), seq_id, j))

                all_seq_scores.append((seq_id, acceptance_probabilities[i].item(), len(known_labels), seq_len))

            actual_latent.append(act_latent)
            predicted_latent.append(pred_latent)
            sequence_predictions = (acceptance_probabilities >= 0.5)
            predicted.extend(sequence_predictions)
            actual.extend(labels)

            # Sequence loss
            seq_loss = sequence_criterion(acceptance_probabilities, labels.float())

            # CNN loss on labeled images
            for i in range(batch_size_current):
                seq_id = seq_ids[i].item()
                seq_img_ids = img_ids[i]
                seq_len = len(seq_img_ids)
                known_labels = partially_labeled_sequences[seq_id]['known_labels']

                for j in range(seq_len):
                    if j in known_labels:
                        cnn_outputs.append(nn_outputs[i][j])
                        cnn_targets.append(known_labels[j])

            # sequence_loss_weight = epoch / num_epochs
            sequence_loss_weight = 1.0

            if cnn_outputs:
                cnn_outputs = torch.stack(cnn_outputs).to(device)
                cnn_targets = torch.tensor(cnn_targets).to(device)
                cnn_loss = cnn_criterion(cnn_outputs, cnn_targets)
                total_loss = sequence_loss_weight * seq_loss + cnn_loss
            else:
                total_loss = seq_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_seq_loss += seq_loss.item()

        # Evaluation
        actual_latent = torch.cat(actual_latent).numpy()
        predicted_latent = torch.cat(predicted_latent).numpy()
        latent_f1_macro = f1_score(actual_latent, predicted_latent, average="macro")
        test_f1, test_latent_f1_macro, t_tps, t_fps, t_fns = test_model(model, sfa_dnnf, test_loader, cnn_output_size)
        logger.info(
            f'Epoch {epoch} Loss: {total_seq_loss / len(train_loader):.3f}, F1 Train: {latent_f1_macro:.3f}, F1 Test: {test_f1:.3f}')

        # --------------------
        # Active learning step
        # --------------------
        if epoch % active_learning_frequency == 0:
            logger.info("Active learning step...")

            # Step 1: Rank sequences based on weighted combination
            ranked_sequences = []
            for seq_id, acceptance_prob, labeled_count, seq_len in all_seq_scores:
                label_density = labeled_count / seq_len
                seq_entropy = compute_sequence_entropy(acceptance_prob)
                avg_img_entropy = sum(ent for img_id, ent, _, s_id, _ in all_img_scores if s_id == seq_id) / seq_len
                combined_score = (w_label_density * (1 - label_density)) + (w_seq_entropy * seq_entropy) + (
                            w_img_entropy * avg_img_entropy)
                ranked_sequences.append((seq_id, combined_score, partially_labeled_sequences[seq_id]['seq_label']))

            # Sort sequences descending by score (reverse=True, False for ascending)
            ranked_sequences.sort(key=lambda x: x[1], reverse=False)

            # Step 2: Select top-N positive and negative sequences
            selected_pos, selected_neg = [], []
            for seq_id, score, label in ranked_sequences:
                if label == 1 and len(selected_pos) < num_sequences_to_select:
                    selected_pos.append(seq_id)
                    print('pos', score, label)
                elif label == 0 and len(selected_neg) < num_sequences_to_select:
                    selected_neg.append(seq_id)
                    print('neg', score, label)
                if len(selected_pos) >= num_sequences_to_select and len(selected_neg) >= num_sequences_to_select:
                    break

            logger.info(f"Selected {len(selected_pos)} positive and {len(selected_neg)} negative sequences.")

            # Step 3: Apply gradient-based ELR to select images for labeling
            # Step 3: Gradient-based ELR image selection
            # Step 3: Gradient-based ELR image selection
            # Step 3: Gradient-based ELR image selection
            labeled_this_step = 0
            model.eval()

            for seq_group in [selected_pos, selected_neg]:
                for seq_id in seq_group:
                    known_labels = partially_labeled_sequences[seq_id]['known_labels']
                    seq_digit_preds = partially_labeled_sequences[seq_id]['digit_preds']
                    image_indices = [idx for idx in range(len(seq_digit_preds)) if idx not in known_labels]

                    # Skip fully labeled sequences
                    if not image_indices:
                        continue

                    # Find batch with sequence
                    for batch in train_loader:
                        batch_seq_ids = batch[3]
                        if seq_id in batch_seq_ids:
                            batch_index = batch_seq_ids.tolist().index(seq_id)
                            labels = batch[1].to(device)
                            batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)

                            # Forward pass over sequence (no stacking)
                            seq_len = batch[0].shape[1]
                            logits_seq = []
                            grads_for_images = []
                            for t_idx in range(seq_len):
                                img = batch[0][:, t_idx]
                                logits = model(img, apply_softmax=False, store_output=True)
                                logits.retain_grad()
                                # Register hook to capture gradient
                                logits.register_hook(lambda grad: grads_for_images.append(grad.detach()))
                                logits_seq.append(logits)

                            # Concatenate logits for processing
                            flat_logits = torch.cat(logits_seq, dim=0)  # (batch_size * seq_len, 10)
                            nn_outputs = F.softmax(flat_logits, dim=1)
                            nn_outputs = nn_outputs.view(batch[0].shape[0], seq_len, cnn_output_size)

                            # Acceptance probability
                            output_transposed = nn_outputs.transpose(1, 2)
                            probabilities = {sfa_dnnf.symbols[i]: output_transposed[:, i, :] for i in
                                             range(len(sfa_dnnf.symbols))}
                            labelling_function = create_labelling_function(probabilities, sfa_dnnf.symbols)
                            acceptance_probs = torch.clamp(sfa_dnnf.forward(labelling_function), 0, 1)

                            loss = sequence_criterion(acceptance_probs, labels.float())

                            # Backprop
                            model.zero_grad()
                            loss.backward(retain_graph=True)

                            # Compute grad norms per image
                            grad_norms = []
                            for j in image_indices:
                                grad = grads_for_images[j][batch_index]
                                grad_norms.append((j, torch.norm(grad).item()))

                            grad_norms.sort(key=lambda x: x[1], reverse=True)  # select those with largest grads

                            # Label top-K images
                            k = min(len(grad_norms), num_images_to_label // (2 * num_sequences_to_select))

                            if k == 0:
                                stop = 'stop'

                            for j, _ in grad_norms[:k]:
                                true_label = train_loader.dataset.all_seqs[seq_id][0][j][1]
                                known_labels[j] = true_label
                                labeled_images.add(train_loader.dataset.all_seqs[seq_id][0][j][2])
                                labeled_this_step += 1
                            break

            logger.info(f"Labeled {labeled_this_step} new images.")

            # Step 4: Write sequences to facts and revise automaton
            all_facts = []
            for seq_id in selected_pos + selected_neg:
                seq_data = partially_labeled_sequences[seq_id]
                facts = sequence_to_facts(seq_id, seq_data['digit_preds'], seq_data['known_labels'],
                                          seq_data['seq_label'], seq_data['entropy_weight'])
                all_facts.append(facts)
            nn_seq = '\n'.join(all_facts)
            nn_seq = {0: nn_seq}

            logger.info("Revising SFA...")
            sfa_dnnf, sfa_asal = induce_sfa(args, asp_comp_program, nn_provided_data=nn_seq, existing_sfa=sfa_asal)

logger.info("Training complete.")