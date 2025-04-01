import os
import sys
import torch
import torch.nn as nn
import math
from torch.utils.data._utils.collate import default_collate

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.neurasal.dev_version.mnist_seqs import get_data_loaders
from src.asal_nesy.neurasal.dev_version.utils import *
from src.args_parser import parse_args
from src.asal_nesy.device import device


# Custom collate function to prevent img_ids transposing
def custom_collate_fn(batch):
    tensor_seqs, labels, symb_seqs, seq_ids, image_ids = zip(*batch)
    return (default_collate(tensor_seqs),
            default_collate(labels),
            default_collate(symb_seqs),
            default_collate(seq_ids),
            list(image_ids))


# Convert sequence to logical facts
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


if __name__ == "__main__":

    parser = parse_args()
    args = parser.parse_args()

    print(f'Device: {device}')

    max_states = 4
    target_class = 1

    pre_train_nn = False  # Pre-train the CNN on a few labeled images.
    pre_training_size = 10  # num of fully labeled seed sequences.

    num_epochs = 1000
    active_learning_frequency = 3
    num_images_to_label = 50
    num_sequences_to_select = 20
    entropy_scaling_factor = 100

    batch_size = 50
    cnn_output_size = 10  # digits num. for MNIST

    asp_comp_program = mnist_even_odd_learn

    model = DigitCNN(out_features=cnn_output_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sequence_criterion = nn.BCELoss()
    cnn_criterion = nn.CrossEntropyLoss()

    logger.info('Generating training/testing data')
    OOD = True  # need to add this to the get_data_loaders method.
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Apply custom collate function to train_loader
    train_loader.collate_fn = custom_collate_fn

    if pre_train_nn:
        logger.info(f'Pre-training on images from {pre_training_size} sequences')
        pre_train_model(train_loader, test_loader, 10, model, optimizer, num_epochs=100)

    logger.info(f'Inducing initial automaton...')
    sfa_dnnf, sfa_asal = induce_sfa(args, asp_comp_program)

    partially_labeled_sequences = {}  # seq_id -> {digit_preds, known_labels, seq_label, entropy_weight, seq_entropy_val}
    labeled_images = set()
    labeled_images_dict = {}  # img_id -> label

    logger.info(f'Starting CNN + Automaton training with interleaved active learning...')

    for epoch in range(1, num_epochs + 1):
        logger.info(f'Epoch {epoch}/{num_epochs}')

        actual, predicted, actual_latent, predicted_latent = [], [], [], []
        total_seq_loss = 0.0
        start_time = time.time()

        all_seq_scores = []
        all_img_scores = []
        image_outputs_dict = {}

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
                digit_probs = torch.max(nn_outputs[i],
                                        dim=1).values.cpu().tolist()  # Extract probabilities of predictions

                if seq_id not in partially_labeled_sequences:
                    entropy_val = seq_entropy_scores[i].item()
                    max_entropy = math.log(cnn_output_size)  # CNN output size = 10 for MNIST
                    normalized_entropy = entropy_val / max_entropy
                    inv_weight = 1 - normalized_entropy
                    scaled_weight = int(inv_weight * entropy_scaling_factor)

                    # inv_weight = math.exp(-seq_entropy_scores[i].item() * entropy_scaling_factor)

                    partially_labeled_sequences[seq_id] = {
                        'digit_preds': digit_preds,
                        'digit_probs': digit_probs,
                        'known_labels': {},
                        'seq_label': labels[i].item(),
                        # 'entropy_weight': int(seq_entropy_scores[i].item() * entropy_scaling_factor)
                        'entropy_weight': scaled_weight,
                        'seq_entropy_val': entropy_val
                    }
                known_labels = partially_labeled_sequences[seq_id]['known_labels']

                for j in range(seq_len):
                    img_id = seq_img_ids[j]
                    entropy_val = entropy_per_img[i][j]
                    all_img_scores.append((img_id, entropy_val.item(), symbolic_sequences[i][j].item(), seq_id, j))
                    image_outputs_dict[img_id] = nn_outputs[i][j]

                all_seq_scores.append((seq_id, seq_entropy_scores[i].item()))

            actual_latent.append(act_latent)
            predicted_latent.append(pred_latent)
            sequence_predictions = (acceptance_probabilities >= 0.5)
            predicted.extend(sequence_predictions)
            actual.extend(labels)

            # Sequence loss
            seq_loss = sequence_criterion(acceptance_probabilities, labels.float())

            # CNN loss on all labeled images globally
            for i in range(batch_size_current):
                seq_id = seq_ids[i].item()
                seq_img_ids = img_ids[i]
                seq_len = len(seq_img_ids)
                known_labels = partially_labeled_sequences[seq_id]['known_labels']

                for j in range(seq_len):
                    if j in known_labels:
                        cnn_outputs.append(nn_outputs[i][j])
                        cnn_targets.append(known_labels[j])

            # seq_loss_weight = epoch/(num_epochs * 100000)
            seq_loss_weight = 1.0

            if cnn_outputs:
                cnn_outputs = torch.stack(cnn_outputs).to(device)
                cnn_targets = torch.tensor(cnn_targets).to(device)
                cnn_loss = cnn_criterion(cnn_outputs, cnn_targets)

                total_loss = seq_loss_weight * seq_loss + cnn_loss
                # total_loss = cnn_loss   # just for debugging and experiments without the seq loss...
            else:
                # total_loss = torch.tensor(0.0, requires_grad=True) # seq_loss_weight * seq_loss
                total_loss = seq_loss_weight * seq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_seq_loss += seq_loss.item()

        actual_latent = torch.cat(actual_latent).numpy()
        predicted_latent = torch.cat(predicted_latent).numpy()

        latent_f1_macro = f1_score(actual_latent, predicted_latent, average="macro")
        latent_f1_micro = f1_score(actual_latent, predicted_latent, average="micro")

        test_f1, test_latent_f1_macro, t_tps, t_fps, t_fns = test_model(model, sfa_dnnf, test_loader, cnn_output_size)
        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        logger.info(
            f'Epoch {epoch}\nLoss: {total_seq_loss / len(train_loader):.3f}, Time: {time.time() - start_time:.3f} secs\n'
            f'Train F1: {train_f1:.3f} ({tps}, {fps}, {fns}) | latent: {latent_f1_macro:.3f}\n'
            f'Test F1: {test_f1:.3f} ({t_tps}, {t_fps}, {t_fns}) | latent: {test_latent_f1_macro:.3f}\n'
            f'Labeled images so far: {len(labeled_images)}')

        # Active learning step every active_learning_frequency epochs
        if epoch % active_learning_frequency == 0:
            logger.info("Active learning step...")

            # reverse=True because we're using entropy, so we want sequences with larger entropy first.
            # change it if you use e.g. prediction probability as an uncertainty measure.
            all_seq_scores.sort(key=lambda x: x[1], reverse=True)
            top_seq_ids = [seq_id for seq_id, _ in all_seq_scores[:num_sequences_to_select]]

            avg_entropy = sum([score for seq_id, score in all_seq_scores if seq_id in top_seq_ids]) / len(top_seq_ids)
            logger.info(yellow(f'Average entropy over selected sequences: {avg_entropy:.3f}'))

            filtered_img_scores = [(img_id, entropy, true_symbol, seq_id, img_idx) for
                                   (img_id, entropy, true_symbol, seq_id, img_idx) in all_img_scores if
                                   seq_id in top_seq_ids]
            filtered_img_scores.sort(key=lambda x: x[1], reverse=True)

            newly_labeled = 0
            for img_id, entropy, true_symbol, seq_id, img_idx in filtered_img_scores:
                known_labels = partially_labeled_sequences[seq_id]['known_labels']
                if img_idx not in known_labels:
                    known_labels[img_idx] = true_symbol
                    labeled_images.add(img_id)
                    labeled_images_dict[img_id] = true_symbol
                    newly_labeled += 1
                    if newly_labeled >= num_images_to_label:
                        break

            logger.info(f"Labeled {newly_labeled} new images")

            all_facts = []
            for seq_id, seq_data in partially_labeled_sequences.items():
                #--------------------------------------------------------------
                if len(seq_data['known_labels']) > 0 and seq_id in top_seq_ids:
                    facts = sequence_to_facts(seq_id, seq_data['digit_preds'], seq_data['known_labels'],
                                              seq_data['seq_label'], seq_data['entropy_weight'])

                    digit_probs = [round(prob, 3) for prob in seq_data['digit_probs']]
                    print(seq_id, seq_data['known_labels'], seq_data['seq_label'], seq_data['digit_preds'],
                          digit_probs, seq_data['seq_entropy_val'], seq_data['entropy_weight'])

                    all_facts.append(facts)

            nn_seq = '\n'.join(all_facts)
            nn_seq = {0: nn_seq}

            logger.info("Revising SFA...")
            sfa_dnnf, sfa_asal = induce_sfa(args, asp_comp_program, data=nn_seq, existing_sfa=sfa_asal)

    logger.info("Training complete.")
