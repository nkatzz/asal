import os
import sys
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.neurasal.dev_version.mnist_seqs import get_data_loaders
from src.asal_nesy.neurasal.dev_version.utils import *
from src.args_parser import parse_args

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

# Custom collate function to prevent img_ids transposing
def custom_collate_fn(batch):
    tensor_seqs, labels, symb_seqs, seq_ids, image_ids = zip(*batch)
    return (default_collate(tensor_seqs),
            default_collate(labels),
            default_collate(symb_seqs),
            default_collate(seq_ids),
            list(image_ids))

if __name__ == "__main__":

    parser = parse_args()
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Device: {device}')

    max_states = 4
    target_class = 1

    # Learn an SFA from a few initial, fully labeled sequences
    sfa = induce_sfa(args)

    pre_train_nn = False  # Pre-train the CNN on a few labeled images.
    pre_training_size = 10  # num of fully labeled seed sequences.
    num_epochs = 200

    batch_size = 50
    cnn_output_size = 10  # digits num. for MNIST
    model = DigitCNN(out_features=cnn_output_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sequence_criterion = nn.BCELoss()
    cnn_criterion = nn.CrossEntropyLoss()

    logger.info('Generating training/testing data')
    OOD = True
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Apply custom collate function to train_loader
    train_loader.collate_fn = custom_collate_fn

    if pre_train_nn:
        logger.info(f'Pre-training on images from {pre_training_size} sequences')
        pre_train_model(train_loader, test_loader, 10, model, optimizer, num_epochs=100)

    # Active Learning Settings
    num_images_to_label = 50
    labeled_images = set()

    logger.info(f'Training the network with the SFA and Active Learning...')

    for epoch in range(num_epochs):
        actual, predicted, actual_latent, predicted_latent = [], [], [], []
        total_seq_loss = 0.0
        start_time = time.time()

        all_seq_scores = []  # (seq_id, score)
        all_img_scores = []  # (img_id, entropy, true_symbol)
        image_outputs_dict = {}  # {img_id: nn_output_vector}

        for batch in train_loader:
            labels = batch[1].to(device)
            symbolic_sequences = batch[2].to(device)

            (acceptance_probabilities, act_latent,
             pred_latent, seq_ids, img_ids,
             seq_entropy_scores, entropy_per_img, nn_outputs) = process_batch(batch, model, sfa, cnn_output_size)

            batch_size_current = len(img_ids)
            cnn_outputs = []
            cnn_targets = []

            for i in range(batch_size_current):
                seq_id = seq_ids[i]
                seq_img_ids = img_ids[i]
                seq_len = len(seq_img_ids)
                for j in range(seq_len):
                    img_id = seq_img_ids[j]
                    entropy_val = entropy_per_img[i][j]
                    all_img_scores.append((img_id, entropy_val.item(), symbolic_sequences[i][j].item()))
                    image_outputs_dict[img_id] = nn_outputs[i][j]

                    # Collect CNN outputs if labeled
                    if img_id in labeled_images:
                        cnn_outputs.append(nn_outputs[i][j])
                        cnn_targets.append(symbolic_sequences[i][j].item())

                all_seq_scores.append((seq_id, seq_entropy_scores[i].item()))

            actual_latent.append(act_latent)
            predicted_latent.append(pred_latent)
            sequence_predictions = (acceptance_probabilities >= 0.5)
            predicted.extend(sequence_predictions)
            actual.extend(labels)

            # Sequence loss
            seq_loss = sequence_criterion(acceptance_probabilities, labels.float())

            # === Active Learning Step ===
            all_img_scores.sort(key=lambda x: x[1], reverse=True)
            newly_labeled = 0
            for img_id, entropy, true_symbol in all_img_scores:
                if img_id not in labeled_images:
                    labeled_images.add(img_id)
                    newly_labeled += 1
                    if newly_labeled >= num_images_to_label:
                        break

            # CNN loss (if we have labeled images in batch)
            if cnn_outputs:
                cnn_outputs = torch.stack(cnn_outputs).to(device)
                cnn_targets = torch.tensor(cnn_targets).to(device)
                cnn_loss = cnn_criterion(cnn_outputs, cnn_targets)
                total_loss = seq_loss + cnn_loss
            else:
                total_loss = seq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_seq_loss += seq_loss.item()

        actual_latent = torch.cat(actual_latent).numpy()
        predicted_latent = torch.cat(predicted_latent).numpy()

        latent_f1_macro = f1_score(actual_latent, predicted_latent, average="macro")
        latent_f1_micro = f1_score(actual_latent, predicted_latent, average="micro")

        # === Evaluation ===
        test_f1, test_latent_f1_macro, t_tps, t_fps, t_fns = test_model(model, sfa, test_loader, cnn_output_size)
        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        logger.info(
            f'Epoch {epoch}\nLoss: {total_seq_loss / len(train_loader):.3f}, Time: {time.time() - start_time:.3f} secs\n'
            f'Train F1: {train_f1:.3f} ({tps}, {fps}, {fns}) | latent: {latent_f1_macro:.3f}\n'
            f'Test F1: {test_f1:.3f} ({t_tps}, {t_fps}, {t_fns}) | latent: {test_latent_f1_macro:.3f}\n'
            f'Labeled images so far: {len(labeled_images)}')