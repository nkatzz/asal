import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.asal_nesy.neurasal.sfa import *
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.dsfa_old.mnist_seqs_new import get_data_loaders, get_data_loaders_OOD
from src.asal_nesy.neurasal.utils import *
from src.args_parser import parse_args
from src.asal_nesy.device import device

if __name__ == "__main__":

    parser = parse_args()
    args = parser.parse_args()

    print(f'Device: {device}')

    max_states = 4
    target_class = 1

    asp_comp_program = mnist_even_odd_learn

    # Learn an SFA from a few initial, fully labeled sequences
    sfa = induce_sfa(args, asp_comp_program)

    pre_train_nn = True  # Pre-train the CNN on a few labeled images.
    pre_training_size = 10  # num of fully labeled seed sequences.
    num_epochs = 200

    # batch size vs lr: bs=50 --> lr=0.01, bs=1 --> lr=0.001
    batch_size = 50
    cnn_output_size = 10  # digits num. for MNIST
    model = DigitCNN(out_features=cnn_output_size)
    # model = SimpleCNN()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 0.001
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()

    logger.info('Generating training/testing data')
    OOD = False  # Out-of-distribution test data, if true the training/testing seqs are of different size.
    train_loader, test_loader = (
        get_data_loaders_OOD(batch_size=batch_size)) if OOD else get_data_loaders(batch_size=batch_size)

    if pre_train_nn:
        logger.info(f'Pre-training on images from {pre_training_size} sequences')
        # num_samples is number of randomly selected sequences. We pre-train on every image from these seqs.
        pre_train_model(train_loader, test_loader, 10, model, optimizer, num_epochs=100)

    logger.info(f'Training the network with the SFA...')

    for epoch in range(num_epochs):
        actual, predicted, actual_latent, predicted_latent = [], [], [], []
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            labels = batch[1].to(device)
            acceptance_probabilities, act_latent, pred_latent = process_batch(batch, model, sfa, cnn_output_size)

            actual_latent.append(act_latent)
            predicted_latent.append(pred_latent)
            sequence_predictions = (acceptance_probabilities >= 0.5)
            predicted.extend(sequence_predictions)
            actual.extend(labels)

            loss = criterion(acceptance_probabilities, labels.float())
            backprop(loss, optimizer)
            total_loss += loss.item()

        actual_latent = torch.cat(actual_latent).numpy()  # Concatenates list of tensors into one
        predicted_latent = torch.cat(predicted_latent).numpy()

        latent_f1_macro = f1_score(actual_latent, predicted_latent, average="macro")
        latent_f1_micro = f1_score(actual_latent, predicted_latent, average="micro")

        test_f1, test_latent_f1_macro, t_tps, t_fps, t_fns = test_model(model, sfa, test_loader, cnn_output_size)

        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        logger.info(
            f'Epoch {epoch}\nLoss: {total_loss / len(train_loader):.3f}, Time: {time.time() - start_time:.3f} secs\n'
            f'Train F1: {train_f1:.3f} ({tps}, {fps}, {fns}) | latent: {latent_f1_macro:.3f}\n'
            f'Test F1: {test_f1:.3f} ({t_tps}, {t_fps}, {t_fns}) | latent: {test_latent_f1_macro:.3f}')
