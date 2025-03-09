import os
import sys
import torch
import torch.nn as nn
import time

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.cirquits.asp_programs import *
from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.circuit_auxils import model_count_nnf, nnf_map_to_sfa
from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.dsfa_old.mnist_seqs_new import get_data_loaders, get_data_loaders_OOD
from src.asal_nesy.neurasal.utils import *

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device: {device}')

# The automaton that will be used for neuro-symbolic training, in a logic programming representation.
# It represents a regex that is used to generate the positive sequences in our data. The regex dictates
# that a positive sequence is one where an even number larger than 6 is observed, followed at some point
# by an odd number <= than 6, followed by a number <= 3.
# regex: r"\d*(8)\d*(1|3|5)\d*(0|1|2|3)"
automaton = \
    """
    accepting(4).
    transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). 
    transition(2,f(2,3),3). transition(3,f(3,3),3). transition(3,f(3,4),4). transition(4,f(4,4),4).
    f(1,2) :- equals(even,1), equals(gt_6,1).
    f(2,3) :- equals(odd,1), equals(leq_6,1).
    f(3,4) :- equals(leq_3,1).
    f(3,3) :- not f(3,4).
    f(2,2) :- not f(2,3).
    f(4,4) :- #true.
    f(1,1) :- not f(1,2).
    """

# Necessary for compiling the automaton's rules into circuits
query_defs = \
    """
    query(guard,f(1,2)) :- f(1,2).
    query(guard,f(2,3)) :- f(2,3).
    query(guard,f(3,4)) :- f(3,4).
    query(guard,f(1,1)) :- f(1,1).
    query(guard,f(2,2)) :- f(2,2).
    query(guard,f(3,3)) :- f(3,3).
    query(guard,f(4,4)) :- f(4,4).

    #show value/2.
    #show query/2.
    """

if __name__ == "__main__":
    logger.info('Compiling the automaton rules into DNNF circuits...')
    asp_program = mnist_even_odd_learn
    asp_program = asp_program + automaton + query_defs
    sdd_builder = SDDBuilder(asp_program,
                             vars_names=['d'],
                             categorical_vars=['d'],
                             clear_fields=False)
    sdd_builder.build_nnfs()
    circuits = sdd_builder.circuits
    sfa = nnf_map_to_sfa(circuits)
    logger.info(f'\nCompiled SFA: {sfa.transitions}\nStates: {sfa.states}\nSymbols: {sfa.symbols}')

    #-------------------------------#
    # Code for training from here on.
    # -------------------------------#

    pre_train_cnn = True  # Pre-train the CNN on a few labeled images.
    pre_training_size = 10  # num of fully labeled seed sequences.
    num_epochs = 200
    batch_size = 50
    cnn_output_size = 10  # digits num. for MNIST

    # The CNN that recognizes digits from images in the input image sequences.
    model = DigitCNN(out_features=cnn_output_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Training/testing sequences are generated on the fly.
    # NOTE: the sequence generation code is not optimized for long sequences. It currently generates
    # sequences of length 10 (i.e. consisting of 10 images) and if you ask for out-of-distribution data,
    # it generates training sequences of length 10 and testing sequences of length 50. These values are
    # currently hard-coded in the get_data_loaders methods below. The generation won't work for larger
    # sequences (e.g. 100), it needs some work.
    logger.info('Generating training/testing data')
    OOD = False  # Out-of-distribution test data, if true the training/testing seqs are of different size.
    train_loader, test_loader = get_data_loaders_OOD(batch_size=50) if OOD else get_data_loaders(batch_size=50)

    # Optionally pre-train the CNN a little bit on a few labeled images.
    if pre_train_cnn:
        logger.info(f'Pre-training on images from {pre_training_size} sequences')
        # num_samples is number of randomly selected sequences. We pre-train on every image from these seqs.
        pre_train_model(train_loader, test_loader, 10, model, optimizer, num_epochs=100)

    logger.info(f'Training the network alongside the automaton...')

    for epoch in range(num_epochs):
        actual, predicted, actual_latent, predicted_latent = [], [], [], []
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            labels = batch[1].to(device)
            """
            The process_batch method is the core of the forward pass, it is defined in
            
            src.asal_nesy.neurasal.utils. 
            
            Briefly, the forward pass works as follows:
            
            We start from an initial distribution over the automaton's states, e.g. [1, 0, 0, 0] for our 4-state 
            automaton. Note that the distribution vector has 1 in the start state and 0 everywhere else. 
            At each point while processing a sequence, we compute a transition matrix with the probabilities of 
            moving from state to state. These probabilities are obtained from the circuits that correspond to the 
            transition rules of the automaton, which accept as input the probabilities of the CNN predictions 
            (i.e. a prob. distribution over the digits for each image) in a sequence. To obtain the distribution 
            over the automaton's states at the next time point in a sequence, we multiply the previous states 
            distribution vector with the current transition matrix. When we reach the end of the sequence we extract 
            the acceptance probability, i.e. the probability of being in the accepting state and compare it to the 
            ground truth (pos/neg sequence). The sequence is classified as positive if acceptance_probability >= 0.5.
            
            If you need to take a look at the automaton's forward method that does all the above (although I do
            # think it will be necessary), check the forward method of the class Automaton in
            
            src.asal_nesy.deepfa.automaton.py
             
            You can run a purely neural baseline (that uses an LSTM instead of the automaton) in
            
            src.asal_nesy.neurasal.baselines.py
            """

            acceptance_probabilities, act_latent, pred_latent = process_batch(batch, model, sfa,
                                                                              batch_size, cnn_output_size)

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

        test_f1, test_latent_f1_macro, t_tps, t_fps, t_fns = test_model(model, sfa, test_loader,
                                                                        batch_size, cnn_output_size)

        _, train_f1, _, tps, fps, fns, _ = get_stats(predicted, actual)

        logger.info(
            f'Epoch {epoch}\nLoss: {total_loss / len(train_loader):.3f}, Time: {time.time() - start_time:.3f} secs\n'
            f'Train F1: {train_f1:.3f} ({tps}, {fps}, {fns}) | latent: {latent_f1_macro:.3f}\n'
            f'Test F1: {test_f1:.3f} ({t_tps}, {t_fps}, {t_fns}) | latent: {test_latent_f1_macro:.3f}')
